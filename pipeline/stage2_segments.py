"""Stage 2 — cut single-speaker chunks from each leg with Silero VAD.

There are no transcripts yet (Gemini labels come in stage 3), so segmentation
is purely acoustic. Per kept call and per non-silent leg:

- decode the leg to 16 kHz mono (legs may differ in length; they share t=0)
- run Silero VAD, merge speech regions separated by <= max_merge_gap into
  chunk groups, cap at max_chunk_dur (an oversize single region is force-split,
  flagged boundary_clipped); groups shorter than min_chunk_dur are dropped —
  no tiny chunks
- bleed filter: telephony legs carry a faint echo of the far party. For each
  chunk, the speech RMS on this leg is compared to the other leg's RMS over the
  same samples. If this leg is not at least bleed_db_min louder, the "speech"
  VAD found is most likely the far party's echo -> chunk dropped.

Output:
  work_dir/chunks/<call_id>/<chunk_id>.flac   (16 kHz mono)
  work_dir/chunks.jsonl                       (one row per chunk)
  work_dir/calls_done.jsonl                   (resume markers)
CPU only, multiprocess, resumable.
"""
import argparse
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from .common import (JsonlWriter, add_common_args, decode_mono, done_ids,
                     load_config_args, read_jsonl, rms)

SR = 16000

CFG = None
VAD = None
_get_speech_ts = None


def _init(cfg):
    global CFG, VAD, _get_speech_ts
    import torch
    torch.set_num_threads(1)
    from silero_vad import load_silero_vad, get_speech_timestamps
    CFG = cfg
    VAD = load_silero_vad()
    _get_speech_ts = get_speech_timestamps


def vad_regions(x16, cfg):
    import torch
    ts = _get_speech_ts(
        torch.from_numpy(x16),
        VAD,
        sampling_rate=SR,
        threshold=cfg["vad"]["threshold"],
        min_speech_duration_ms=cfg["vad"]["min_speech_duration_ms"],
        min_silence_duration_ms=cfg["vad"]["min_silence_duration_ms"],
        speech_pad_ms=cfg["vad"]["speech_pad_ms"],
    )
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def build_groups(regions, cfg):
    """Merge VAD regions into chunk groups; force-split oversize single regions."""
    groups, cur = [], None
    for r0, r1 in regions:
        if r1 - r0 > cfg["max_chunk_dur"]:
            if cur:
                groups.append(cur)
                cur = None
            n = math.ceil((r1 - r0) / cfg["max_chunk_dur"])
            step = (r1 - r0) / n
            for k in range(n):
                groups.append({"t0": r0 + k * step, "t1": r0 + (k + 1) * step,
                               "clipped": True, "n_regions": 1})
            continue
        if cur and (r0 - cur["t1"]) <= cfg["max_merge_gap"] \
                and (r1 - cur["t0"]) <= cfg["max_chunk_dur"]:
            cur["t1"] = r1
            cur["n_regions"] += 1
        else:
            if cur:
                groups.append(cur)
            cur = {"t0": r0, "t1": r1, "clipped": False, "n_regions": 1}
    if cur:
        groups.append(cur)
    return groups


def overlap_len(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def speech_mask(regions, c0, c1):
    i0, i1 = int(c0 * SR), int(c1 * SR)
    mask = np.zeros(i1 - i0, dtype=bool)
    for r0, r1 in regions:
        a = max(i0, int(r0 * SR)) - i0
        b = min(i1, int(r1 * SR)) - i0
        if b > a:
            mask[a:b] = True
    return i0, i1, mask


def snr_db(x16, regions, c0, c1):
    """Speech RMS vs in-chunk non-speech RMS, on the speaker's own leg."""
    i0, i1, mask = speech_mask(regions, c0, c1)
    chunk = x16[i0:i1]
    speech, noise = chunk[mask[:len(chunk)]], chunk[~mask[:len(chunk)]]
    if len(speech) == 0:
        return -10.0
    if len(noise) < int(0.2 * SR):
        return 40.0
    val = 20.0 * math.log10((rms(speech) + 1e-9) / (rms(noise) + 1e-9))
    return float(min(max(val, -10.0), 60.0))


def bleed_db(xa, xb, regions, c0, c1):
    """How much louder this leg's speech is than the other leg, same samples.

    Strongly negative = the other party is louder there = the VAD fired on
    echo/bleed, not on this speaker. +40 when the other leg has no audio there.
    """
    if xb is None or len(xb) == 0:
        return 40.0
    i0, i1, mask = speech_mask(regions, c0, c1)
    chunk_a = xa[i0:i1]
    a = chunk_a[mask[: len(chunk_a)]]
    chunk_b = xb[i0:i1]
    b = chunk_b[mask[: len(chunk_b)]]
    if len(a) == 0 or len(b) == 0:
        return 40.0
    val = 20.0 * math.log10((rms(a) + 1e-9) / (rms(b) + 1e-9))
    return float(min(max(val, -40.0), 40.0))


def process_call(job):
    import soundfile as sf

    cfg = CFG["stage2"]
    audio_dir = CFG["paths"]["audio_dir"]
    out_root = os.path.join(CFG["paths"]["work_dir"], "chunks")
    call_dir = os.path.join(out_root, job["call_id"])

    decoded = {}
    for leg in ("r", "t"):
        li = job["legs"].get(leg)
        if li:
            decoded[leg] = decode_mono(os.path.join(audio_dir, li["path"]), sr=SR)

    drops = Counter()
    rows = []
    made_dir = False
    for leg, x16 in decoded.items():
        if job["legs"][leg]["silent"]:
            continue
        other = decoded.get("t" if leg == "r" else "r")
        audio_dur = len(x16) / SR
        regions = vad_regions(x16, cfg)
        if not regions:
            drops["leg:no_speech"] += 1
            continue
        groups = build_groups(regions, cfg)

        for gi, g in enumerate(groups):
            c0 = max(0.0, g["t0"] - cfg["boundary_pad"])
            c1 = min(audio_dur, g["t1"] + cfg["boundary_pad"])
            # padding must not reach into a neighbouring group's speech
            if gi > 0:
                c0 = max(c0, (groups[gi - 1]["t1"] + g["t0"]) / 2)
            if gi + 1 < len(groups):
                c1 = min(c1, (g["t1"] + groups[gi + 1]["t0"]) / 2)
            dur = c1 - c0
            if dur < cfg["min_chunk_dur"]:
                drops["chunk:too_short"] += 1
                continue

            cov = sum(overlap_len(r0, r1, c0, c1) for r0, r1 in regions) / dur
            if cov < cfg["min_speech_coverage"]:
                drops["chunk:low_speech_coverage"] += 1
                continue
            bdb = bleed_db(x16, other, regions, c0, c1)
            if bdb < cfg["bleed_db_min"]:
                drops["chunk:bleed"] += 1
                continue

            chunk_id = f"{job['call_id']}_{leg}_{gi:03d}"
            if not made_dir:
                os.makedirs(call_dir, exist_ok=True)
                made_dir = True
            cut = x16[int(c0 * SR): int(c1 * SR)]
            rel = os.path.join("chunks", job["call_id"], chunk_id + ".flac")
            sf.write(os.path.join(call_dir, chunk_id + ".flac"),
                     (np.clip(cut, -1.0, 1.0) * 32767).astype(np.int16), SR)

            rows.append({
                "chunk_id": chunk_id,
                "call_id": job["call_id"],
                "extension": job["extension"],
                "channel": leg,
                "path": rel,
                "t0": round(c0, 3),
                "t1": round(c1, 3),
                "duration": round(dur, 3),
                "n_regions": g["n_regions"],
                "snr_db": round(snr_db(x16, regions, c0, c1), 2),
                "bleed_db": round(bdb, 2),
                "speech_coverage": round(min(cov, 1.0), 3),
                "boundary_clipped": int(g["clipped"]),
            })
    return job["call_id"], rows, drops


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config_args(args)
    work = cfg["paths"]["work_dir"]

    done = done_ids(os.path.join(work, "calls_done.jsonl"), "call_id")
    manifest = [r for r in read_jsonl(os.path.join(work, "manifest.jsonl")) if r["keep"]]
    jobs = sorted((r for r in manifest if r["call_id"] not in done),
                  key=lambda r: r["call_id"])
    if args.limit:
        jobs = jobs[:args.limit]
    if done:
        print(f"resume: {len(done)} calls already done, {len(jobs)} to process")
    if not jobs:
        print("nothing to do")
        return

    chunk_writer = JsonlWriter(os.path.join(work, "chunks.jsonl"))
    done_writer = JsonlWriter(os.path.join(work, "calls_done.jsonl"))
    workers = cfg["runtime"]["num_workers"] or max(1, (os.cpu_count() or 2) - 1)
    stats, n_chunks, total_dur = Counter(), 0, 0.0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(process_call, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="stage2"):
            call_id, rows, drops = fut.result()
            for row in rows:
                chunk_writer.write(row)
                n_chunks += 1
                total_dur += row["duration"]
            done_writer.write({"call_id": call_id, "n_chunks": len(rows)})
            stats.update(drops)
    chunk_writer.close()
    done_writer.close()
    print(f"stage2 done: {n_chunks} chunks, {total_dur / 3600:.1f} h of audio")
    print("drops:", dict(stats))
    print("inspect before spending API money:  python -m pipeline.viewer --config",
          args.config)


if __name__ == "__main__":
    main()
