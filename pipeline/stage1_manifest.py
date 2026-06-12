"""Stage 1 — inventory the per-leg recordings and pair them into calls.

Source layout: <audio_dir>/<subfolder>/external-<ext>-<line>-<date>-<time>-<uid>_{r,t}.wav
Each file is one leg of a call (8 kHz mono PCM): `_r` = received audio,
`_t` = transmitted audio — one speaker per file.

Per call this stage:
- pairs the two legs by filename stem (a missing leg is kept, just flagged)
- measures duration and RMS per leg; near-silent legs are flagged so stage 2
  skips them (they still serve as the bleed reference for the other leg)
- correlates the two legs' energy envelopes — near-identical legs mean the
  recorder duplicated a mixed signal into both files (both speakers in one
  channel), which breaks the single-speaker assumption -> call dropped

Output: work_dir/manifest.jsonl (one row per call)
CPU only, parallel.
"""
import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from .common import (DRY_RUN_CALLS, JsonlWriter, add_common_args, decode_mono,
                     done_ids, envelope, load_config_args, parse_leg_filename,
                     rms, safe_corr)

SR = 8000  # decode legs at the native telephony rate here; stage 2 works at 16 k


def scan_calls(audio_dir):
    """{call_id: {"meta": ..., "legs": {"r": relpath, "t": relpath}}}"""
    calls = defaultdict(lambda: {"meta": None, "legs": {}})
    for root, _, files in os.walk(audio_dir):
        for name in sorted(files):
            if not name.endswith(".wav"):
                continue
            call_id, leg, meta = parse_leg_filename(name)
            rel = os.path.relpath(os.path.join(root, name), audio_dir)
            c = calls[call_id]
            c["meta"] = c["meta"] or meta
            c["legs"][leg or "r"] = rel
    return dict(calls)


CFG = None


def _init(cfg):
    global CFG
    CFG = cfg


def process_call(job):
    cfg = CFG["stage1"]
    call_id, meta, legs = job["call_id"], job["meta"], job["legs"]
    audio_dir = CFG["paths"]["audio_dir"]

    decoded, info = {}, {}
    for leg, rel in legs.items():
        x = decode_mono(os.path.join(audio_dir, rel), sr=SR)
        decoded[leg] = x
        r = rms(x)
        info[leg] = {
            "path": rel,
            "dur": round(len(x) / SR, 2),
            "rms": round(r, 5),
            "silent": bool(r < cfg["silent_rms"] or len(x) == 0),
        }

    env_corr = 0.0
    if "r" in decoded and "t" in decoded and not info["r"]["silent"] and not info["t"]["silent"]:
        ea = envelope(decoded["r"], SR // 50)  # 50 Hz energy envelope
        eb = envelope(decoded["t"], SR // 50)
        m = min(len(ea), len(eb))
        env_corr = round(safe_corr(ea[:m], eb[:m]), 3)

    drop = None
    max_dur = max((i["dur"] for i in info.values()), default=0.0)
    if max_dur < cfg["min_call_dur"]:
        drop = "too_short"
    elif all(i["silent"] for i in info.values()):
        drop = "all_legs_silent"
    elif env_corr > cfg["env_corr_max"]:
        drop = "legs_duplicated"

    return {
        "call_id": call_id,
        "folder": os.path.dirname(next(iter(legs.values()))),
        "extension": meta["extension"],
        "line": meta["line"],
        "date": meta["date"],
        "time": meta["time"],
        "legs": {leg: info.get(leg) for leg in ("r", "t")},
        "env_corr": env_corr,
        "keep": drop is None,
        "drop_reason": drop,
    }


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config_args(args)
    work = cfg["paths"]["work_dir"]
    out_path = os.path.join(work, "manifest.jsonl")

    calls = scan_calls(cfg["paths"]["audio_dir"])
    done = done_ids(out_path, "call_id")
    todo = sorted(set(calls) - done)
    limit = args.limit or (DRY_RUN_CALLS if args.dry_run else 0)
    if limit:
        todo = todo[:limit]
    print(f"calls found: {len(calls)} | already done: {len(done)} | to process: {len(todo)}")
    if not todo:
        return

    jobs = [{"call_id": cid, **calls[cid]} for cid in todo]
    writer = JsonlWriter(out_path)
    workers = cfg["runtime"]["num_workers"] or max(1, (os.cpu_count() or 2) - 1)
    stats = Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(process_call, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="stage1"):
            row = fut.result()
            writer.write(row)
            stats["keep" if row["keep"] else f"drop:{row['drop_reason']}"] += 1
            for leg in ("r", "t"):
                if row["legs"][leg] is None:
                    stats[f"missing_leg:{leg}"] += 1
                elif row["legs"][leg]["silent"]:
                    stats[f"silent_leg:{leg}"] += 1
    writer.close()
    print("stage1 done:", dict(stats))


if __name__ == "__main__":
    main()
