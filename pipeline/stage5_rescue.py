"""Stage 5 — rescue chunks the alignment gate flagged, before dropping them.

A low alignment score usually means one of: speech clipped at a chunk
boundary, a Gemini mistake on hard audio, or genuinely garbled audio. The
rescue distinguishes them at the cost of ONE extra API call per flagged chunk:

Phase A (CPU + network, resumable):
  - if the weak words sit at the chunk edges (or alignment failed outright),
    re-cut the chunk with expand_s extra context per side — bounded so it
    never reaches into a neighbouring chunk of the same leg
  - re-transcribe the (possibly re-cut) audio at rescue temperature
    -> work_dir/rescue_retry.jsonl

Phase B (GPU, resumable):
  - align the retry transcript (and, when the audio was re-cut, the original
    transcript on the new audio); keep whichever aligns better
  - retry_cer = CER between the two transcripts (self-consistency signal:
    stable -> trustworthy, divergent -> the model is guessing)
    -> work_dir/rescue.jsonl

Stage 6 then picks the best of (original, rescue) per chunk; chunks that fail
both stay dropped.
"""
import argparse
import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from .aligner import align_text, load_aligner
from .common import (JsonlWriter, add_common_args, cer, decode_mono,
                     done_ok_ids, load_config_args, normalize_train,
                     read_chunks, read_dict_jsonl, read_jsonl)
from .openrouter import transcribe_items

SR = 16000


def read_sharded(work, prefix):
    import glob
    out = {}
    for path in sorted(glob.glob(os.path.join(work, f"{prefix}.*.jsonl"))):
        for r in read_jsonl(path):
            out[r["chunk_id"]] = r
    return out


def needs_rescue(al, s5):
    if not al.get("ok"):
        return True
    return (al["ctc_align_score"] < s5["trigger_align_score"]
            or al["ctc_align_min_word"] < s5["trigger_min_word"])


def new_bounds(c, al, s5, neighbors, leg_dur):
    """(t0, t1, expanded) — widen the cut where the alignment looks clipped."""
    scores = al.get("word_scores") or []
    failed = not al.get("ok")
    left = failed or c["boundary_clipped"] or (bool(scores) and scores[0] < s5["edge_conf"])
    right = failed or c["boundary_clipped"] or (bool(scores) and scores[-1] < s5["edge_conf"])
    t0, t1 = c["t0"], c["t1"]
    if left:
        t0 = max(0.0, t0 - s5["expand_s"])
    if right:
        t1 = min(leg_dur, t1 + s5["expand_s"])
    prev_t1, next_t0 = neighbors
    if prev_t1 is not None:
        t0 = max(t0, prev_t1 + 0.02)
    if next_t0 is not None:
        t1 = min(t1, next_t0 - 0.02)
    if t1 - t0 > 29.0:  # stay under Whisper's 30 s window
        t1 = t0 + 29.0
    expanded = (c["t0"] - t0 > 0.05) or (t1 - c["t1"] > 0.05)
    return t0, t1, expanded


def phase_a(cfg, work, candidates, chunks, manifest):
    s5 = cfg["stage5"]
    orc = cfg["openrouter"]
    retry_path = os.path.join(work, "rescue_retry.jsonl")
    done = done_ok_ids(retry_path)
    todo = [cid for cid in candidates if cid not in done]
    if not todo:
        return
    print(f"phase A: {len(todo)} chunks to re-cut/re-transcribe")

    by_leg = defaultdict(list)
    for cid, c in chunks.items():
        by_leg[(c["call_id"], c["channel"])].append(c)
    for legs in by_leg.values():
        legs.sort(key=lambda c: c["t0"])

    import soundfile as sf
    todo_set = set(todo)
    jobs = []
    for key, legs in sorted(by_leg.items()):
        if not any(c["chunk_id"] in todo_set for c in legs):
            continue
        call_id, channel = key
        m = manifest.get(call_id)
        leg_info = (m or {}).get("legs", {}).get(channel)
        leg_audio = None
        for i, c in enumerate(legs):
            if c["chunk_id"] not in todo_set:
                continue
            al = candidates[c["chunk_id"]]
            neighbors = (legs[i - 1]["t1"] if i > 0 else None,
                         legs[i + 1]["t0"] if i + 1 < len(legs) else None)
            leg_dur = leg_info["dur"] if leg_info else c["t1"]
            t0, t1, expanded = new_bounds(c, al, s5, neighbors, leg_dur)
            rel = c["path"]
            if expanded and leg_info:
                if leg_audio is None:
                    leg_audio = decode_mono(
                        os.path.join(cfg["paths"]["audio_dir"], leg_info["path"]), sr=SR)
                cut = leg_audio[int(t0 * SR): int(t1 * SR)]
                rel = os.path.join("chunks", call_id, c["chunk_id"] + "_x.flac")
                sf.write(os.path.join(work, rel),
                         (np.clip(cut, -1.0, 1.0) * 32767).astype(np.int16), SR)
            else:
                t0, t1, expanded = c["t0"], c["t1"], False
            jobs.append({
                "abs_path": os.path.join(work, rel),
                "audio_sec": round(t1 - t0, 3),
                "echo": {"chunk_id": c["chunk_id"], "pass": "retry", "path": rel,
                         "t0": round(t0, 3), "t1": round(t1, 3),
                         "duration": round(t1 - t0, 3), "expanded": int(expanded)},
            })

    writer = JsonlWriter(retry_path)

    def on_row(row):
        if row.get("ok") and not row.get("no_speech"):
            row["text"] = normalize_train(row["text_raw"])
        elif row.get("ok"):
            row["text"] = ""
        writer.write(row)

    model = s5["model"] or orc["model"]
    state = transcribe_items(jobs, orc, model, s5["temperature"],
                             s5["max_cost_usd"], on_row)
    writer.close()
    print(f"phase A done: {state['n_ok']} ok, {state['n_err']} errors, "
          f"${state['cost']:.2f} spent")
    if state["stop_reason"]:
        print(f"STOPPED EARLY: {state['stop_reason']} — re-run to continue")


def phase_b(cfg, work, candidates, chunks, transcripts):
    lang = cfg["stage4"]["language"]
    retries = read_dict_jsonl(os.path.join(work, "rescue_retry.jsonl"))
    out_path = os.path.join(work, "rescue.jsonl")
    done = done_ok_ids(out_path)  # retry rows whose last attempt errored (e.g. old crashes)
    todo = [cid for cid in sorted(candidates)
            if cid not in done and retries.get(cid, {}).get("ok")]
    if not todo:
        print("phase B: nothing to align")
        return
    print(f"phase B: {len(todo)} chunks to align/choose")

    import soundfile as sfile
    handle = load_aligner(cfg)
    print(f"alignment backend: {handle[3]}")

    writer = JsonlWriter(out_path)
    n_ok, n_retry_chosen = 0, 0
    for cid in tqdm(todo, desc="stage5b"):
        c, al, rt = chunks[cid], candidates[cid], retries[cid]
        primary_text = transcripts[cid]["text"]
        wav, _ = sfile.read(os.path.join(work, rt["path"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        options = []
        if rt["text"] and not rt.get("no_speech"):
            res = align_text(handle, wav, rt["text"], lang)
            if res["ok"]:
                options.append(("retry", rt["text"], rt["path"],
                                rt["t0"], rt["t1"], rt["duration"], res))
        if rt["expanded"]:
            res = align_text(handle, wav, primary_text, lang)
            if res["ok"]:
                options.append(("primary", primary_text, rt["path"],
                                rt["t0"], rt["t1"], rt["duration"], res))
        elif al.get("ok"):
            options.append(("primary", primary_text, c["path"],
                            c["t0"], c["t1"], c["duration"], al))

        retry_cer = round(cer(primary_text, rt["text"]), 4) if rt["text"] else 1.0
        row = {"chunk_id": cid, "ok": False, "chosen": None, "retry_cer": retry_cer,
               "primary_model": transcripts[cid].get("model"),
               "retry_model": rt.get("model"), "expanded": rt["expanded"],
               "error": None}
        if options:
            options.sort(key=lambda o: (o[6]["ctc_align_score"], o[0] == "primary"),
                         reverse=True)
            src, text, path, t0, t1, dur, res = options[0]
            row.update({
                "ok": True, "chosen": src, "text": text, "path": path,
                "t0": t0, "t1": t1, "duration": dur,
                "ctc_align_score": res["ctc_align_score"],
                "ctc_align_min_word": res["ctc_align_min_word"],
                "first_word_start": res["first_word_start"],
                "last_word_end": res["last_word_end"],
            })
            n_ok += 1
            n_retry_chosen += int(src == "retry")
        else:
            row["error"] = "no_alignable_candidate"
        writer.write(row)
    writer.close()
    print(f"phase B done: {n_ok}/{len(todo)} rescued ok "
          f"({n_retry_chosen} chose the retry transcript)")


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    args = ap.parse_args()
    cfg = load_config_args(args)
    if args.device:
        cfg["runtime"]["device"] = args.device
    work = cfg["paths"]["work_dir"]
    s5 = cfg["stage5"]

    chunks = read_chunks(work)
    transcripts = read_dict_jsonl(os.path.join(work, "transcripts.jsonl"))
    aligns = read_sharded(work, "align")
    manifest = {r["call_id"]: r for r in read_jsonl(os.path.join(work, "manifest.jsonl"))}

    candidates = {}
    for cid, al in aligns.items():
        tr = transcripts.get(cid)
        if not tr or not tr.get("ok") or tr.get("no_speech") or not tr.get("text"):
            continue
        if needs_rescue(al, s5):
            candidates[cid] = al
    if args.limit:
        candidates = {cid: candidates[cid] for cid in sorted(candidates)[:args.limit]}
    print(f"aligned: {len(aligns)} | flagged for rescue: {len(candidates)} "
          f"(score < {s5['trigger_align_score']} or min word < {s5['trigger_min_word']})")
    if not candidates:
        return

    phase_a(cfg, work, candidates, chunks, manifest)
    phase_b(cfg, work, candidates, chunks, transcripts)


if __name__ == "__main__":
    main()
