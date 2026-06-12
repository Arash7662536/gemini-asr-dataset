"""Metric distributions over all chunks BEFORE final filtering.

Run after any stage to see how far the pipeline got and whether the default
thresholds in config.yaml make sense — e.g. if p50 of ctc_align_score is 0.55,
a 0.40 cutoff keeps most rows; if it is 0.35, the cutoff is doing real work
and is worth a listen (pipeline.viewer).
"""
import argparse
import os

import numpy as np

from .common import add_common_args, load_config_args, read_chunks, read_dict_jsonl
from .stage5_rescue import read_sharded


def pct_line(name, vals):
    if not vals:
        return f"{name:22s}  (no data)"
    v = np.asarray(vals, dtype=np.float64)
    ps = np.percentile(v, [5, 10, 25, 50, 75, 90, 95])
    return (f"{name:22s}  n={len(v):7d}  "
            + "  ".join(f"p{p}={x:.3f}" for p, x in zip([5, 10, 25, 50, 75, 90, 95], ps)))


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config_args(args)
    work = cfg["paths"]["work_dir"]

    chunks = read_chunks(work)
    transcripts = read_dict_jsonl(os.path.join(work, "transcripts.jsonl"))
    aligns = read_sharded(work, "align")
    rescues = read_dict_jsonl(os.path.join(work, "rescue.jsonl"))

    total_h = sum(c["duration"] for c in chunks.values()) / 3600
    print(f"chunks: {len(chunks)} ({total_h:.1f} h) | transcripts: {len(transcripts)} "
          f"| align: {len(aligns)} | rescue: {len(rescues)}\n")

    print(pct_line("duration_sec", [c["duration"] for c in chunks.values()]))
    print(pct_line("snr_db", [c["snr_db"] for c in chunks.values()]))
    print(pct_line("bleed_db", [c["bleed_db"] for c in chunks.values()]))
    print(pct_line("speech_coverage", [c["speech_coverage"] for c in chunks.values()]))

    if transcripts:
        ok = [t for t in transcripts.values() if t.get("ok")]
        n_ns = sum(1 for t in ok if t.get("no_speech"))
        n_rep = sum(1 for t in ok if t.get("repetition"))
        cost = sum(t.get("cost_usd") or 0.0 for t in transcripts.values())
        print(f"\ntranscripts: {len(ok)} ok ({n_ns} no_speech, {n_rep} repetition-flagged), "
              f"{len(transcripts) - len(ok)} errored | spent ${cost:.2f}")
        rates = [len(t['text'].replace(' ', '')) / chunks[cid]['duration']
                 for cid, t in transcripts.items()
                 if t.get('text') and cid in chunks]
        print(pct_line("char_rate", rates))

    ok_al = [a for a in aligns.values() if a.get("ok")]
    if aligns:
        print()
        print(pct_line("ctc_align_score", [a["ctc_align_score"] for a in ok_al]))
        print(pct_line("ctc_align_min_word", [a["ctc_align_min_word"] for a in ok_al]))
        print(f"align failures: {sum(1 for a in aligns.values() if not a.get('ok'))}")

    if rescues:
        ok_rs = [r for r in rescues.values() if r.get("ok")]
        n_retry = sum(1 for r in ok_rs if r["chosen"] == "retry")
        cost = sum(t.get("cost_usd") or 0.0
                   for t in read_dict_jsonl(os.path.join(work, "rescue_retry.jsonl")).values())
        print(f"\nrescue: {len(ok_rs)}/{len(rescues)} ok ({n_retry} chose retry) | "
              f"retry pass cost ${cost:.2f}")
        print(pct_line("retry_cer", [r["retry_cer"] for r in rescues.values()
                                     if r.get("retry_cer") is not None]))
        print(pct_line("rescued_align_score", [r["ctc_align_score"] for r in ok_rs]))

    f = cfg["stage6"]["filters"]
    if ok_al:
        best = {a["chunk_id"]: a["ctc_align_score"] for a in ok_al}
        for r in rescues.values():
            if r.get("ok"):
                best[r["chunk_id"]] = max(best.get(r["chunk_id"], 0.0),
                                          r["ctc_align_score"])
        surv = [cid for cid, s in best.items() if s >= f["ctc_align_score_min"]]
        h = sum(chunks[cid]["duration"] for cid in surv if cid in chunks) / 3600
        print(f"\nwith ctc_align_score_min={f['ctc_align_score_min']}: "
              f"~{len(surv)} rows, ~{h:.1f} h would survive (before other filters)")


if __name__ == "__main__":
    main()
