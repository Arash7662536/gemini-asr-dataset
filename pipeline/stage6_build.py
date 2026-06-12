"""Stage 6 — join everything, apply quality thresholds, write the Arrow dataset.

Final schema:
  audio      : Audio(sampling_rate=16000), flac bytes embedded (self-contained)
  text       : normalized Persian transcript
  model      : the model id that produced the chosen transcript
  provenance : compact json — primary/retry model, which pass was chosen,
               number of passes, retry CER (how the label was made, per chunk)
  metrics    : list of {name, value} — every metric used for filtering, so the
               dataset can be re-filtered later without re-running anything

Per chunk the best of (original transcript, rescue result) is used — "best" =
higher CTC alignment score, original wins ties. Splits hash on the agent
extension (or call id, see split_by), so an agent's voice never crosses splits.

Output: work_dir/dataset/ (datasets.DatasetDict.save_to_disk, Arrow format)
        work_dir/build_report.json, work_dir/rows_index.jsonl (audit trail)
CPU only.
"""
import argparse
import hashlib
import io
import json
import os
from collections import Counter, defaultdict

from .common import add_common_args, has_repetition, latin_ratio, \
    load_config_args, read_chunks, read_dict_jsonl
from .stage5_rescue import read_sharded

METRIC_ORDER = [
    "ctc_align_score",
    "ctc_align_min_word",
    "retry_cer",        # -1 when no rescue pass was made
    "snr_db",
    "bleed_db",
    "speech_coverage",
    "char_rate",
    "boundary_clipped",
    "n_passes",
    "duration_sec",
]


def split_of(key, val_frac, test_frac, salt):
    h = hashlib.md5(f"{salt}:{key}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "validation"
    return "train"


def effective(c, tr, al, rs):
    """Best usable (text, path, duration, align, source) for a chunk, or None."""
    options = []
    if tr and tr.get("ok") and tr.get("text") and al and al.get("ok"):
        options.append({"source": "primary", "text": tr["text"], "path": c["path"],
                        "duration": c["duration"], "align": al,
                        "model": tr.get("model")})
    if rs and rs.get("ok"):
        options.append({"source": f"rescue_{rs['chosen']}", "text": rs["text"],
                        "path": rs["path"], "duration": rs["duration"],
                        "align": {k: rs[k] for k in
                                  ("ctc_align_score", "ctc_align_min_word",
                                   "first_word_start", "last_word_end")},
                        "model": rs.get("retry_model") if rs["chosen"] == "retry"
                                 else (tr or {}).get("model")})
    if not options:
        return None
    options.sort(key=lambda o: (o["align"]["ctc_align_score"],
                                o["source"] == "primary"), reverse=True)
    return options[0]


def drop_reason(c, tr, al, eff, f):
    if tr is None or not tr.get("ok"):
        return "missing_transcript"
    if tr.get("no_speech"):
        return "no_speech"
    if al is None:
        return "missing_align"
    if eff is None:
        return "align_failed"
    a = eff["align"]
    if a["ctc_align_score"] < f["ctc_align_score_min"]:
        return "low_align_score"
    if a["ctc_align_min_word"] < f["ctc_align_min_word_min"]:
        return "low_min_word_score"
    if f.get("retry_cer_max") is not None and eff.get("retry_cer", -1) > f["retry_cer_max"]:
        return "high_retry_cer"
    if f.get("snr_db_min") is not None and c["snr_db"] < f["snr_db_min"]:
        return "low_snr"
    if f.get("bleed_db_min") is not None and c["bleed_db"] < f["bleed_db_min"]:
        return "bleed"
    if not (f["min_duration"] <= eff["duration"] <= f["max_duration"]):
        return "duration"
    n_chars = len(eff["text"].replace(" ", ""))
    if not (f["min_char_rate"] <= n_chars / eff["duration"] <= f["max_char_rate"]):
        return "char_rate"
    if latin_ratio(eff["text"]) > f["max_latin_ratio"]:
        return "latin"
    if f.get("drop_repetition") and has_repetition(eff["text"]):
        return "repetition"
    return None


def trimmed_bounds(eff, pad):
    """Trim chunk to aligned speech (drops leading/trailing bleed and silence)."""
    a = eff["align"]
    dur = eff["duration"]
    if a.get("first_word_start") is None or a.get("last_word_end") is None:
        return None
    t0 = max(0.0, a["first_word_start"] - pad)
    t1 = min(dur, a["last_word_end"] + pad)
    if t1 - t0 < 0.5 or t1 <= t0:
        return None
    if t0 < 0.15 and dur - t1 < 0.15:  # only bother when it removes something
        return None
    return t0, t1


def gen_rows(rows, work_dir, trim, trim_pad):
    import soundfile as sfile

    for r in rows:
        eff = r["eff"]
        path = os.path.join(work_dir, eff["path"])
        bounds = trimmed_bounds(eff, trim_pad) if trim else None
        if bounds is None:
            with open(path, "rb") as f:
                audio_bytes = f.read()
        else:
            x, sr = sfile.read(path, dtype="int16")
            x = x[int(bounds[0] * sr): int(bounds[1] * sr)]
            buf = io.BytesIO()
            sfile.write(buf, x, sr, format="FLAC")
            audio_bytes = buf.getvalue()
        yield {
            "audio": {"bytes": audio_bytes, "path": r["chunk_id"] + ".flac"},
            "text": eff["text"],
            "model": eff["model"] or "",
            "provenance": json.dumps(r["provenance"], ensure_ascii=False),
            "metrics": [{"name": k, "value": float(v)} for k, v in r["metrics"]],
        }


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config_args(args)
    work = cfg["paths"]["work_dir"]
    s6 = cfg["stage6"]
    filters = s6["filters"]

    chunks = read_chunks(work)
    transcripts = read_dict_jsonl(os.path.join(work, "transcripts.jsonl"))
    aligns = read_sharded(work, "align")
    rescues = read_dict_jsonl(os.path.join(work, "rescue.jsonl"))
    print(f"chunks: {len(chunks)} | transcripts: {len(transcripts)} | "
          f"align rows: {len(aligns)} | rescue rows: {len(rescues)}")
    missing = len(set(chunks) - set(aligns))
    if missing > 0.05 * len(chunks):
        print(f"WARNING: {missing} chunks have no alignment row — did stage 4 finish?")

    drops = Counter()
    by_split = defaultdict(list)
    index_rows = []
    for cid in sorted(chunks):
        c = chunks[cid]
        tr, al, rs = transcripts.get(cid), aligns.get(cid), rescues.get(cid)
        eff = effective(c, tr, al, rs)
        if eff is not None and rs:
            eff["retry_cer"] = rs.get("retry_cer", -1.0)
        why = drop_reason(c, tr, al, eff, filters)
        if why:
            drops[why] += 1
            continue
        a = eff["align"]
        metrics = [
            ("ctc_align_score", a["ctc_align_score"]),
            ("ctc_align_min_word", a["ctc_align_min_word"]),
            ("retry_cer", eff.get("retry_cer", -1.0)),
            ("snr_db", c["snr_db"]),
            ("bleed_db", c["bleed_db"]),
            ("speech_coverage", c["speech_coverage"]),
            ("char_rate", round(len(eff["text"].replace(" ", "")) / eff["duration"], 3)),
            ("boundary_clipped", c["boundary_clipped"]),
            ("n_passes", 2 if rs else 1),
            ("duration_sec", eff["duration"]),
        ]
        assert [k for k, _ in metrics] == METRIC_ORDER
        split_key = c["extension"] if s6["split_by"] == "extension" else c["call_id"]
        split = split_of(split_key, s6["val_frac"], s6["test_frac"], s6["seed_salt"])
        provenance = {
            "primary_model": (tr or {}).get("model"),
            "retry_model": (rs or {}).get("retry_model"),
            "chosen": eff["source"],
            "n_passes": 2 if rs else 1,
            "retry_cer": eff.get("retry_cer"),
        }
        by_split[split].append({"chunk_id": cid, "eff": eff,
                                "provenance": provenance, "metrics": metrics})
        index_rows.append({
            "chunk_id": cid, "split": split, "call_id": c["call_id"],
            "channel": c["channel"], "extension": c["extension"],
            "source": eff["source"], "model": eff["model"],
            "t0": c["t0"], "t1": c["t1"],
            **{k: v for k, v in metrics},
        })

    kept = sum(len(v) for v in by_split.values())
    hours = {k: round(sum(r["eff"]["duration"] for r in v) / 3600, 2)
             for k, v in by_split.items()}
    print(f"kept {kept}/{len(chunks)} chunks | hours per split: {hours}")
    print("drop reasons:", dict(drops))
    if kept == 0:
        raise SystemExit("no rows survived filtering — check thresholds (pipeline.stats helps)")

    from datasets import Audio, Dataset, DatasetDict, Features, Value

    features = Features({
        "audio": Audio(sampling_rate=16000),
        "text": Value("string"),
        "model": Value("string"),
        "provenance": Value("string"),
        "metrics": [{"name": Value("string"), "value": Value("float64")}],
    })
    dd = {}
    for split in ("train", "validation", "test"):
        rows = by_split.get(split)
        if not rows:
            continue
        dd[split] = Dataset.from_generator(
            gen_rows,
            gen_kwargs={"rows": rows, "work_dir": work,
                        "trim": s6["trim_to_alignment"], "trim_pad": s6["trim_pad"]},
            features=features,
        )
    out_dir = os.path.join(work, "dataset")
    DatasetDict(dd).save_to_disk(out_dir)

    with open(os.path.join(work, "rows_index.jsonl"), "w", encoding="utf-8") as f:
        for r in index_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "total_chunks": len(chunks), "kept": kept,
        "drop_reasons": dict(drops), "hours_per_split": hours,
        "rows_per_split": {k: len(v) for k, v in by_split.items()},
        "filters": filters, "metric_order": METRIC_ORDER,
    }
    with open(os.path.join(work, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"dataset saved to {out_dir}")


if __name__ == "__main__":
    main()
