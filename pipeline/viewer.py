"""Sample chunks into a self-contained HTML page: audio player + text + metrics.

This is the manual-inspection tool — run it after ANY stage and open the HTML
in a browser (audio is referenced relatively, so keep the file inside work_dir).

  after stage 2:  python -m pipeline.viewer --config config.yaml
                  (listen: are chunks clean single-speaker cuts? bleed gone?)
  after stage 3:  --status transcribed   (read transcripts while listening)
  after stage 4:  --status low_align     (what does the gate want to throw away?)
  after stage 5:  --status rescued       (did rescue actually fix them?)

Output: work_dir/review_<status>.html
"""
import argparse
import html
import os
import random

from .common import load_config, read_chunks, read_dict_jsonl
from .stage5_rescue import read_sharded

CSS = """
body{font-family:sans-serif;max-width:1100px;margin:20px auto;background:#fafafa}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:10px 0}
.meta{color:#666;font-size:12px;margin-bottom:6px}
.metrics span{display:inline-block;background:#eef;border-radius:4px;padding:2px 8px;
  margin:2px;font-size:12px}
.text{direction:rtl;text-align:right;font-size:18px;margin:8px 0;line-height:1.8}
.text.retry{color:#875c00;font-size:15px}
.bad{background:#fdd}.good{background:#dfd}
audio{width:100%}
"""


def status_of(c, tr, al, rs, cfg):
    if tr is None:
        return "untranscribed"
    if not tr.get("ok"):
        return "transcribe_error"
    if tr.get("no_speech"):
        return "no_speech"
    if al is None:
        return "transcribed"
    if rs is not None:
        return "rescued"
    trig = cfg["stage5"]["trigger_align_score"]
    if not al.get("ok") or al["ctc_align_score"] < trig:
        return "low_align"
    return "aligned_ok"


def card(c, tr, al, rs, status):
    parts = [f'<div class="card">']
    parts.append(f'<div class="meta">{c["chunk_id"]} | ch={c["channel"]} '
                 f'ext={c["extension"]} | {c["t0"]}–{c["t1"]}s '
                 f'({c["duration"]}s) | <b>{status}</b></div>')
    path = (rs or {}).get("path") or c["path"]
    parts.append(f'<audio controls preload="none" src="{html.escape(path)}"></audio>')
    text = (rs or {}).get("text") or (tr or {}).get("text") or ""
    if text:
        parts.append(f'<div class="text">{html.escape(text)}</div>')
    if rs and tr and rs.get("chosen") == "retry" and tr.get("text"):
        parts.append(f'<div class="text retry">primary: {html.escape(tr["text"])}</div>')
    m = []
    m.append(("snr_db", c["snr_db"], None))
    m.append(("bleed_db", c["bleed_db"], c["bleed_db"] < 3))
    m.append(("coverage", c["speech_coverage"], None))
    if al:
        score = (rs or {}).get("ctc_align_score", al.get("ctc_align_score", 0.0))
        m.append(("align", score, score < 0.4))
        m.append(("min_word", (rs or al).get("ctc_align_min_word", 0.0), None))
    if rs:
        m.append(("retry_cer", rs.get("retry_cer"), (rs.get("retry_cer") or 0) > 0.2))
        m.append(("chosen", rs.get("chosen"), None))
    if tr:
        m.append(("model", (rs or {}).get("retry_model") or tr.get("model"), None))
    spans = "".join(
        f'<span class="{"bad" if bad else ""}">{k}={v}</span>' for k, v, bad in m)
    parts.append(f'<div class="metrics">{spans}</div></div>')
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", default="any",
                    choices=["any", "untranscribed", "transcribed", "no_speech",
                             "transcribe_error", "low_align", "rescued", "aligned_ok"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = cfg["paths"]["work_dir"]
    if args.dry_run:
        work = work.rstrip("/") + "_dryrun"

    chunks = read_chunks(work)
    transcripts = read_dict_jsonl(os.path.join(work, "transcripts.jsonl"))
    aligns = read_sharded(work, "align")
    rescues = read_dict_jsonl(os.path.join(work, "rescue.jsonl"))

    rows = []
    for cid in sorted(chunks):
        c = chunks[cid]
        tr, al, rs = transcripts.get(cid), aligns.get(cid), rescues.get(cid)
        st = status_of(c, tr, al, rs, cfg)
        if args.status in ("any", st):
            rows.append((c, tr, al, rs, st))
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    if not rows:
        print(f"no chunks with status '{args.status}'")
        return

    out = os.path.join(work, f"review_{args.status}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>")
        f.write(f"<h2>{len(rows)} chunks (status={args.status}, seed={args.seed})</h2>")
        for r in rows:
            f.write(card(*r))
    print(f"open in a browser: {out}")


if __name__ == "__main__":
    main()
