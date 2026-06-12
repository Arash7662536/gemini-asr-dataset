"""Stage 3 — transcribe every chunk once with Gemini Flash via OpenRouter.

One pass, temperature 0, thinking disabled. Each result row records the exact
model id that produced the text (provenance flows into the final dataset).
Persian normalization is applied here so all downstream stages see one
orthography; the raw model output is kept too.

Spend is metered against openrouter.max_cost_usd — the stage stops cleanly at
the cap and resumes where it left off (rows whose last attempt errored are
retried on re-run; ok rows are never re-sent).

Output: work_dir/transcripts.jsonl
Network only (no GPU). Resumable.
"""
import argparse
import os

from .common import (JsonlWriter, add_common_args, done_ok_ids, has_repetition,
                     latin_ratio, load_config_args, normalize_train, read_chunks)
from .openrouter import transcribe_items


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args()
    cfg = load_config_args(args)
    work = cfg["paths"]["work_dir"]
    orc = cfg["openrouter"]
    out_path = os.path.join(work, "transcripts.jsonl")

    chunks = read_chunks(work)
    done = done_ok_ids(out_path)
    todo = sorted((c for cid, c in chunks.items() if cid not in done),
                  key=lambda c: c["chunk_id"])
    if args.limit:
        todo = todo[:args.limit]
    hours = sum(c["duration"] for c in todo) / 3600
    print(f"chunks: {len(chunks)} | done: {len(done)} | to transcribe: {len(todo)} "
          f"({hours:.1f} h of audio)")
    if not todo:
        return

    items = [{
        "abs_path": os.path.join(work, c["path"]),
        "audio_sec": c["duration"],
        "echo": {"chunk_id": c["chunk_id"], "pass": "primary"},
    } for c in todo]

    writer = JsonlWriter(out_path)

    def on_row(row):
        if row.get("ok") and not row.get("no_speech"):
            t = normalize_train(row["text_raw"])
            row["text"] = t
            row["latin_ratio"] = round(latin_ratio(t), 3)
            row["repetition"] = int(has_repetition(t))
        elif row.get("ok"):
            row["text"] = ""
        writer.write(row)

    state = transcribe_items(items, orc, orc["model"], orc["temperature"],
                             orc["max_cost_usd"], on_row)
    writer.close()

    print(f"stage3 done: {state['n_ok']} ok, {state['n_err']} errors")
    if state["audio_sec"] > 0:
        rate = state["cost"] / (state["audio_sec"] / 3600)
        print(f"spent ${state['cost']:.2f} on {state['audio_sec'] / 3600:.1f} h "
              f"(${rate:.2f}/h of speech)")
    if state["stop_reason"]:
        print(f"STOPPED EARLY: {state['stop_reason']} — re-run to continue "
              f"(raise openrouter.max_cost_usd or top up first)")


if __name__ == "__main__":
    main()
