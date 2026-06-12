"""Stage 4 — CTC forced alignment of every Gemini transcript against its chunk.

This is the quality gate: a transcript that does not acoustically match its
audio (hallucinated, truncated, wrong) gets a low ctc_align_score /
ctc_align_min_word and is either rescued in stage 5 or dropped in stage 6.

Output: work_dir/align.<i>of<n>.jsonl
GPU recommended. Resumable, shardable across GPUs with --shard i/n.
"""
import argparse
import os

from tqdm import tqdm

from .aligner import align_text, load_aligner
from .common import (JsonlWriter, add_common_args, done_ids, load_config_args,
                     parse_shard, read_chunks, read_dict_jsonl)


def main():
    import faulthandler
    faulthandler.enable()  # native crashes (SIGABRT/SIGSEGV) print a Python stack

    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--shard", default="0/1", help="i/n to split work across GPUs/processes")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None,
                    help="override runtime.device from config (e.g. --device cpu to test)")
    args = ap.parse_args()
    cfg = load_config_args(args)
    if args.device:
        cfg["runtime"]["device"] = args.device
    work = cfg["paths"]["work_dir"]
    si, sn = parse_shard(args.shard)

    out_path = os.path.join(work, f"align.{si}of{sn}.jsonl")
    done = done_ids(out_path, "chunk_id")

    chunks = read_chunks(work)
    transcripts = read_dict_jsonl(os.path.join(work, "transcripts.jsonl"))
    ready = sorted(
        (c for cid, c in chunks.items()
         if transcripts.get(cid, {}).get("ok") and transcripts[cid].get("text")),
        key=lambda c: c["chunk_id"])
    todo = [c for c in ready[si::sn] if c["chunk_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"shard {si}/{sn}: {len(ready[si::sn])} transcribed chunks, {len(todo)} to align")
    if not todo:
        return

    import soundfile as sfile

    handle = load_aligner(cfg)
    print(f"alignment backend: {handle[3]}")
    lang = cfg["stage4"]["language"]

    writer = JsonlWriter(out_path)
    n_ok = 0
    for c in tqdm(todo, desc=f"stage4[{si}/{sn}]"):
        wav, sr = sfile.read(os.path.join(work, c["path"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        row = align_text(handle, wav, transcripts[c["chunk_id"]]["text"], lang)
        row["chunk_id"] = c["chunk_id"]
        n_ok += int(row["ok"])
        writer.write(row)
    writer.close()
    print(f"stage4 done: {n_ok}/{len(todo)} aligned ok -> {out_path}")


if __name__ == "__main__":
    main()
