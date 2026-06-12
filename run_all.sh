#!/usr/bin/env bash
# Run the whole pipeline. Any extra args (e.g. --dry-run) pass to every stage.
#   bash run_all.sh config.yaml --dry-run   # smoke test first (~10 calls)
#   bash run_all.sh config.yaml
set -euo pipefail
CFG=${1:-config.yaml}
shift || true

python -m pipeline.stage1_manifest   --config "$CFG" "$@"
python -m pipeline.stage2_segments   --config "$CFG" "$@"
python -m pipeline.stage3_transcribe --config "$CFG" "$@"
python -m pipeline.stage4_align      --config "$CFG" "$@"
python -m pipeline.stage5_rescue     --config "$CFG" "$@"
python -m pipeline.stats             --config "$CFG" "$@"
python -m pipeline.stage6_build      --config "$CFG" "$@"
