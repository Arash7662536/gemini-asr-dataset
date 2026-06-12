# gemini-asr-dataset

Builds a high-quality **Persian ASR fine-tuning dataset** from unlabeled
per-leg call-center recordings (`part_2/<day>/external-...-<uid>_{r,t}.wav`),
using **Gemini Flash via OpenRouter** for transcription and **CTC forced
alignment** as the quality gate. Output is an Arrow `DatasetDict`:

| column       | type                                   | content                                  |
|--------------|----------------------------------------|------------------------------------------|
| `audio`      | `Audio(sampling_rate=16000)`            | mono 16 kHz flac, bytes embedded          |
| `text`       | `string`                                | normalized Persian transcript             |
| `model`      | `string`                                | model id that produced the chosen text    |
| `provenance` | `string` (json)                         | how the label was made: primary/retry model, chosen pass, n_passes, retry CER |
| `metrics`    | `list of {name: string, value: float}`  | all quality metrics incl. CTC scores      |

## Why the pipeline looks like this

- Audio is **8 kHz mono PCM, one leg per file** (`_r` received / `_t`
  transmitted) — already single-speaker, which is ideal for ASR labels.
- There are **no transcripts**, so segmentation is **VAD-first** (Silero):
  cut clean ≤28 s single-speaker chunks, *then* transcribe each chunk.
  Asking Gemini for whole-call timestamps is the wrong way round — its audio
  timestamps are too coarse to cut training rows from.
- Telephony legs carry a faint **echo of the far party**; a per-chunk
  cross-leg energy check (`bleed_db`) drops chunks where the VAD actually
  fired on the echo.
- Gemini labels are pseudo-labels. Every transcript is **verified by CTC
  forced alignment** (MMS-300M, checkpoint
  `MahmoudAshraf/mms-300m-1130-forced-aligner`, lang `fas`): text that does
  not acoustically match the audio scores low. Flagged chunks get **one
  rescue attempt** (boundary re-cut + one retry pass at temperature 0.5,
  keep whichever aligns better); chunks that still fail are dropped.
- All metrics ride along in the dataset, so you can **re-filter later without
  re-running anything**.

## Stages

| stage | what it does | needs |
|-------|--------------|-------|
| 1 `stage1_manifest`   | pair `_r`/`_t` legs, flag silent legs, drop duplicated-mix calls | CPU |
| 2 `stage2_segments`   | Silero VAD per leg, merge to 3–28 s chunks, bleed filter, cut flacs | CPU |
| 3 `stage3_transcribe` | one Gemini pass per chunk via OpenRouter (parallel, budget-capped) | network |
| 4 `stage4_align`      | CTC forced alignment of every transcript → per-word confidences | **GPU** |
| 5 `stage5_rescue`     | re-cut + one retry pass for low-score chunks, keep the better | network + **GPU** |
| 6 `stage6_build`      | thresholds, trim to aligned speech, split by extension, save Arrow | CPU |

Plus `stats` (look before you filter) and `viewer` (HTML page with audio
players — listen to what any stage produced).

## Setup

```bash
apt-get install -y ffmpeg        # if missing
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
# edit config.yaml -> paths.audio_dir / paths.work_dir
```

## Run — step by step, inspecting as you go

Smoke-test everything first (~10 calls, separate `work_dryrun/`, a few cents):

```bash
bash run_all.sh config.yaml --dry-run
```

Then the real run, stage by stage:

```bash
python -m pipeline.stage1_manifest   --config config.yaml
python -m pipeline.stage2_segments   --config config.yaml
python -m pipeline.viewer            --config config.yaml            # LISTEN to the cuts
python -m pipeline.stage3_transcribe --config config.yaml            # spends money
python -m pipeline.viewer            --config config.yaml --status transcribed
python -m pipeline.stage4_align      --config config.yaml            # GPU
python -m pipeline.stats             --config config.yaml            # where do scores sit?
python -m pipeline.viewer            --config config.yaml --status low_align
python -m pipeline.stage5_rescue     --config config.yaml            # spends a little money
python -m pipeline.viewer            --config config.yaml --status rescued
python -m pipeline.stage6_build      --config config.yaml
```

Every stage appends jsonl and skips finished work — **just re-run after any
interruption** (budget cap, network, preemption). Stage 4 takes `--shard i/n`
to split across GPUs. Every stage takes `--limit N` to test on a few items.

## Money

- Stage 3 sends each chunk once: ~$0.60 per hour of speech at gemini-3.5-flash
  prices (audio $3/M tok @ 32 tok/s, output $9/M tok). part_2 ≈ 35–45 h of
  VAD speech ≈ **$25–30**. The stage prints real $/h as it runs — use that to
  decide the part_1 budget (or to test `google/gemini-3-flash-preview`, which
  is ~3× cheaper; change `openrouter.model` and compare on a few hundred chunks).
- `openrouter.max_cost_usd` is a hard stop, separately for stage 3 and the
  stage 5 retry pass. Costs come from the API's own `usage.cost` accounting.
- `reasoning_enabled: false` keeps Gemini's thinking off — transcription gains
  nothing from it and thinking tokens bill as output.
- OpenRouter has **no batch API**; throughput = `openrouter.concurrency`
  parallel requests. Raise it if your account's rate limit allows.

## Tuning thresholds

Run `python -m pipeline.stats` after stage 4 and look at the percentiles
before building. The main dial is `ctc_align_score_min` (default 0.40): set it
relative to the printed distribution (e.g. cut the bottom 10–20 %), and use
the viewer to listen to chunks near the cutoff. `stage5.trigger_align_score`
(default 0.45) decides who gets a rescue attempt — anything between the two
thresholds survives even if rescue fails.

Re-filter a built dataset without re-running:

```python
from datasets import load_from_disk
dd = load_from_disk("work/dataset")
m = lambda row, k: next(x["value"] for x in row["metrics"] if x["name"] == k)
strict = dd["train"].filter(lambda r: m(r, "ctc_align_score") >= 0.55)
```

## Notes

- Text normalization: Arabic→Persian letters (`ي→ی`, `ك→ک`), Arabic-Indic→
  Persian digits, tashkeel/tatweel stripped, ZWNJ tidied. Punctuation kept in
  training text; the retry CER uses a more aggressive normalizer.
- The prompt asks for numbers **in words as spoken** — digit strings romanize
  to nothing and would be excluded from alignment confidence (they would align
  as `<star>` absorbers, which still keeps their timing correct if Gemini
  outputs digits anyway).
- Splits hash on the agent **extension** from the filename, so an agent's
  voice never appears in two splits. With few extensions the split sizes are
  coarse — check `build_report.json`, and switch `split_by: call` if you only
  care about call-level leakage.
- The dataset is fully pseudo-labeled, including the test split. For honest
  WER numbers, manually correct a few hundred test rows before relying on it.
- 8 kHz telephony upsampled to 16 kHz is the standard way to fine-tune ASR
  models on call audio — apply the same resample at inference time.
- Scaling to part_1 (8×): point `paths.audio_dir` at part_1 and use a fresh
  `work_dir`. Everything else stays the same; budget caps are the only knobs
  to revisit.
