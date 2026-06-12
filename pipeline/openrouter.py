"""Async OpenRouter chat-completions client for audio transcription.

OpenRouter has no batch API; throughput comes from `concurrency` parallel
requests (each with retry/backoff). Audio goes inline as base64 WAV in an
OpenAI-style `input_audio` content part. Costs are taken from the response
usage (`usage: {include: true}`), with a token-price fallback estimate, and
accumulate against a hard budget cap — when the cap is hit no new requests
start and the stage exits cleanly (it is resumable).

Gemini "thinking" is disabled via OpenRouter's unified `reasoning` parameter
when `reasoning_enabled: false` (saves output tokens and latency; transcription
does not benefit from it).
"""
import asyncio
import base64
import io
import os
import random
import time

import httpx

NO_SPEECH = "[NO_SPEECH]"
AUDIO_TOKENS_PER_SEC = 32.0  # Gemini audio tokenization rate

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 524, 529}


def require_api_key(orc):
    key = os.environ.get(orc["api_key_env"], "")
    if not key:
        raise SystemExit(f"set the OpenRouter key first:  export {orc['api_key_env']}=sk-or-...")
    return key


def wav_b64(path):
    """flac chunk -> base64 WAV bytes (OpenRouter input_audio accepts wav/mp3)."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="int16")
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def clean_text(content):
    t = (content or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        first, _, rest = t.partition("\n")
        if first.strip().lower() in ("text", "plaintext", "txt"):
            t = rest
    return t.strip().strip('"').strip()


def is_no_speech(t):
    u = t.upper().replace(" ", "").replace("[", "").replace("]", "")
    return u in ("", "NOSPEECH", "NO_SPEECH")


def estimate_cost(prices, audio_sec, prompt_tokens, completion_tokens):
    audio_tok = audio_sec * AUDIO_TOKENS_PER_SEC
    text_tok = max(0.0, float(prompt_tokens or 0) - audio_tok)
    return (audio_tok * prices["audio"] + text_tok * prices["prompt"]
            + float(completion_tokens or 0) * prices["completion"]) / 1e6


def build_payload(orc, model, temperature, b64):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": orc["system_prompt"]},
            {"role": "user", "content": [
                {"type": "text", "text": orc["user_prompt"]},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            ]},
        ],
        "temperature": temperature,
        "max_tokens": orc["max_tokens"],
        "usage": {"include": True},
    }
    if orc.get("reasoning_enabled") is False:
        payload["reasoning"] = {"enabled": False}
    if orc.get("provider"):
        payload["provider"] = orc["provider"]
    if orc.get("extra_body"):
        payload.update(orc["extra_body"])
    return payload


async def _transcribe_one(client, sem, orc, model, temperature, item, state):
    async with sem:
        if state["stop"]:
            return None
        t_start = time.time()
        try:
            b64 = await asyncio.to_thread(wav_b64, item["abs_path"])
        except Exception as e:
            return {**item["echo"], "ok": False, "error": f"audio: {type(e).__name__}: {e}"}
        payload = build_payload(orc, model, temperature, b64)

        err = None
        for attempt in range(orc["max_retries"] + 1):
            if state["stop"]:
                return None
            try:
                r = await client.post("/chat/completions", json=payload)
            except httpx.HTTPError as e:
                err = f"net: {type(e).__name__}"
                await asyncio.sleep(min(60.0, 2.0 ** attempt + random.random()))
                continue
            if r.status_code in RETRY_STATUS:
                err = f"http {r.status_code}"
                wait = float(r.headers.get("retry-after") or 0) or min(60.0, 2.0 ** attempt)
                await asyncio.sleep(wait + random.random())
                continue
            if r.status_code != 200:
                err = f"http {r.status_code}: {r.text[:200]}"
                if r.status_code == 402:  # out of credits — pointless to continue
                    state["stop"] = True
                    state["stop_reason"] = "OpenRouter returned 402 (out of credits)"
                break
            try:
                data = r.json()
            except ValueError:
                err = "unparseable response body"
                continue
            if data.get("error"):
                code = data["error"].get("code")
                err = f"api {code}: {str(data['error'].get('message'))[:200]}"
                if code in RETRY_STATUS:
                    await asyncio.sleep(min(60.0, 2.0 ** attempt + random.random()))
                    continue
                break

            msg = (data.get("choices") or [{}])[0].get("message") or {}
            text = clean_text(msg.get("content"))
            usage = data.get("usage") or {}
            cost = usage.get("cost")
            if cost is None:
                cost = estimate_cost(orc["prices_per_mtok"], item["audio_sec"],
                                     usage.get("prompt_tokens"), usage.get("completion_tokens"))
            state["cost"] += float(cost)
            state["audio_sec"] += item["audio_sec"]
            if state["cost"] >= state["max_cost"]:
                state["stop"] = True
                state["stop_reason"] = f"budget cap ${state['max_cost']:.2f} reached"
            if not text and attempt < orc["max_retries"]:
                err = "empty completion"
                continue
            no_speech = is_no_speech(text)
            return {
                **item["echo"],
                "ok": True,
                "no_speech": no_speech,
                "text_raw": "" if no_speech else text,
                "model": data.get("model") or model,
                "temperature": temperature,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": round(float(cost), 6),
                "latency_s": round(time.time() - t_start, 2),
            }
        return {**item["echo"], "ok": False, "error": err or "exhausted retries"}


async def _run(items, orc, model, temperature, max_cost_usd, on_row):
    from tqdm import tqdm
    state = {"cost": 0.0, "audio_sec": 0.0, "stop": False, "stop_reason": None,
             "max_cost": max_cost_usd, "n_ok": 0, "n_err": 0}
    headers = {"Authorization": f"Bearer {require_api_key(orc)}"}
    sem = asyncio.Semaphore(orc["concurrency"])
    async with httpx.AsyncClient(base_url=orc["base_url"], headers=headers,
                                 timeout=httpx.Timeout(orc["timeout_s"])) as client:
        tasks = [asyncio.create_task(
            _transcribe_one(client, sem, orc, model, temperature, it, state))
            for it in items]
        pbar = tqdm(total=len(tasks), desc=f"transcribe[{model}@{temperature}]")
        for fut in asyncio.as_completed(tasks):
            row = await fut
            pbar.update(1)
            if row is None:
                continue
            state["n_ok" if row["ok"] else "n_err"] += 1
            on_row(row)
            pbar.set_postfix(cost=f"${state['cost']:.2f}", err=state["n_err"])
        pbar.close()
    return state


def transcribe_items(items, orc, model, temperature, max_cost_usd, on_row):
    """items: [{abs_path, audio_sec, echo: {...}}] — echo fields are copied into
    every result row. on_row(row) is called as results arrive (event-loop thread,
    no locking needed). Returns the final state dict (cost, counts, stop_reason)."""
    if not items:
        return {"cost": 0.0, "audio_sec": 0.0, "stop": False, "stop_reason": None,
                "n_ok": 0, "n_err": 0}
    return asyncio.run(_run(items, orc, model, temperature, max_cost_usd, on_row))
