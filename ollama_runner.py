from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

OLLAMA = "http://127.0.0.1:11434"


def _json_request(url: str, method: str = "GET", payload=None, timeout: int = 3600):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def list_models():
    obj = _json_request(f"{OLLAMA}/api/tags")
    out = []
    for model in obj.get("models", []):
        size = int(model.get("size") or 0)
        out.append({
            "name": model.get("name") or model.get("model"),
            "size": size,
            "size_gb": round(size / (1024**3), 2) if size else None,
            "modified_at": model.get("modified_at"),
            "details": model.get("details", {}),
        })
    return out


def ns_to_s(value: Any):
    if not isinstance(value, (int, float)):
        return None
    return round(value / 1_000_000_000, 6)


def classify_result(response: str, done_reason: str | None, error: str | None) -> str:
    if error:
        return "error"
    if not (response or "").strip():
        return "empty_final"
    if done_reason == "length":
        return "truncated"
    return "completed"


def _options(settings: dict[str, Any]) -> dict[str, Any]:
    options = {}
    for key in ("temperature", "top_p", "top_k", "repeat_penalty", "num_ctx", "num_predict", "seed"):
        value = settings.get(key)
        if value is not None:
            options[key] = value
    return options


def generate_messages(model: str, messages: list[dict[str, str]], settings: dict[str, Any], cancel_check=None, progress_callback=None):
    """Generate one assistant turn from a complete chat history.

    Uses Ollama streaming internally so a background job can be cancelled while a
    model is generating. Closing the HTTP response disconnects the client, which
    tells Ollama to stop the in-flight generation.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": int(settings.get("keep_alive", 0)),
        "options": _options(settings),
        "think": bool(settings.get("think", False)),
    }

    started = time.monotonic()
    response_parts, thinking_parts = [], []
    final_raw = {}
    cancelled = False
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{settings.get('ollama_url') or OLLAMA}/api/chat",
            data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=int(settings.get("timeout_seconds", 7200))) as http_response:
            while True:
                if cancel_check and cancel_check():
                    cancelled = True
                    http_response.close()
                    break
                line = http_response.readline()
                if not line:
                    break
                raw = json.loads(line.decode("utf-8"))
                if raw.get("error"):
                    raise RuntimeError(str(raw["error"]))
                message = raw.get("message") or {}
                if message.get("content"):
                    response_parts.append(message["content"])
                if message.get("thinking"):
                    thinking_parts.append(message["thinking"])
                if progress_callback:
                    progress_callback({
                        "elapsed_seconds": time.monotonic() - started,
                        "output_chars": sum(len(part) for part in response_parts),
                        "thinking_chars": sum(len(part) for part in thinking_parts),
                    })
                final_raw = raw
                if raw.get("done"):
                    break

        wall = time.monotonic() - started
        response = "".join(response_parts)
        thinking = "".join(thinking_parts)
        if cancelled:
            return {
                "model": model, "response": response, "thinking": thinking,
                "thinking_exists": bool(thinking), "thinking_token_count": None,
                "thinking_duration_raw_ns": None, "thinking_duration_seconds": None,
                "wall_clock_seconds": round(wall, 4), "total_duration_raw_ns": None,
                "load_duration_raw_ns": None, "prompt_eval_duration_raw_ns": None,
                "eval_duration_raw_ns": None, "total_duration_seconds": None,
                "load_duration_seconds": None, "prompt_eval_duration_seconds": None,
                "eval_duration_seconds": None, "prompt_tokens": None, "output_tokens": None,
                "tokens_per_second": None, "done_reason": "cancelled", "error": None,
                "status": "cancelled", "raw": final_raw,
            }

        raw = final_raw
        done_reason = raw.get("done_reason")
        eval_count = raw.get("eval_count")
        eval_duration = raw.get("eval_duration")
        eval_seconds = ns_to_s(eval_duration)
        tps = round(eval_count / eval_seconds, 3) if isinstance(eval_count, (int, float)) and eval_seconds else None
        record = {
            "model": model, "response": response, "thinking": thinking,
            "thinking_exists": bool(thinking), "thinking_token_count": raw.get("thinking_count"),
            "thinking_duration_raw_ns": raw.get("thinking_duration"),
            "thinking_duration_seconds": ns_to_s(raw.get("thinking_duration")),
            "wall_clock_seconds": round(wall, 4), "total_duration_raw_ns": raw.get("total_duration"),
            "load_duration_raw_ns": raw.get("load_duration"),
            "prompt_eval_duration_raw_ns": raw.get("prompt_eval_duration"),
            "eval_duration_raw_ns": eval_duration, "total_duration_seconds": ns_to_s(raw.get("total_duration")),
            "load_duration_seconds": ns_to_s(raw.get("load_duration")),
            "prompt_eval_duration_seconds": ns_to_s(raw.get("prompt_eval_duration")),
            "eval_duration_seconds": eval_seconds, "prompt_tokens": raw.get("prompt_eval_count"),
            "output_tokens": eval_count, "tokens_per_second": tps, "done_reason": done_reason,
            "error": None, "raw": raw,
        }
        record["status"] = classify_result(response, done_reason, None)
        return record
    except Exception as exc:
        if cancel_check and cancel_check():
            return {
                "model": model, "response": "".join(response_parts), "thinking": "".join(thinking_parts),
                "thinking_exists": bool(thinking_parts), "thinking_token_count": None,
                "thinking_duration_raw_ns": None, "thinking_duration_seconds": None,
                "wall_clock_seconds": round(time.monotonic() - started, 4), "total_duration_raw_ns": None,
                "load_duration_raw_ns": None, "prompt_eval_duration_raw_ns": None,
                "eval_duration_raw_ns": None, "total_duration_seconds": None, "load_duration_seconds": None,
                "prompt_eval_duration_seconds": None, "eval_duration_seconds": None, "prompt_tokens": None,
                "output_tokens": None, "tokens_per_second": None, "done_reason": "cancelled",
                "error": None, "status": "cancelled", "raw": final_raw,
            }
        error = f"{type(exc).__name__}: {exc}"
        return {
            "model": model, "response": "", "thinking": "", "thinking_exists": False,
            "thinking_token_count": None, "thinking_duration_raw_ns": None, "thinking_duration_seconds": None,
            "wall_clock_seconds": round(time.monotonic() - started, 4), "total_duration_raw_ns": None,
            "load_duration_raw_ns": None, "prompt_eval_duration_raw_ns": None, "eval_duration_raw_ns": None,
            "total_duration_seconds": None, "load_duration_seconds": None, "prompt_eval_duration_seconds": None,
            "eval_duration_seconds": None, "prompt_tokens": None, "output_tokens": None,
            "tokens_per_second": None, "done_reason": None, "error": error, "status": "error", "raw": {},
        }


def generate(model: str, system_prompt: str, user_prompt: str, settings: dict[str, Any], cancel_check=None, progress_callback=None):
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return generate_messages(model, messages, settings, cancel_check=cancel_check, progress_callback=progress_callback)
