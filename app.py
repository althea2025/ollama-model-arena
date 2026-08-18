from __future__ import annotations

import copy
import os
import csv
import json
import random
import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr

from ollama_runner import generate, generate_messages, list_models

ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
EXPORTS.mkdir(exist_ok=True)
CONFIG_PATH = ROOT / "config.json"

PRESETS = {
    "安全長輸出 / 推理": {
        "temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1,
        "num_ctx": 32768, "num_predict": 8192, "seed": 42, "think": False,
    },
    "創作 / RP": {
        "temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1,
        "num_ctx": 32768, "num_predict": 4096, "seed": 42, "think": False,
    },
    "Thinking 長輸出": {
        "temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1,
        "num_ctx": 32768, "num_predict": 8192, "seed": 42, "think": True,
    },
}


def load_config():
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_config(config):
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(CONFIG_PATH)


CONFIG = load_config()
DEFAULTS = CONFIG.get("defaults", {})


def dflt(key, fallback):
    return DEFAULTS.get(key, fallback)


def save_preferences(temperature, top_p, top_k, repeat_penalty, num_ctx, num_predict, seed, think):
    cfg = load_config()
    defaults = cfg.setdefault("defaults", {})
    defaults.update({
        "temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k),
        "repeat_penalty": float(repeat_penalty), "num_ctx": int(num_ctx),
        "num_predict": int(num_predict), "seed": int(seed), "think": bool(think),
    })
    write_config(cfg)
    return "✅ 已儲存為下次啟動的預設值。"


def _friendly_model_label(name, size_gb):
    """Compact label for narrow/mobile selectors while keeping the exact tag in mapping."""
    gb = f"{size_gb} GB" if size_gb is not None else "size ?"
    display = name
    owner = ""
    if name.startswith("hf.co/"):
        parts = name.split("/", 2)
        if len(parts) == 3:
            owner = parts[1]
            display = parts[2]
    if len(display) > 48:
        display = display[:45] + "…"
    owner_part = f" · {owner}" if owner else ""
    return f"{display}{owner_part} · {gb}"


def model_choices():
    models = list_models()
    labels, mapping = [], {}
    used = set()
    for model in models:
        label = _friendly_model_label(model["name"], model.get("size_gb"))
        # Keep labels unique if two local entries collapse to the same friendly label.
        if label in used:
            label = f"{label} · {model['name']}"
        used.add(label)
        labels.append(label)
        mapping[label] = model["name"]
    return labels, mapping


def selected_models_preview(selected):
    if not selected:
        return "尚未選擇模型。"
    _, mapping = model_choices()
    lines = [f"**已選 {len(selected)} 個模型**"]
    for item in selected:
        actual = mapping.get(item, item)
        lines.append(f"- `{actual}`")
    return "\n".join(lines)


def refresh_both():
    try:
        labels, _ = model_choices()
        msg = f"找到 **{len(labels)}** 個本機 Ollama 模型。"
        return gr.update(choices=labels, value=[]), gr.update(choices=labels, value=[]), msg, msg
    except Exception as exc:
        msg = f"❌ 無法連線 Ollama：{exc}"
        return gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), msg, msg


def unique_run_dir(run_id):
    path = EXPORTS / run_id
    counter = 1
    while path.exists():
        path = EXPORTS / f"{run_id}_{counter:02d}"
        counter += 1
    path.mkdir(parents=True)
    return path


def status_label(result):
    return {
        "completed": "✅ COMPLETED",
        "truncated": "⚠️ TRUNCATED — OUTPUT TOKEN LIMIT",
        "empty_final": "⚠️ EMPTY FINAL RESPONSE",
        "error": "❌ ERROR",
    }.get(result.get("status"), result.get("status") or "UNKNOWN")


def settings_warning(think, num_predict):
    value = int(num_predict or 0)
    if think and value < 8192:
        return "⚠️ **Thinking 已開啟，但 Max output tokens < 8192。** 長推理可能吃掉輸出預算，建議改成 8192 以上。"
    if value < 4096:
        return "⚠️ Max output tokens 偏低，長文很容易被截斷。"
    return "✅ 輸出上限設定正常。"


def prompt_stats(system_prompt, user_prompt):
    return f"System：{len(system_prompt or ''):,} 字元　|　User：{len(user_prompt or ''):,} 字元"


def apply_preset(name):
    p = PRESETS[name]
    return (p["temperature"], p["top_p"], p["top_k"], p["repeat_penalty"], p["num_ctx"],
            p["num_predict"], p["seed"], p["think"], settings_warning(p["think"], p["num_predict"]))


def build_settings(temperature, top_p, top_k, repeat_penalty, num_ctx, num_predict, seed, think):
    cfg = load_config()
    return {
        "temperature": float(temperature), "top_p": float(top_p), "top_k": int(top_k),
        "repeat_penalty": float(repeat_penalty), "num_ctx": int(num_ctx),
        "num_predict": int(num_predict), "seed": int(seed), "think": bool(think),
        "ollama_url": cfg.get("ollama_url", "http://127.0.0.1:11434"),
        "timeout_seconds": int(cfg.get("defaults", {}).get("timeout_seconds", 7200)),
        "keep_alive": int(cfg.get("defaults", {}).get("keep_alive", cfg.get("keep_alive", 0))),
    }


# ---------- Background job manager ----------
# Jobs are executed by a server-side worker thread. Once submitted, generation no longer
# depends on the browser connection staying open. All progress is persisted under exports/.
JOB_QUEUE = queue.Queue()
JOB_META_NAME = "job.json"
JOB_REQUEST_NAME = "request.json"


def _now_iso():
    return datetime.now().astimezone().isoformat()


def _atomic_write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _job_meta_path(run_dir):
    return Path(run_dir) / JOB_META_NAME


def _read_job_meta(run_dir):
    path = _job_meta_path(run_dir)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _update_job_meta(run_dir, **updates):
    meta = _read_job_meta(run_dir)
    meta.update(updates)
    meta["updated_at"] = _now_iso()
    _atomic_write_json(_job_meta_path(run_dir), meta)
    return meta


def _new_job(run_dir, mode, request, total_steps):
    settings = request.get("settings") or {}
    meta = {
        "job_id": Path(run_dir).name,
        "mode": mode,
        "state": "queued",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "completed_steps": 0,
        "total_steps": int(total_steps),
        "current": "等待背景 worker",
        "cancel_requested": False,
        "error": None,
        "anonymous": bool(request.get("anonymous")),
        "think": bool(settings.get("think")),
        "model_count": len(request.get("models") or []),
    }
    _atomic_write_json(Path(run_dir) / JOB_REQUEST_NAME, request)
    _atomic_write_json(_job_meta_path(run_dir), meta)
    return meta


def _job_cancel_requested(run_dir):
    return bool(_read_job_meta(run_dir).get("cancel_requested"))


def _execute_single_background(run_dir, request):
    models = request["models"]
    settings = request["settings"]
    results = []
    mapping_out = request["mapping"]
    run_id = Path(run_dir).name
    for index, item in enumerate(models, 1):
        model, alias = item["model"], item["display_name"]
        _update_job_meta(run_dir, current=f"{alias} ({index}/{len(models)})")
        result = generate(model, request["system_prompt"], request["user_prompt"], settings)
        result["think"] = bool(settings.get("think"))
        result["display_name"] = alias
        results.append(result)
        partial = {
            "run_id": run_id, "mode": "single_turn", "system_prompt": request["system_prompt"],
            "user_prompt": request["user_prompt"], "settings": settings,
            "anonymous": request["anonymous"], "mapping": mapping_out, "results": results,
        }
        export_run(partial, Path(run_dir))
        _update_job_meta(run_dir, completed_steps=index, current=f"完成 {alias}: {status_label(result)}")
        if _job_cancel_requested(run_dir):
            return "cancelled"
    return "completed"


def _execute_multi_background(run_dir, request):
    settings = request["settings"]
    user_rounds = request["user_rounds"]
    models = request["models"]
    mapping_out = request["mapping"]
    sessions = []
    completed_steps = 0
    run_id = Path(run_dir).name

    for model_index, item in enumerate(models, 1):
        model, alias = item["model"], item["display_name"]
        session = {"model": model, "display_name": alias, "rounds": [], "stats": {}}
        sessions.append(session)
        messages = []
        if request["system_prompt"].strip():
            messages.append({"role": "system", "content": request["system_prompt"]})

        for round_idx, user_text in enumerate(user_rounds, 1):
            _update_job_meta(
                run_dir,
                current=f"{alias} — Round {round_idx}/{len(user_rounds)} (Model {model_index}/{len(models)})",
                completed_steps=completed_steps,
            )
            messages.append({"role": "user", "content": user_text})
            result = generate_messages(model, messages, settings)
            result["think"] = bool(settings.get("think"))
            result["display_name"] = alias
            session["rounds"].append({"round": round_idx, "user": user_text, "result": result})
            if result.get("response", "").strip():
                messages.append({"role": "assistant", "content": result["response"]})
            recompute_session_stats(session)
            completed_steps += 1

            partial = {
                "run_id": run_id, "mode": "multi_turn", "system_prompt": request["system_prompt"],
                "user_rounds": user_rounds, "settings": settings, "anonymous": request["anonymous"],
                "mapping": mapping_out, "sessions": sessions,
            }
            export_multiturn(partial, Path(run_dir))
            _update_job_meta(
                run_dir,
                completed_steps=completed_steps,
                current=f"完成 {alias} — Round {round_idx}: {status_label(result)}",
            )

            if _job_cancel_requested(run_dir):
                return "cancelled"
            if result.get("status") in {"error", "empty_final"}:
                session["halted_after_round"] = round_idx
                session["halt_reason"] = result.get("status")
                break
    return "completed"


def _job_worker():
    while True:
        item = JOB_QUEUE.get()
        run_dir = Path(item["run_dir"])
        try:
            if _job_cancel_requested(run_dir):
                _update_job_meta(run_dir, state="cancelled", finished_at=_now_iso(), current="已取消")
                continue
            _update_job_meta(run_dir, state="running", started_at=_now_iso(), current="背景任務已開始")
            if item["mode"] == "single_turn":
                final_state = _execute_single_background(run_dir, item["request"])
            else:
                final_state = _execute_multi_background(run_dir, item["request"])
            _update_job_meta(
                run_dir,
                state=final_state,
                finished_at=_now_iso(),
                current="已完成" if final_state == "completed" else "已取消（目前生成完成後停止）",
            )
        except Exception:
            _update_job_meta(
                run_dir,
                state="failed",
                finished_at=_now_iso(),
                current="背景 worker 發生錯誤",
                error=traceback.format_exc(),
            )
        finally:
            JOB_QUEUE.task_done()


def _recover_stale_jobs():
    # A server restart cannot resume an in-flight HTTP generation safely. Mark stale jobs
    # explicitly instead of pretending they are still running.
    for path in EXPORTS.iterdir():
        if not path.is_dir() or not (path / JOB_META_NAME).exists():
            continue
        meta = _read_job_meta(path)
        if meta.get("state") in {"queued", "running"}:
            _update_job_meta(
                path,
                state="interrupted",
                finished_at=_now_iso(),
                current="Arena 曾重新啟動；此任務已中斷",
            )


_recover_stale_jobs()
_JOB_THREAD = threading.Thread(target=_job_worker, name="ollama-arena-worker", daemon=True)
_JOB_THREAD.start()


def _resolve_models(selected, anonymous):
    _, mapping = model_choices()
    models = [mapping.get(item, item.split("  —  ")[0]) for item in selected]
    if anonymous:
        random.shuffle(models)
    mapping_out = {}
    resolved = []
    for index, model in enumerate(models, 1):
        alias = f"Model {chr(64 + index)}" if anonymous else model
        mapping_out[alias] = model
        resolved.append({"model": model, "display_name": alias})
    return resolved, mapping_out


def submit_single_job(selected, system_prompt, user_prompt, temperature, top_p, top_k, repeat_penalty,
                      num_ctx, num_predict, seed, think, anonymous, show_thinking):
    if not selected:
        return "❌ 請至少選一個模型。"
    if not (user_prompt or "").strip():
        return "❌ User Prompt 不可空白。"
    models, mapping_out = _resolve_models(selected, anonymous)
    settings = build_settings(temperature, top_p, top_k, repeat_penalty, num_ctx, num_predict, seed, think)
    run_dir = unique_run_dir(datetime.now().strftime("%Y%m%d_%H%M%S") + "_single")
    request = {
        "mode": "single_turn", "system_prompt": system_prompt or "", "user_prompt": user_prompt,
        "settings": settings, "anonymous": bool(anonymous), "show_thinking": bool(show_thinking),
        "models": models, "mapping": mapping_out,
    }
    _new_job(run_dir, "single_turn", request, len(models))
    JOB_QUEUE.put({"run_dir": str(run_dir), "mode": "single_turn", "request": request})
    return (f"✅ 已建立背景任務 `{run_dir.name}`。你現在可以切到別的 App、關閉 Safari 分頁，"
            f"Mac 仍會繼續執行。到「📋 背景任務」分頁查看進度與結果。")


def submit_multi_job(selected, system_prompt, temperature, top_p, top_k, repeat_penalty, num_ctx,
                     num_predict, seed, think, anonymous, show_thinking, round_count, *round_texts):
    if not selected:
        return "❌ 請至少選一個模型。"
    user_rounds = normalize_round_texts(round_count, *round_texts)
    if not user_rounds:
        return "❌ 請至少填寫一輪 User Message。"
    models, mapping_out = _resolve_models(selected, anonymous)
    settings = build_settings(temperature, top_p, top_k, repeat_penalty, num_ctx, num_predict, seed, think)
    run_dir = unique_run_dir(datetime.now().strftime("%Y%m%d_%H%M%S") + "_multiturn")
    request = {
        "mode": "multi_turn", "system_prompt": system_prompt or "", "user_rounds": user_rounds,
        "settings": settings, "anonymous": bool(anonymous), "show_thinking": bool(show_thinking),
        "models": models, "mapping": mapping_out,
    }
    _new_job(run_dir, "multi_turn", request, len(models) * len(user_rounds))
    JOB_QUEUE.put({"run_dir": str(run_dir), "mode": "multi_turn", "request": request})
    return (f"✅ 已建立背景任務 `{run_dir.name}`（{len(models)} models × {len(user_rounds)} rounds）。"
            f"你現在可以離開頁面，Mac 會繼續跑。到「📋 背景任務」分頁查看。")


def _job_icon(state):
    return {
        "queued": "🕓", "running": "🟡", "completed": "✅", "cancelled": "🛑",
        "failed": "❌", "interrupted": "⚠️",
    }.get(state, "•")


def _job_progress(meta):
    done = int(meta.get("completed_steps") or 0)
    total = int(meta.get("total_steps") or 0)
    pct = round(done * 100 / total) if total else 0
    return done, total, pct


def list_jobs(limit=50):
    items = []
    for path in sorted((p for p in EXPORTS.iterdir() if p.is_dir()), reverse=True):
        meta = _read_job_meta(path)
        if not meta:
            continue
        done, total, pct = _job_progress(meta)
        label = f"{_job_icon(meta.get('state'))} {path.name} · {meta.get('mode')} · {done}/{total} ({pct}%)"
        items.append((label, path.name))
        if len(items) >= limit:
            break
    return items


def refresh_job_list(current=None):
    choices = list_jobs()
    values = [value for _, value in choices]
    value = current if current in values else (values[0] if values else None)
    summary = f"共找到 **{len(choices)}** 個背景任務。" if choices else "尚無背景任務。"
    return gr.update(choices=choices, value=value), summary


def _job_status_markdown(meta):
    if not meta:
        return "找不到任務。"
    done, total, pct = _job_progress(meta)
    think_label = "ON — 想清楚再答" if bool(meta.get("think")) else "OFF — 自然生成"
    anonymous_label = "ON" if bool(meta.get("anonymous")) else "OFF"
    model_count = meta.get("model_count")
    model_line = f"**模型數：** {model_count}  \n" if model_count is not None else ""
    return (
        f"## {_job_icon(meta.get('state'))} {meta.get('job_id')}\n\n"
        f"**狀態：** `{meta.get('state')}`  \n"
        f"**模式：** `{meta.get('mode')}`  \n"
        f"{model_line}"
        f"**Thinking：** `{think_label}`  \n"
        f"**匿名模式：** `{anonymous_label}`  \n"
        f"**進度：** {done}/{total}（{pct}%）  \n"
        f"**目前：** {meta.get('current') or '-'}  \n"
        f"**建立：** {meta.get('created_at') or '-'}  \n"
        f"**開始：** {meta.get('started_at') or '-'}  \n"
        f"**完成：** {meta.get('finished_at') or '-'}"
        + (f"\n\n**錯誤：**\n```text\n{meta.get('error')}\n```" if meta.get("error") else "")
    )


def load_job(job_id, reveal_models=False, show_thinking=False):
    if not job_id:
        return "請選擇任務。", "", None, None, None
    run_dir = EXPORTS / job_id
    meta = _read_job_meta(run_dir)
    status = _job_status_markdown(meta)
    mode = meta.get("mode")
    if mode == "multi_turn":
        result_path = run_dir / "multiturn_result.json"
        md_path, csv_path = run_dir / "multiturn_result.md", run_dir / "multiturn_result.csv"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            body = render_multiturn(data.get("sessions", []), bool(reveal_models or not data.get("anonymous")), show_thinking)
        else:
            body = "尚未產生第一輪結果。"
    else:
        result_path = run_dir / "result.json"
        md_path, csv_path = run_dir / "result.md", run_dir / "result.csv"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            body = render_results(data.get("results", []), bool(reveal_models or not data.get("anonymous")), show_thinking)
        else:
            body = "尚未產生第一個模型結果。"
    return status, body, str(md_path) if md_path.exists() else None, str(result_path) if result_path.exists() else None, str(csv_path) if csv_path.exists() else None



def _masked_variant_name(show_models, show_thinking):
    if show_models and show_thinking:
        return "full_copy"
    if not show_models and not show_thinking:
        return "blind"
    if not show_models:
        return "models_masked"
    return "thinking_hidden"


def _masked_alias(index):
    return f"Model {index + 1}"


def _build_masked_run(data, show_models=True, show_thinking=True):
    """Create a derived copy for export. Source data is never mutated."""
    masked = copy.deepcopy(data)
    masked["derived_export"] = {
        "source_run_id": data.get("run_id"),
        "show_real_model_names": bool(show_models),
        "show_thinking_content": bool(show_thinking),
    }

    if masked.get("mode") == "multi_turn":
        sessions = masked.get("sessions", [])
        for index, session in enumerate(sessions):
            alias = session.get("model") if show_models else _masked_alias(index)
            if show_models:
                session["display_name"] = session.get("model") or session.get("display_name")
            else:
                session["model"] = alias
                session["display_name"] = alias
            for rr in session.get("rounds", []):
                result = rr.get("result") or {}
                if not show_models:
                    result["model"] = alias
                    result["display_name"] = alias
                if not show_thinking:
                    result["thinking"] = ""
        if not show_models:
            masked["mapping"] = {}
            masked["anonymous"] = True
    else:
        results = masked.get("results", [])
        for index, result in enumerate(results):
            alias = result.get("model") if show_models else _masked_alias(index)
            if show_models:
                result["display_name"] = result.get("model") or result.get("display_name")
            else:
                result["model"] = alias
                result["display_name"] = alias
            if not show_thinking:
                result["thinking"] = ""
        if not show_models:
            masked["mapping"] = {}
            masked["anonymous"] = True

    return masked


def _write_masked_single(run, masked_dir, suffix, show_models, show_thinking):
    masked_dir.mkdir(parents=True, exist_ok=True)
    json_path = masked_dir / f"result_{suffix}.json"
    md_path = masked_dir / f"result_{suffix}.md"
    csv_path = masked_dir / f"result_{suffix}.csv"

    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Ollama Model Arena — Derived Export", "",
        f"- Source run: `{run.get('run_id')}`",
        f"- Real model names: `{'shown' if show_models else 'masked'}`",
        f"- Thinking content: `{'shown' if show_thinking else 'hidden'}`",
        "", "## System Prompt", run.get("system_prompt") or "(空)",
        "", "## User Prompt", run.get("user_prompt") or "",
        "", "## Settings", "```json",
        json.dumps(run.get("settings", {}), ensure_ascii=False, indent=2), "```",
    ]
    for result in run.get("results", []):
        md += [
            "", f"## {result.get('display_name')}",
            f"- Status: **{result.get('status')}**",
            f"- Done reason: `{result.get('done_reason')}`",
            f"- Thinking: **{'ON' if result.get('think') else 'OFF'}**",
            f"- Wall: {result.get('wall_clock_seconds')} s",
            f"- Prompt tokens: {result.get('prompt_tokens')}",
            f"- Output tokens: {result.get('output_tokens')}",
            f"- Speed: {result.get('tokens_per_second')} tok/s",
            f"- Error: {result.get('error') or 'None'}",
        ]
        if show_thinking and result.get("thinking"):
            md += ["", "### Thinking", "", result.get("thinking", "")]
        md += ["", "### Response", "", result.get("response", "")]
    md_path.write_text("\n".join(md), encoding="utf-8")

    fields = [
        "display_name", "model", "status", "done_reason", "think", "thinking",
        "thinking_token_count", "wall_clock_seconds", "total_duration_seconds",
        "load_duration_seconds", "prompt_eval_duration_seconds", "eval_duration_seconds",
        "prompt_tokens", "output_tokens", "tokens_per_second", "error", "response",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in run.get("results", []):
            writer.writerow({key: result.get(key) for key in fields})

    return str(md_path), str(json_path), str(csv_path)


def _write_masked_multiturn(run, masked_dir, suffix, show_models, show_thinking):
    masked_dir.mkdir(parents=True, exist_ok=True)
    json_path = masked_dir / f"multiturn_result_{suffix}.json"
    md_path = masked_dir / f"multiturn_result_{suffix}.md"
    csv_path = masked_dir / f"multiturn_result_{suffix}.csv"

    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Ollama Model Arena — Multi-turn Derived Export", "",
        f"- Source run: `{run.get('run_id')}`",
        f"- Real model names: `{'shown' if show_models else 'masked'}`",
        f"- Thinking content: `{'shown' if show_thinking else 'hidden'}`",
        f"- Rounds: `{len(run.get('user_rounds', []))}`",
        "", "## System Prompt", run.get("system_prompt") or "(空)",
        "", "## Settings", "```json",
        json.dumps(run.get("settings", {}), ensure_ascii=False, indent=2), "```",
    ]
    for session in run.get("sessions", []):
        md += ["", f"# {session.get('display_name')}"]
        for rr in session.get("rounds", []):
            result = rr.get("result") or {}
            md += [
                "", f"## Round {rr.get('round')}",
                "", "### User", "", rr.get("user", ""),
                "", "### Assistant",
                f"- Status: **{result.get('status')}**",
                f"- Done reason: `{result.get('done_reason')}`",
                f"- Thinking: **{'ON' if result.get('think') else 'OFF'}**",
                f"- Wall: {result.get('wall_clock_seconds')} s",
                f"- Prompt tokens: {result.get('prompt_tokens')}",
                f"- Output tokens: {result.get('output_tokens')}",
                f"- Speed: {result.get('tokens_per_second')} tok/s",
                f"- Error: {result.get('error') or 'None'}",
            ]
            if show_thinking and result.get("thinking"):
                md += ["", "### Thinking", "", result.get("thinking", "")]
            md += ["", result.get("response", "")]
        md += ["", "### Session totals", "```json",
               json.dumps(session.get("stats", {}), ensure_ascii=False, indent=2), "```"]
    md_path.write_text("\n".join(md), encoding="utf-8")

    fields = [
        "display_name", "model", "round", "user", "status", "done_reason", "think",
        "thinking", "thinking_token_count", "wall_clock_seconds",
        "total_duration_seconds", "load_duration_seconds", "prompt_eval_duration_seconds",
        "eval_duration_seconds", "prompt_tokens", "output_tokens",
        "tokens_per_second", "error", "response",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for session in run.get("sessions", []):
            for rr in session.get("rounds", []):
                result = rr.get("result") or {}
                row = {
                    "display_name": session.get("display_name"),
                    "model": session.get("model"),
                    "round": rr.get("round"),
                    "user": rr.get("user"),
                }
                row.update({k: result.get(k) for k in fields if k not in row})
                writer.writerow(row)

    return str(md_path), str(json_path), str(csv_path)


def create_masked_export(job_id, show_models=True, show_thinking=True):
    """Create a derived export under masked/. Never overwrites original result files."""
    if not job_id:
        return "請先選擇任務。", None, None, None

    run_dir = EXPORTS / job_id
    meta = _read_job_meta(run_dir)
    if not meta:
        return "找不到任務。", None, None, None

    source_json = run_dir / ("multiturn_result.json" if meta.get("mode") == "multi_turn" else "result.json")
    if not source_json.exists():
        return "目前還沒有可匯出的結果。", None, None, None

    source_data = json.loads(source_json.read_text(encoding="utf-8"))
    derived = _build_masked_run(source_data, bool(show_models), bool(show_thinking))
    suffix = _masked_variant_name(bool(show_models), bool(show_thinking))
    masked_dir = run_dir / "masked"

    if meta.get("mode") == "multi_turn":
        md_path, json_path, csv_path = _write_masked_multiturn(
            derived, masked_dir, suffix, bool(show_models), bool(show_thinking)
        )
    else:
        md_path, json_path, csv_path = _write_masked_single(
            derived, masked_dir, suffix, bool(show_models), bool(show_thinking)
        )

    return (
        f"✅ 已產生衍生輸出：`masked/{Path(md_path).name}`。原始結果完全未修改。",
        md_path, json_path, csv_path,
    )


def request_cancel(job_id):
    if not job_id:
        return "請先選擇任務。"
    run_dir = EXPORTS / job_id
    meta = _read_job_meta(run_dir)
    if not meta:
        return "找不到任務。"
    if meta.get("state") in {"completed", "cancelled", "failed", "interrupted"}:
        return f"任務已是 `{meta.get('state')}`，不需要取消。"
    _update_job_meta(run_dir, cancel_requested=True, current="已要求停止；會在目前這次模型生成完成後停止")
    return "🛑 已要求停止。正在生成中的 Ollama request 不會硬切斷；完成目前模型/輪次後會停止後續工作。"


# ---------- Single-turn ----------
def export_run(run, run_dir):
    (run_dir / "result.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Ollama Model Arena", "", f"- Run: `{run['run_id']}`", f"- Anonymous: `{run['anonymous']}`",
          "", "## System Prompt", run["system_prompt"] or "(空)", "", "## User Prompt", run["user_prompt"],
          "", "## Settings", "```json", json.dumps(run["settings"], ensure_ascii=False, indent=2), "```"]
    for result in run["results"]:
        md += ["", f"## {result['display_name']}", f"- Actual model: `{result['model']}`",
               f"- Status: **{result.get('status')}**", f"- Done reason: `{result.get('done_reason')}`",
               f"- Thinking: **{'ON' if result.get('think') else 'OFF'}**",
               f"- Wall: {result.get('wall_clock_seconds')} s", f"- Total: {result.get('total_duration_seconds')} s",
               f"- Load: {result.get('load_duration_seconds')} s", f"- Prompt eval: {result.get('prompt_eval_duration_seconds')} s",
               f"- Generation: {result.get('eval_duration_seconds')} s", f"- Prompt tokens: {result.get('prompt_tokens')}",
               f"- Output tokens: {result.get('output_tokens')}", f"- Thinking tokens: {result.get('thinking_token_count')}",
               f"- Speed: {result.get('tokens_per_second')} tok/s", f"- Error: {result.get('error') or 'None'}"]
        if result.get("status") == "truncated": md += ["", "⚠️ **TRUNCATED — OUTPUT TOKEN LIMIT**"]
        elif result.get("status") == "empty_final": md += ["", "⚠️ **EMPTY FINAL RESPONSE**"]
        if result.get("thinking"): md += ["", "### Thinking", "", result["thinking"]]
        md += ["", "### Response", "", result.get("response", "")]
    (run_dir / "result.md").write_text("\n".join(md), encoding="utf-8")

    fields = ["display_name", "model", "status", "done_reason", "think", "thinking", "thinking_token_count",
              "wall_clock_seconds", "total_duration_seconds", "load_duration_seconds", "prompt_eval_duration_seconds",
              "eval_duration_seconds", "prompt_tokens", "output_tokens", "tokens_per_second", "error", "response"]
    with (run_dir / "result.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for result in run["results"]: writer.writerow({key: result.get(key) for key in fields})
    return str(run_dir / "result.md"), str(run_dir / "result.json"), str(run_dir / "result.csv")


def render_results(results, revealed=True, show_thinking=False):
    parts = []
    for result in results:
        title = result["model"] if revealed else result["display_name"]
        parts.append(f"## {title}\n\n**{status_label(result)}**  \n**Done reason:** `{result.get('done_reason')}`　"
                     f"**Thinking:** {'ON' if result.get('think') else 'OFF'}  \n"
                     f"**Wall:** {result.get('wall_clock_seconds')}s　**Generation:** {result.get('eval_duration_seconds')}s　"
                     f"**Output tokens:** {result.get('output_tokens')}　**Speed:** {result.get('tokens_per_second')} tok/s  \n"
                     f"**Error:** {result.get('error') or 'None'}\n")
        if show_thinking and result.get("thinking"):
            parts.append(f"\n<details><summary>Thinking</summary>\n\n{result['thinking']}\n\n</details>\n")
        response = result.get("response", "")
        if result.get("status") == "empty_final": response = "⚠️ *模型 API 成功回傳，但 final response 為空。*"
        parts.append(f"\n### Response\n\n{response}")
    return "\n\n---\n\n".join(parts)


def run_arena(selected, system_prompt, user_prompt, temperature, top_p, top_k, repeat_penalty, num_ctx,
              num_predict, seed, think, anonymous, show_thinking):
    if not selected:
        yield "請至少選一個模型。", "", None, None, None, [], {}; return
    if not user_prompt.strip():
        yield "User Prompt 不可空白。", "", None, None, None, [], {}; return
    _, mapping = model_choices(); models = [mapping.get(item, item.split("  —  ")[0]) for item in selected]
    if anonymous: random.shuffle(models)
    settings = build_settings(temperature, top_p, top_k, repeat_penalty, num_ctx, num_predict, seed, think)
    run_dir = unique_run_dir(datetime.now().strftime("%Y%m%d_%H%M%S")); run_id = run_dir.name
    results, mapping_out = [], {}
    for index, model in enumerate(models, 1):
        alias = f"Model {chr(64 + index)}" if anonymous else model; mapping_out[alias] = model
        yield f"正在執行 {index}/{len(models)}：{alias}", render_results(results, not anonymous, show_thinking), None, None, None, results, mapping_out
        result = generate(model, system_prompt, user_prompt, settings); result["think"] = bool(think); result["display_name"] = alias; results.append(result)
        partial = {"run_id": run_id, "mode": "single_turn", "system_prompt": system_prompt, "user_prompt": user_prompt,
                   "settings": settings, "anonymous": anonymous, "mapping": mapping_out, "results": results}
        paths = export_run(partial, run_dir)
        yield f"完成 {index}/{len(models)}：{alias} — {status_label(result)}", render_results(results, not anonymous, show_thinking), *paths, results, mapping_out


def reveal(results, mapping, show_thinking):
    if not results: return ""
    return render_results(results, True, show_thinking) + "\n\n### Anonymous mapping\n" + "\n".join(f"- {k} = `{v}`" for k, v in mapping.items())


# ---------- Multi-turn ----------
MAX_ROUNDS = 20


def normalize_round_texts(round_count, *round_texts):
    """Return the visible, non-empty round messages in display order."""
    try:
        count = max(1, min(int(round_count or 1), MAX_ROUNDS))
    except Exception:
        count = 1
    out = []
    for text in round_texts[:count]:
        text = "" if text is None else str(text).strip()
        if text:
            out.append(text)
    return out


def _round_visibility_updates(count):
    return [gr.update(visible=i < count) for i in range(MAX_ROUNDS)]


def add_round_boxes(round_count, *round_texts):
    try:
        count = int(round_count or 1)
    except Exception:
        count = 1
    count = min(MAX_ROUNDS, max(1, count + 1))
    status = round_stats_boxes(count, *round_texts)
    return (count, *_round_visibility_updates(count), status)


def delete_round_boxes(round_count, *round_texts):
    try:
        count = int(round_count or 1)
    except Exception:
        count = 1
    count = max(1, count - 1)
    # Keep text in hidden rounds so an accidental delete can be undone by Add.
    status = round_stats_boxes(count, *round_texts)
    return (count, *_round_visibility_updates(count), status)


def round_stats_boxes(round_count, *round_texts):
    try:
        count = max(1, min(int(round_count or 1), MAX_ROUNDS))
    except Exception:
        count = 1
    chars = sum(len(str(x or "")) for x in round_texts[:count])
    filled = sum(1 for x in round_texts[:count] if str(x or "").strip())
    return f"目前 {count} 輪　|　已填 {filled} 輪　|　User 訊息共 {chars:,} 字元"


def render_multiturn(sessions, revealed=True, show_thinking=False):
    parts = []
    for session in sessions:
        title = session["model"] if revealed else session["display_name"]
        parts.append(f"# {title}")
        for rr in session.get("rounds", []):
            result = rr["result"]
            parts.append(f"## Round {rr['round']}\n\n**User**\n\n{rr['user']}\n\n**Assistant — {status_label(result)}**  \n"
                         f"Tokens: {result.get('output_tokens')}　Speed: {result.get('tokens_per_second')} tok/s　"
                         f"Wall: {result.get('wall_clock_seconds')}s")
            if show_thinking and result.get("thinking"):
                parts.append(f"\n<details><summary>Thinking</summary>\n\n{result['thinking']}\n\n</details>")
            parts.append(result.get("response") or ("⚠️ EMPTY FINAL RESPONSE" if result.get("status") == "empty_final" else ""))
        stats = session.get("stats", {})
        parts.append(f"### 累計\nRounds: {stats.get('rounds_completed', 0)}　Output tokens: {stats.get('output_tokens', 0)}　"
                     f"Wall: {stats.get('wall_seconds', 0):.2f}s　Avg speed: {stats.get('avg_tokens_per_second')} tok/s")
    return "\n\n---\n\n".join(parts)


def recompute_session_stats(session):
    rounds = session.get("rounds", [])
    tokens = sum((r["result"].get("output_tokens") or 0) for r in rounds)
    wall = sum((r["result"].get("wall_clock_seconds") or 0) for r in rounds)
    gen = sum((r["result"].get("eval_duration_seconds") or 0) for r in rounds)
    avg = round(tokens / gen, 3) if gen else None
    session["stats"] = {"rounds_completed": len(rounds), "output_tokens": tokens, "wall_seconds": wall,
                        "generation_seconds": gen, "avg_tokens_per_second": avg}


def export_multiturn(run, run_dir):
    (run_dir / "multiturn_result.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Ollama Model Arena — Multi-turn", "", f"- Run: `{run['run_id']}`", f"- Anonymous: `{run['anonymous']}`",
          f"- Rounds: `{len(run['user_rounds'])}`", "", "## System Prompt", run["system_prompt"] or "(空)",
          "", "## Settings", "```json", json.dumps(run["settings"], ensure_ascii=False, indent=2), "```"]
    for session in run["sessions"]:
        md += ["", f"# {session['display_name']}", f"- Actual model: `{session['model']}`"]
        for rr in session["rounds"]:
            r = rr["result"]
            md += ["", f"## Round {rr['round']}", "", "### User", "", rr["user"], "", "### Assistant",
                   f"- Status: **{r.get('status')}**", f"- Done reason: `{r.get('done_reason')}`",
                   f"- Thinking: **{'ON' if r.get('think') else 'OFF'}**", f"- Wall: {r.get('wall_clock_seconds')} s",
                   f"- Prompt tokens: {r.get('prompt_tokens')}", f"- Output tokens: {r.get('output_tokens')}",
                   f"- Speed: {r.get('tokens_per_second')} tok/s", f"- Error: {r.get('error') or 'None'}"]
            if r.get("thinking"): md += ["", "#### Thinking", "", r["thinking"]]
            md += ["", r.get("response", "")]
        md += ["", "### Session totals", "```json", json.dumps(session.get("stats", {}), ensure_ascii=False, indent=2), "```"]
    (run_dir / "multiturn_result.md").write_text("\n".join(md), encoding="utf-8")

    fields = ["display_name", "model", "round", "user", "status", "done_reason", "think", "wall_clock_seconds",
              "prompt_tokens", "output_tokens", "tokens_per_second", "error", "thinking", "response"]
    with (run_dir / "multiturn_result.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for session in run["sessions"]:
            for rr in session["rounds"]:
                r = rr["result"]
                row = {"display_name": session["display_name"], "model": session["model"], "round": rr["round"], "user": rr["user"]}
                row.update({k: r.get(k) for k in fields if k not in row})
                writer.writerow(row)
    return str(run_dir / "multiturn_result.md"), str(run_dir / "multiturn_result.json"), str(run_dir / "multiturn_result.csv")


def run_multiturn(selected, system_prompt, temperature, top_p, top_k, repeat_penalty, num_ctx,
                  num_predict, seed, think, anonymous, show_thinking, round_count, *round_texts):
    if not selected:
        yield "請至少選一個模型。", "", None, None, None, [], {}; return
    user_rounds = normalize_round_texts(round_count, *round_texts)
    if not user_rounds:
        yield "請至少填寫一輪 User Message。", "", None, None, None, [], {}; return

    _, mapping = model_choices(); models = [mapping.get(item, item.split("  —  ")[0]) for item in selected]
    if anonymous: random.shuffle(models)
    settings = build_settings(temperature, top_p, top_k, repeat_penalty, num_ctx, num_predict, seed, think)
    run_dir = unique_run_dir(datetime.now().strftime("%Y%m%d_%H%M%S") + "_multiturn"); run_id = run_dir.name
    sessions, mapping_out = [], {}

    for index, model in enumerate(models, 1):
        alias = f"Model {chr(64 + index)}" if anonymous else model; mapping_out[alias] = model
        session = {"model": model, "display_name": alias, "rounds": [], "stats": {}}
        sessions.append(session)
        messages = []
        if system_prompt.strip(): messages.append({"role": "system", "content": system_prompt})
        for round_idx, user_text in enumerate(user_rounds, 1):
            yield (f"{alias} — Round {round_idx}/{len(user_rounds)}", render_multiturn(sessions, not anonymous, show_thinking),
                   None, None, None, sessions, mapping_out)
            messages.append({"role": "user", "content": user_text})
            result = generate_messages(model, messages, settings); result["think"] = bool(think); result["display_name"] = alias
            session["rounds"].append({"round": round_idx, "user": user_text, "result": result})
            if result.get("response", "").strip():
                messages.append({"role": "assistant", "content": result["response"]})
            recompute_session_stats(session)

            partial = {"run_id": run_id, "mode": "multi_turn", "system_prompt": system_prompt, "user_rounds": user_rounds,
                       "settings": settings, "anonymous": anonymous, "mapping": mapping_out, "sessions": sessions}
            paths = export_multiturn(partial, run_dir)
            yield (f"完成 {alias} — Round {round_idx}: {status_label(result)}", render_multiturn(sessions, not anonymous, show_thinking),
                   *paths, sessions, mapping_out)

            # If a turn produced no usable assistant message, do not fabricate history for later rounds.
            # Stop only this model; the rest of the arena continues.
            if result.get("status") in {"error", "empty_final"}:
                session["halted_after_round"] = round_idx
                session["halt_reason"] = result.get("status")
                break


def reveal_multiturn(sessions, mapping, show_thinking):
    if not sessions: return ""
    return render_multiturn(sessions, True, show_thinking) + "\n\n### Anonymous mapping\n" + "\n".join(f"- {k} = `{v}`" for k, v in mapping.items())


def clear_single(): return "", None, None, None, [], {}, "畫面已清除；歷史 exports 不會刪除。"
def clear_multi(): return "", None, None, None, [], {}, "畫面已清除；歷史 exports 不會刪除。"


def latest_history():
    dirs = sorted([p for p in EXPORTS.iterdir() if p.is_dir()], reverse=True)[:10]
    if not dirs: return "尚無紀錄"
    lines = ["### 最近 10 次輸出"]
    for path in dirs:
        mode = "multi" if (path / "multiturn_result.json").exists() else "single"
        result_path = path / ("multiturn_result.json" if mode == "multi" else "result.json")
        count = "?"
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            count = len(data.get("sessions", [])) if mode == "multi" else len(data.get("results", []))
        except Exception: pass
        lines.append(f"- `{path.name}` — {mode} — {count} models")
    return "\n".join(lines)



MOBILE_CSS = r"""
/* Model selector: readable on phones instead of a narrow half-column popover. */
.model-row { align-items: flex-end; }
.model-picker { min-width: 0; flex: 1 1 auto; }
.model-refresh { min-width: 180px; }
.model-preview { margin-top: -4px; font-size: 0.92rem; }

@media (max-width: 768px) {
  .model-row {
    flex-direction: column !important;
    gap: 8px !important;
    align-items: stretch !important;
  }
  .model-row > * {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
  }
  .model-picker, .model-refresh { width: 100% !important; }
  .model-refresh button { min-height: 46px !important; }
  .model-picker input, .model-picker textarea { font-size: 16px !important; }
  .model-picker [role=option],
  .model-picker li {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    line-height: 1.35 !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
  }
  .model-preview {
    max-height: 180px;
    overflow-y: auto;
    padding: 8px 10px;
    border-radius: 8px;
  }
  .mobile-title h1 { font-size: 1.72rem !important; line-height: 1.15 !important; }
  .mobile-title p { font-size: 0.96rem !important; }
}
"""

def generation_controls(prefix=""):
    preset = gr.Dropdown(choices=list(PRESETS), value="安全長輸出 / 推理", label="快速預設")
    apply_btn = gr.Button("套用預設"); save_btn = gr.Button("💾 儲存目前設定為預設")
    with gr.Row():
        temperature = gr.Slider(0, 2, value=dflt("temperature", 0.7), step=.05, label="Temperature")
        top_p = gr.Slider(0, 1, value=dflt("top_p", 0.9), step=.01, label="Top P")
        top_k = gr.Number(value=dflt("top_k", 40), precision=0, label="Top K")
        repeat = gr.Number(value=dflt("repeat_penalty", 1.1), label="Repeat penalty")
    with gr.Row():
        ctx = gr.Number(value=dflt("num_ctx", 32768), precision=0, label="Context")
        predict = gr.Number(value=dflt("num_predict", 8192), precision=0, label="Max output tokens")
        seed = gr.Number(value=dflt("seed", 42), precision=0, label="Seed")
        think = gr.Checkbox(value=bool(dflt("think", False)), label="🧠 思考模式（Thinking）")
    warning = gr.Markdown(settings_warning(bool(dflt("think", False)), dflt("num_predict", 8192)))
    save_status = gr.Markdown("")
    apply_btn.click(apply_preset, inputs=preset, outputs=[temperature, top_p, top_k, repeat, ctx, predict, seed, think, warning])
    save_btn.click(save_preferences, inputs=[temperature, top_p, top_k, repeat, ctx, predict, seed, think], outputs=save_status)
    think.change(settings_warning, inputs=[think, predict], outputs=warning); predict.change(settings_warning, inputs=[think, predict], outputs=warning)
    return temperature, top_p, top_k, repeat, ctx, predict, seed, think


with gr.Blocks(title="Ollama Model Arena", css=MOBILE_CSS) as demo:
    gr.Markdown("# 🥊 Ollama Multi-Model Arena v1.8.3\n單輪比較與多輪 RP / 對話測試分成兩個分頁；多輪模式會為每顆模型獨立累積 conversation history。", elem_classes=["mobile-title"])

    with gr.Tabs():
        with gr.Tab("📝 單輪 Arena"):
            s_status = gr.Markdown("準備中")
            with gr.Row(elem_classes=["model-row"]):
                s_models = gr.Dropdown(
                    label="選擇本機 Ollama 模型（可多選）", multiselect=True, choices=[],
                    filterable=True, elem_classes=["model-picker"],
                )
                s_refresh = gr.Button("🔄 重新整理模型", elem_classes=["model-refresh"])
            s_model_preview = gr.Markdown("尚未選擇模型。", elem_classes=["model-preview"])
            s_system = gr.Textbox(label="System Prompt（可留空）", lines=5)
            s_user = gr.Textbox(label="User Prompt", lines=12)
            s_stats = gr.Markdown("System：0 字元　|　User：0 字元")
            with gr.Accordion("Generation Settings", open=True):
                s_temp, s_top_p, s_top_k, s_repeat, s_ctx, s_predict, s_seed, s_think = generation_controls("single")
            with gr.Row():
                s_anon = gr.Checkbox(value=False, label="匿名 Arena Mode")
                s_show = gr.Checkbox(value=False, label="顯示 Thinking 內容")
            with gr.Row():
                s_run = gr.Button("🚀 送出背景任務", variant="primary"); s_clear = gr.Button("🧹 清除畫面")
            s_results = gr.Markdown(); s_reveal = gr.Button("👀 Reveal Models")
            with gr.Row():
                s_md = gr.File(label="Markdown"); s_json = gr.File(label="JSON"); s_csv = gr.File(label="CSV")
            s_state = gr.State([]); s_map = gr.State({})

        with gr.Tab("💬 多輪對話 / RP"):
            m_status = gr.Markdown("準備中")
            gr.Markdown("每一輪都用完整多行輸入框編輯；新增好 Round 1、2、3… 的 User 訊息後，再一次跑所有模型。每顆模型只會看到**自己的前輪回答**，模型彼此完全隔離。")
            with gr.Row(elem_classes=["model-row"]):
                m_models = gr.Dropdown(
                    label="選擇本機 Ollama 模型（可多選）", multiselect=True, choices=[],
                    filterable=True, elem_classes=["model-picker"],
                )
                m_refresh = gr.Button("🔄 重新整理模型", elem_classes=["model-refresh"])
            m_model_preview = gr.Markdown("尚未選擇模型。", elem_classes=["model-preview"])
            m_system = gr.Textbox(label="System Prompt / 角色設定（可留空）", lines=8)
            gr.Markdown("### 對話輪次\n每一輪都是完整的多行輸入框，不必在表格小格子裡編輯。可用下方按鈕新增／刪除輪次。")
            m_round_count = gr.State(1)
            m_round_boxes = []
            for i in range(MAX_ROUNDS):
                box = gr.Textbox(
                    label=f"Round {i + 1} — User Message",
                    lines=7,
                    max_lines=20,
                    visible=(i == 0),
                    placeholder=f"輸入 Round {i + 1} 的完整 User 訊息…",
                )
                m_round_boxes.append(box)
            with gr.Row():
                m_add = gr.Button("➕ 新增一輪"); m_del = gr.Button("➖ 刪除最後一輪")
            m_round_stats = gr.Markdown(f"目前 1 輪　|　已填 0 輪　|　User 訊息共 0 字元（最多 {MAX_ROUNDS} 輪）")
            with gr.Accordion("Generation Settings", open=True):
                m_temp, m_top_p, m_top_k, m_repeat, m_ctx, m_predict, m_seed, m_think = generation_controls("multi")
            with gr.Row():
                m_anon = gr.Checkbox(value=False, label="匿名 Arena Mode")
                m_show = gr.Checkbox(value=False, label="顯示 Thinking 內容")
            with gr.Row():
                m_run = gr.Button("🚀 送出多輪背景任務", variant="primary"); m_clear = gr.Button("🧹 清除結果")
            m_results = gr.Markdown(); m_reveal = gr.Button("👀 Reveal Models")
            with gr.Row():
                m_md = gr.File(label="Markdown"); m_json = gr.File(label="JSON"); m_csv = gr.File(label="CSV")
            m_state = gr.State([]); m_map = gr.State({})

        with gr.Tab("📋 背景任務"):
            gr.Markdown("手機送出測試後可以直接離開 Safari。任務由 Mac 上的背景 worker 繼續執行，結果會逐模型／逐輪寫入 `exports/`。")
            with gr.Row():
                j_select = gr.Dropdown(label="背景任務", choices=[], filterable=True)
                j_refresh = gr.Button("🔄 更新任務列表")
            j_list_status = gr.Markdown("尚未讀取任務。")
            gr.Markdown(
                "### 任務設定（唯讀）\n"
                "Thinking 與匿名模式在送出任務時就已固定；下方選項只控制結果怎麼顯示，不會修改或重新執行任務。"
            )
            gr.Markdown("### 結果顯示 / 衍生輸出")
            gr.Markdown(
                "下面兩個選項只控制預覽與新產生的衍生輸出。"
                "**原始 Markdown / JSON / CSV 永遠保留，不會被覆蓋。**"
            )
            with gr.Row():
                j_reveal = gr.Checkbox(value=True, label="顯示真實模型名稱")
                j_show = gr.Checkbox(value=True, label="顯示推理紀錄")
            with gr.Row():
                j_load = gr.Button("📄 查看 / 更新結果", variant="primary")
                j_mask = gr.Button("🎭 產生遮罩 / 衍生輸出")
                j_cancel = gr.Button("🛑 停止任務")
            j_cancel_status = gr.Markdown("")
            j_mask_status = gr.Markdown("")
            j_status = gr.Markdown("")
            j_results = gr.Markdown("")
            gr.Markdown("#### 原始結果（永遠保留）")
            with gr.Row():
                j_md = gr.File(label="原始 Markdown")
                j_json = gr.File(label="原始 JSON")
                j_csv = gr.File(label="原始 CSV")
            gr.Markdown("#### 遮罩 / 衍生輸出")
            with gr.Row():
                j_mask_md = gr.File(label="衍生 Markdown")
                j_mask_json = gr.File(label="衍生 JSON")
                j_mask_csv = gr.File(label="衍生 CSV")

    history = gr.Markdown()

    # shared refresh
    s_refresh.click(refresh_both, outputs=[s_models, m_models, s_status, m_status])
    m_refresh.click(refresh_both, outputs=[s_models, m_models, s_status, m_status])
    demo.load(refresh_both, outputs=[s_models, m_models, s_status, m_status])
    demo.load(latest_history, outputs=history)
    s_models.change(selected_models_preview, inputs=s_models, outputs=s_model_preview)
    m_models.change(selected_models_preview, inputs=m_models, outputs=m_model_preview)

    # single events
    s_system.change(prompt_stats, inputs=[s_system, s_user], outputs=s_stats); s_user.change(prompt_stats, inputs=[s_system, s_user], outputs=s_stats)
    s_run.click(submit_single_job, inputs=[s_models, s_system, s_user, s_temp, s_top_p, s_top_k, s_repeat, s_ctx, s_predict, s_seed, s_think, s_anon, s_show],
                outputs=s_status)
    s_reveal.click(reveal, inputs=[s_state, s_map, s_show], outputs=s_results)
    s_show.change(reveal, inputs=[s_state, s_map, s_show], outputs=s_results)
    s_clear.click(clear_single, outputs=[s_results, s_md, s_json, s_csv, s_state, s_map, s_status])

    # multi events
    _round_outputs = [m_round_count] + m_round_boxes + [m_round_stats]
    _round_inputs = [m_round_count] + m_round_boxes
    m_add.click(add_round_boxes, inputs=_round_inputs, outputs=_round_outputs)
    m_del.click(delete_round_boxes, inputs=_round_inputs, outputs=_round_outputs)
    for _box in m_round_boxes:
        _box.change(round_stats_boxes, inputs=_round_inputs, outputs=m_round_stats)
    m_run.click(submit_multi_job, inputs=[m_models, m_system, m_temp, m_top_p, m_top_k, m_repeat, m_ctx, m_predict, m_seed, m_think, m_anon, m_show, m_round_count] + m_round_boxes,
                outputs=m_status)
    m_reveal.click(reveal_multiturn, inputs=[m_state, m_map, m_show], outputs=m_results)
    m_show.change(reveal_multiturn, inputs=[m_state, m_map, m_show], outputs=m_results)
    m_clear.click(clear_multi, outputs=[m_results, m_md, m_json, m_csv, m_state, m_map, m_status])

    # background job events
    demo.load(refresh_job_list, inputs=j_select, outputs=[j_select, j_list_status])
    j_refresh.click(refresh_job_list, inputs=j_select, outputs=[j_select, j_list_status])
    j_load.click(load_job, inputs=[j_select, j_reveal, j_show], outputs=[j_status, j_results, j_md, j_json, j_csv])
    j_select.change(load_job, inputs=[j_select, j_reveal, j_show], outputs=[j_status, j_results, j_md, j_json, j_csv])
    j_reveal.change(load_job, inputs=[j_select, j_reveal, j_show], outputs=[j_status, j_results, j_md, j_json, j_csv])
    j_show.change(load_job, inputs=[j_select, j_reveal, j_show], outputs=[j_status, j_results, j_md, j_json, j_csv])
    j_mask.click(
        create_masked_export,
        inputs=[j_select, j_reveal, j_show],
        outputs=[j_mask_status, j_mask_md, j_mask_json, j_mask_csv],
    )
    j_cancel.click(request_cancel, inputs=j_select, outputs=j_cancel_status)


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        server_name=os.environ.get("OMA_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("OMA_SERVER_PORT", "7860")),
    )
