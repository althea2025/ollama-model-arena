from __future__ import annotations

import csv
import json
import random
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


def model_choices():
    models = list_models()
    labels, mapping = [], {}
    for model in models:
        gb = f"{model['size_gb']} GB" if model["size_gb"] is not None else "size ?"
        label = f"{model['name']}  —  {gb}"
        labels.append(label)
        mapping[label] = model["name"]
    return labels, mapping


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


with gr.Blocks(title="Ollama Model Arena") as demo:
    gr.Markdown("# 🥊 Ollama Multi-Model Arena v1.6\n單輪比較與多輪 RP / 對話測試分成兩個分頁；多輪模式會為每顆模型獨立累積 conversation history。")

    with gr.Tabs():
        with gr.Tab("📝 單輪 Arena"):
            s_status = gr.Markdown("準備中")
            with gr.Row():
                s_models = gr.Dropdown(label="選擇本機 Ollama 模型（可多選）", multiselect=True, choices=[])
                s_refresh = gr.Button("🔄 重新整理模型")
            s_system = gr.Textbox(label="System Prompt（可留空）", lines=5)
            s_user = gr.Textbox(label="User Prompt", lines=12)
            s_stats = gr.Markdown("System：0 字元　|　User：0 字元")
            with gr.Accordion("Generation Settings", open=True):
                s_temp, s_top_p, s_top_k, s_repeat, s_ctx, s_predict, s_seed, s_think = generation_controls("single")
            with gr.Row():
                s_anon = gr.Checkbox(value=False, label="匿名 Arena Mode")
                s_show = gr.Checkbox(value=False, label="顯示 Thinking 內容")
            with gr.Row():
                s_run = gr.Button("🚀 開始逐顆生成", variant="primary"); s_clear = gr.Button("🧹 清除畫面")
            s_results = gr.Markdown(); s_reveal = gr.Button("👀 Reveal Models")
            with gr.Row():
                s_md = gr.File(label="Markdown"); s_json = gr.File(label="JSON"); s_csv = gr.File(label="CSV")
            s_state = gr.State([]); s_map = gr.State({})

        with gr.Tab("💬 多輪對話 / RP"):
            m_status = gr.Markdown("準備中")
            gr.Markdown("每一輪都用完整多行輸入框編輯；新增好 Round 1、2、3… 的 User 訊息後，再一次跑所有模型。每顆模型只會看到**自己的前輪回答**，模型彼此完全隔離。")
            with gr.Row():
                m_models = gr.Dropdown(label="選擇本機 Ollama 模型（可多選）", multiselect=True, choices=[])
                m_refresh = gr.Button("🔄 重新整理模型")
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
                m_run = gr.Button("🚀 開始多輪測試", variant="primary"); m_clear = gr.Button("🧹 清除結果")
            m_results = gr.Markdown(); m_reveal = gr.Button("👀 Reveal Models")
            with gr.Row():
                m_md = gr.File(label="Markdown"); m_json = gr.File(label="JSON"); m_csv = gr.File(label="CSV")
            m_state = gr.State([]); m_map = gr.State({})

    history = gr.Markdown()

    # shared refresh
    s_refresh.click(refresh_both, outputs=[s_models, m_models, s_status, m_status])
    m_refresh.click(refresh_both, outputs=[s_models, m_models, s_status, m_status])
    demo.load(refresh_both, outputs=[s_models, m_models, s_status, m_status])
    demo.load(latest_history, outputs=history)

    # single events
    s_system.change(prompt_stats, inputs=[s_system, s_user], outputs=s_stats); s_user.change(prompt_stats, inputs=[s_system, s_user], outputs=s_stats)
    s_run.click(run_arena, inputs=[s_models, s_system, s_user, s_temp, s_top_p, s_top_k, s_repeat, s_ctx, s_predict, s_seed, s_think, s_anon, s_show],
                outputs=[s_status, s_results, s_md, s_json, s_csv, s_state, s_map])
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
    m_run.click(run_multiturn, inputs=[m_models, m_system, m_temp, m_top_p, m_top_k, m_repeat, m_ctx, m_predict, m_seed, m_think, m_anon, m_show, m_round_count] + m_round_boxes,
                outputs=[m_status, m_results, m_md, m_json, m_csv, m_state, m_map])
    m_reveal.click(reveal_multiturn, inputs=[m_state, m_map, m_show], outputs=m_results)
    m_show.change(reveal_multiturn, inputs=[m_state, m_map, m_show], outputs=m_results)
    m_clear.click(clear_multi, outputs=[m_results, m_md, m_json, m_csv, m_state, m_map, m_status])


if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=7860)
