# Ollama Model Arena v1.4

本機 Ollama 多模型 A/B 測試 Gradio UI。模型採**逐顆串行**生成，不會同時載入多顆模型。

## v1.4 主要更新

### 兩個獨立分頁

- **單輪 Arena**：完整保留 v1.3 的單題多模型比較方式。
- **多輪對話 / RP**：專門測試連續角色扮演與分段式對話。

### 多輪對話 / RP

- Round 數量可自行新增、刪除，不限制 3 輪。
- 每一輪可先填入獨立 User Message。
- 每顆模型維護**自己的 conversation history**：
  - System
  - User Round 1
  - 該模型 Assistant Round 1
  - User Round 2
  - 該模型 Assistant Round 2
  - ...
- 模型彼此 history 完全隔離，不會看到其他模型答案。
- 每輪保存 `completed / truncated / empty_final / error`。
- `error` 或 `empty_final` 時只停止該模型後續輪次，其餘模型繼續。
- `truncated` 保留完整已生成內容並明確標記。
- 每輪保存 tokens、速度、wall time、thinking 內容與 metadata。
- 每顆模型顯示累計 rounds、output tokens、總時間與平均生成速度。
- 多輪結果獨立輸出 `multiturn_result.md / .json / .csv`。
- 每完成一輪立刻保存 partial results，途中關閉也不會把前面結果弄丟。

## v1.3 已有功能仍保留

- `completed` / `truncated` / `empty_final` / `error`
- `done_reason=length` 明確標示 **TRUNCATED — OUTPUT TOKEN LIMIT**
- 不自動 retry
- 預設 `num_predict=8192`、`num_ctx=32768`、`repeat_penalty=1.1`
- Thinking toggle 正確使用 Ollama request top-level `think`
- Markdown / JSON / CSV 保存 status、done reason、tokens、速度與 thinking
- 三個快速預設
- generation settings 可保存為下次啟動預設
- 匿名 Arena Mode
- 最近輸出紀錄

## 安裝 / 啟動

若原本 v1.3 已經在此資料夾建立 `.venv`，更新程式後可直接：

```bash
cd /Users/angelamao/Documents/ollama-model-arena
source .venv/bin/activate
python app.py
```

全新安裝：

```bash
cd /Users/angelamao/Documents/ollama-model-arena
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

## 建議設定

一般長文 / 推理：

```text
temperature=0.7
top_p=0.9
top_k=40
repeat_penalty=1.1
num_ctx=32768
num_predict=8192
seed=42
think=false
```

創作 / RP 可使用 `num_predict=4096`；長文或 Thinking 建議 `8192`。
