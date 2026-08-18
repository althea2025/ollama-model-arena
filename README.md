# Ollama Model Arena

A lightweight local multi-model benchmarking arena for **Ollama**, built with **Gradio**.

Compare multiple local LLMs with the same prompt, test multi-turn conversations while keeping each model's history isolated, toggle thinking mode, inspect generation metrics, and export results for later evaluation.

> Current release: **v1.8.3**

## Features

### Background Jobs for Remote / Mobile Use

Single-turn and multi-turn runs are submitted to a server-side background worker. After a job is submitted, the browser connection no longer needs to stay open.

- Submit a test from a phone, then switch apps or close the Safari tab
- The Mac keeps running the Ollama job in the background
- Progress is persisted in `exports/<run-id>/job.json`
- Partial results are written after every completed model or round
- Reopen the Arena later and use the **Background Jobs** tab to inspect status and download Markdown / JSON / CSV results
- Jobs run serially to avoid loading multiple large local models at the same time
- A running job can be asked to stop after the current Ollama generation finishes

The Arena application itself must remain running. If the Arena process or Mac shuts down, an in-progress job cannot continue. Remote mode uses `caffeinate` on macOS when available to prevent idle sleep while the Arena is running.


### Single-turn Arena

Run the same prompt against multiple Ollama models in one batch.

- Load locally installed models from `ollama list`
- Select multiple models for comparison
- Sequential generation to avoid loading several large models at the same time
- Optional anonymous Arena mode
- Shared generation settings for fair comparisons
- Thinking mode ON / OFF
- Per-model status, token counts, timing, and generation speed
- Export results to Markdown, JSON, and CSV

### Multi-turn Conversation / RP Arena

Designed for role-play, character consistency, instruction-following, and conversational benchmark tests.

- Add or remove conversation rounds as needed
- Each round has its own full-size User Message editor
- Every model maintains its **own independent conversation history**
- Models never see another model's answers
- Each round carries forward that model's previous assistant response
- Per-round status and generation metrics
- Partial results are saved as rounds finish
- Useful for testing:
  - role-play quality
  - character consistency
  - user agency
  - context retention
  - multi-turn instruction following

Conversation history is maintained independently for each model:

```text
System
User Round 1
Assistant Round 1
User Round 2
Assistant Round 2
User Round 3
...
```

### Thinking Mode

Thinking can be enabled or disabled from the UI.

The setting is sent through Ollama's top-level `think` request field:

```text
think=true
```

or:

```text
think=false
```

When available, thinking content is stored separately from the final response.

### Benchmark-friendly Result Status

Each generation is classified as:

- `completed` — final response is present and generation finished normally
- `truncated` — final response is present, but generation stopped because the output token limit was reached
- `empty_final` — the request succeeded but the final response is empty
- `error` — the Ollama API request failed

`truncated` results are preserved and clearly marked instead of being silently treated as complete.

The app does **not** automatically retry failed or truncated generations, helping keep benchmark conditions reproducible.

### Generation Metrics

Results can include:

- output tokens
- prompt tokens
- wall time
- load time
- prompt evaluation time
- generation time
- tokens per second
- Ollama timing metadata
- done reason
- thinking state
- thinking content when returned by the model

## Default Benchmark Settings

The default settings are designed for longer benchmark responses:

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

These values can be changed in the UI.

For shorter creative or RP tests, `num_predict=4096` may be sufficient. For long-form output or Thinking mode, `8192` is recommended to reduce the risk of truncation.

## Requirements

- macOS, Linux, or another environment capable of running Python and Ollama
- Python 3
- Ollama installed and running
- At least one local Ollama model

Ollama must be available from the command line:

```bash
ollama list
```

## Installation

Clone the repository:

```bash
git clone https://github.com/althea2025/ollama-model-arena.git
cd ollama-model-arena
```

### macOS / Linux

Create an isolated Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch the app:

```bash
python app.py
```

### Windows

Create an isolated Python virtual environment:

```bat
py -m venv .venv
```

Install dependencies directly into that environment:

```bat
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Launch the app:

```bat
.venv\Scripts\python.exe app.py
```

If the `py` launcher is unavailable, use `python` instead.

Gradio will print a local address, usually:

```text
http://127.0.0.1:7860
```

Open that address in your browser.

## macOS Quick Launcher

A `start.command` launcher is included.

After completing the installation above, give it execute permission once:

```bash
chmod +x start.command
```

You can then launch Ollama Model Arena by double-clicking `start.command` in Finder.

If macOS blocks it the first time, right-click `start.command`, choose **Open**, and confirm **Open**.

The `.venv` must already exist and the dependencies must already be installed.

## Windows Quick Launcher

A `start.bat` launcher is included for Windows.

After completing the Windows installation above, double-click `start.bat` in File Explorer. No activation command is required: the launcher uses `.venv\Scripts\python.exe` directly.

If `.venv` has not been created yet, the launcher will show the required setup commands instead of closing silently.

## Updating

If you installed the project with Git:

```bash
git pull
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then launch normally with:

```bash
python app.py
```

or double-click `start.command` on macOS.

## Mobile UI

The interface is responsive on phones and tablets. In v1.7, model selection is optimized for narrow screens:

- the model picker and refresh button stack vertically on mobile
- model labels use shorter readable names
- long choices wrap instead of being clipped
- selected models are shown below the picker using their exact Ollama tags

## Remote Access with Tailscale

macOS includes two launchers:

- `start.command` — local-only mode
- `start-remote.command` — network/Tailscale mode (`0.0.0.0:7860`)

Give the remote launcher execute permission once:

```bash
chmod +x start-remote.command
```

Connect your Mac and phone to the same Tailscale network, then open:

```text
http://<Mac-Tailscale-IP>:7860
```

Do not expose port `7860` directly to the public Internet.

## Background Jobs result display

Background tasks keep the generation settings that were fixed when the job was submitted.

In the **Background Jobs** tab:

- Thinking mode is shown as read-only task metadata.
- Anonymous mode is shown as read-only task metadata.
- **Show real model names** is enabled by default and only changes how results are displayed.
- **Show reasoning record** is enabled by default and only changes how results are displayed.
- Turning either display option off does not alter the job, regenerate anything, or modify the original saved result files.


## Non-destructive masked / derived exports

The Background Jobs page can generate additional export files from an existing result **without rerunning any model**.

Original files remain untouched. Derived variants are written into the run's own `masked/` subfolder.

```text
exports/<run-id>/
├── result.* or multiturn_result.*   # original source of truth
└── masked/
    ├── *_full_copy.*
    ├── *_models_masked.*
    ├── *_thinking_hidden.*
    └── *_blind.*
```

- `full_copy`: real model names and reasoning shown
- `models_masked`: model names replaced with neutral aliases
- `thinking_hidden`: reasoning content removed
- `blind`: both model names and reasoning content hidden

Creating a derived export only processes saved files. It does not call Ollama and never overwrites the original result files.

## Output

Generated benchmark results are stored locally and can be exported in formats such as:

- Markdown
- JSON
- CSV

Single-turn and multi-turn results are kept separate.

Local generated results are intentionally excluded from the repository by `.gitignore`.

## Privacy

Ollama Model Arena is designed for **local model testing**.

Prompts and model responses are sent to the Ollama server configured for the app. When using the default local Ollama setup, model inference remains on your local machine.

Before sharing exported benchmark files publicly, review them for prompts, model responses, model names, or other information you may not want to publish.

## Project Structure

```text
ollama-model-arena/
├── app.py
├── requirements.txt
├── start.command       # macOS local launcher
├── start-remote.command # macOS Tailscale / LAN launcher
├── start.bat           # Windows launcher
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
├── tests/
└── docs/
```

Runtime files such as `.venv/`, generated exports, caches, and local settings should not be committed.

## Contributing

Bug reports, feature requests, and pull requests are welcome.

See `CONTRIBUTING.md` for contribution guidelines.

## License

Released under the **MIT License**. See `LICENSE` for details.

## Release

**v1.8.3**

Highlights:

- Single-turn multi-model Arena
- Multi-turn Conversation / RP Arena
- Independent conversation history per model
- Thinking mode
- Benchmark-safe generation status handling
- Token and performance metrics
- Markdown / JSON / CSV export
- macOS local / remote and Windows quick launchers
- Persistent background jobs for mobile / remote use
