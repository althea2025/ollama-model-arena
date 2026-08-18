# Changelog

## v1.8.3 - 2026-08-18

- Added non-destructive masked / derived exports in Background Jobs.
- Original result MD / JSON / CSV are never overwritten.
- Derived exports are stored under each run's `masked/` folder.
- Viewer/export controls can independently hide real model names and reasoning content.
- Added full_copy, models_masked, thinking_hidden, and blind variants.
- Creating derived exports never calls Ollama or reruns a benchmark.
- Launcher files were not changed.


## v1.8.2 - 2026-08-18

- Clarified Background Jobs as a monitoring/results page.
- Thinking and anonymous mode are shown as read-only job settings.
- Result display controls are separate from execution settings.
- "Show real model names" now defaults to ON.
- "Show reasoning record" now defaults to ON.
- Turning either display option off only changes the viewer and never changes the job.
- No launcher files were changed.


## v1.8 - 2026-08-18

- Added persistent server-side background jobs for single-turn and multi-turn tests.
- Browser / phone may disconnect after submission while the Mac continues generation.
- Added a dedicated Background Jobs tab with queued/running/completed/failed/interrupted states.
- Added per-job progress, persisted `job.json` metadata, and `request.json` snapshots.
- Results continue writing after every completed model or conversation round.
- Added graceful stop request after the current Ollama generation.
- Remote macOS launcher uses `caffeinate` when available to prevent idle sleep.
- Preserved serial model execution for memory safety.

## v1.7

- Improved responsive mobile layout.
- Model selector becomes full-width on phones.
- Added compact, readable model labels while preserving exact Ollama tags internally.
- Added selected-model preview with full tags.
- Added separate `start-remote.command` for Tailscale/LAN access while keeping `start.command` local-only.


All notable changes to Ollama Model Arena are documented here.

## 1.6.1 - 2026-08-18

### Added
- Windows `start.bat` quick launcher.
- Windows virtual-environment installation and launch instructions.

### Changed
- README now documents macOS, Linux, and Windows separately.
- Updated UI version label to v1.6.1.

## 1.6.0 - 2026-08-17

### Added
- GitHub-ready project documentation and MIT license.
- macOS `start.command` launcher.
- Contribution guide and cleaner repository ignore rules.
- Public-release notes for privacy, Thinking support and benchmark result semantics.

### Changed
- Updated UI version label to v1.6.
- README no longer contains machine-specific paths.
- Repository package excludes Python caches, pytest caches and generated local results.

## 1.5.0 - 2026-08-17

### Changed
- Multi-turn rounds use full multi-line text editors.
- Rounds can be added and removed without editing prompts in a compact table.

## 1.4.0 - 2026-08-17

### Added
- Dedicated single-turn and multi-turn tabs.
- Independent per-model conversation history for multi-turn tests.
- Multi-turn Markdown, JSON and CSV exports.

## 1.3.0 - 2026-08-17

### Added
- Benchmark-oriented `completed`, `truncated`, `empty_final` and `error` states.
- Thinking toggle and Thinking result recording.
- Generation presets and persistent local defaults.
- Token, timing and tokens/sec metadata.
