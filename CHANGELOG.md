# Changelog

All notable changes to Ollama Model Arena are documented here.

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
