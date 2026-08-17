# Contributing

Contributions and bug reports are welcome.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
pytest -q
python app.py
```

## Pull requests

Please keep changes focused and preserve these behaviors unless the change explicitly targets them:

- models are compared serially by default
- multi-turn histories are isolated per model
- failed/empty generations are not silently retried
- truncated output remains visible and is explicitly marked
- generated exports and private prompts/results are not committed

Do not include personal filesystem paths, private prompts, generated model outputs, `.venv`, or local caches in pull requests.
