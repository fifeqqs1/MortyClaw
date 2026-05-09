# Config

`summarize` supports an optional JSON config file for defaults.

## Location

Default path:

- `~/.summarize/config.json`

## Precedence

For `model`:

1. CLI flag `--model`
2. Env `SUMMARIZE_MODEL`
3. Config file `model`
4. Built-in default (`google/gemini-3-flash-preview`)

## Format

`~/.summarize/config.json`:

```json
{
  "model": "google/gemini-3-flash-preview"
}
```
