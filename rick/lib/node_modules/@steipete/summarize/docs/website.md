# Website mode

Use this for non-YouTube URLs.

## What it does

- Fetches the page HTML.
- Extracts “article-ish” content and normalizes it into clean text.
- If extraction looks blocked or too thin, it can retry via Firecrawl (Markdown).
- In `--extract-only` mode, Firecrawl is only used when HTML extraction looks blocked/thin (or when forced via `--firecrawl always`).
- In `--extract-only` mode, `--markdown auto|llm` can convert HTML → Markdown via an LLM using the configured `--model` (no provider fallback).

## Flags

- `--firecrawl off|auto|always`
- `--markdown off|auto|llm` (default: `auto`; only affects `--extract-only` for non-YouTube URLs)
- Plain-text mode: use `--firecrawl off --markdown off`.
- `--timeout 30s|30|2m|5000ms` (default: `2m`)
- `--extract-only` (print extracted content; no summary LLM call)
- `--json` (emit a single JSON object)
- `--verbose` (progress + which extractor was used)
- `--metrics off|on|detailed` (default: `on`; `detailed` prints a breakdown to stderr)

## API keys

- Optional: `FIRECRAWL_API_KEY` (for the Firecrawl fallback)
- Optional: `XAI_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` (also accepts `GOOGLE_GENERATIVE_AI_API_KEY` / `GOOGLE_API_KEY`) (required only when `--markdown llm` is used, or when `--markdown auto` falls back to LLM conversion)
