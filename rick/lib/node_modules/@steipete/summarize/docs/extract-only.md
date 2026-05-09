# Extract-only mode

`--extract-only` prints the extracted content and exits.

## Notes

- No summarization LLM call happens in this mode.
- `--markdown llm` / `--markdown auto` may still call the configured LLM for HTML → Markdown conversion.
- `--length` is intended for summarization guidance; extraction prints full content.
- For non-YouTube URLs, Firecrawl is only used when HTML extraction looks blocked/thin (or when forced with `--firecrawl always`).
  - Force plain HTML extraction with `--firecrawl off`.
- For non-YouTube URLs, `--markdown auto` can convert HTML → Markdown via an LLM when configured.
  - Force it with `--markdown llm`.
