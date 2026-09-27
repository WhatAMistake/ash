# Ash

A Telegram bot for writers: creative block, ideas, exercises, feedback, and style analysis.

## Features

| Command     | Description                    |
|------------|--------------------------------|
| `/start`   | Welcome + language             |
| `/block`   | Work through creative block    |
| `/develop` | Develop idea or plot           |
| `/character` | Character creation/development |
| `/dialogue` | Dialogue help                  |
| `/prompt`  | Random writing prompt          |
| `/idea`    | Random idea (genre + setting + conflict) |
| `/feedback` | Constructive critique of your text (send excerpt) |
| `/style`   | Analyze style of excerpt       |
| `/reset`   | Reset chat                     |
| `/lang`    | Switch language (ru / en)      |
| `/methodique` | Method-book analysis        |
| `/upload`  | Send a .txt / .docx / .pdf     |
| `/cite`    | Writer's quote                 |
| `/help`    | Command list                   |

...and much more!

## Setup

1. Clone / copy the project.

2. Create `.env` from `.env.example`:
   - `TELEGRAM_BOT_TOKEN` — from [@BotFather](https://t.me/BotFather)
   - `OPENAI_API_KEY` and optionally `OPENAI_API_BASE` (e.g. Together AI)

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the bot:
   ```bash
   cd src
   python telegram_bot.py
   ```
   Options: `--model deepseek-v4-pro` (default), `--no-rag` to disable RAG.

## RAG (books)

The bot can use a RAG over writing-craft books. To enable it:

1. Add `data/book_chunks.json` — a list of chunks, each with:
   - `"text"`: chunk content  
   - `"book_title"`, `"author"`, optional `"chapter"`, `"id"`

2. On first run with RAG enabled, the bot will index chunks into ChromaDB (under `data/chromadb_writers/`).

Suggested books to chunk and add:

- Anne Lamott — *Bird by Bird*
- Ray Bradbury — *Zen in the Art of Writing*
- John Truby — *The Anatomy of Story*
- Robert McKee — *Story*
- Yuri Lotman — *Structure of the Artistic Text* (for Russian)

Without `book_chunks.json`, the bot runs without RAG and still answers using the system prompt and LLM.

## Project layout

```
ash/
├── src/
│   ├── writer_bot.py    # Core logic (block, develop, character, dialogue, prompt, idea, feedback, style)
│   ├── writer_rag.py    # RAG over book chunks (ChromaDB + sentence-transformers)
│   ├── telegram_bot.py  # Telegram interface
│   ├── i18n.py          # ru/en strings
│   └── lang_utils.py    # Language detection
├── prompts/
│   ├── system_prompt.md
│   ├── system_prompt.en.md
│   └── system_prompt.ru.md
├── data/
│   ├── writing_prompts.json  # Prompts, exercises, idea building blocks
│   ├── book_chunks.json     # Optional: chunks for RAG
│   └── user_prefs.json      # Created at runtime (lang preferences)
├── requirements.txt
└── README.md

## Provider compatibility, intent guard and prompt cache

- `LLM_REASONING_POLICY=off` by default (`low|medium|high|passthrough` also
  supported). The adapter selects the Comet / DeepSeek / OpenRouter dialect by
  hostname. Visible answers exclude separate reasoning fields and tagged thinking.
  An unsupported reasoning **parameter** gets one retry without those controls;
  other errors are not swallowed. This fallback cannot guarantee server-side off.
- SDK defaults: `LLM_TIMEOUT_SECONDS=60`, `LLM_MAX_RETRIES=2`. Handler completions
  run in threads, serialized per writer session. Timeout is per SDK attempt,
  not a whole-handler deadline. Optional startup changelog errors do not abort startup.
- Fixed system + command instructions precede the frozen session summary and
  exact wire history. RAG sources live in the final user turn. History is bounded
  by `LLM_HISTORY_MAX_CHARS=24000` and 40 messages, dropping oldest blocks on overflow;
  only rollover updates the writing summary. Secretary sessions do not use it.
- To enable the optional writer intent check, set `INTENT_GUARD_ENABLED=1`.
  `INTENT_GUARD_MODEL` defaults to `deepseek-flash`; optional
  `INTENT_GUARD_API_BASE` / `INTENT_GUARD_API_KEY` override the main configuration.
  A different hostname requires its own key. For OpenRouter, set a provider-qualified
  guard model ID explicitly. Failures/inputs over 16000 characters fail closed with
  a localized retry message. No guard is applied to the secretary or admin/help/reset.
  `/summary` and tool discussions are checked too, including pending documents.
- `SECRETARY_OWNER_NAME` replaces the previously hardcoded account owner name.

### Offline verification and cache diagnostics

```bash
python -m py_compile src/*.py
python -m pytest -q
```

Tests require `pytest` and `pytest-asyncio` in addition to runtime dependencies.
The suite blocks network connections and dotenv loading. `pytest.ini` limits
collection to `tests/`: `scripts/test_changelog.py` is a manual, potentially live
utility, not a unit test. Existing completion tests were migrated from the old
SDK-kwargs contract; new tests exercise provider JSON via MockTransport.

For cache diagnostics, enable **only** the `llm_completion` logger at DEBUG in
an operator-controlled launcher (do not enable SDK HTTP debug logging):

```python
import logging
logger = logging.getLogger("llm_completion")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
# Start the normal application afterwards.
```

Look for `prompt_cache_hit_tokens`, `prompt_cache_miss_tokens`, or `cached_tokens`.
Missing fields are `None`, not zero. Logs contain numeric counters, not prompts.
Compare repeated same-language/command calls within a history epoch. Prefix equality
is tested offline; actual hits, routing and savings require provider usage evidence.
**No live Comet/OpenRouter calls or billing measurements were made for this change.**
