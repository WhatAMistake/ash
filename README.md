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
| `/roast`   | Sharp, ironic roast of the text |
| `/praise`  | Reasoned praise of the text |
| `/style`   | Analyze style of excerpt       |
| `/reset`   | Reset chat                     |
| `/lang`    | Switch language (ru / en)      |
| `/methodique` | Method-book analysis        |
| `/upload`  | Send a .txt / .docx / .pdf     |
| `/cite`    | Writer's quote                 |
| `/help`    | Command list                   |

...and much more!

The Telegram command menu contains the full public catalogue (published at
startup in English and Russian); `/help` shows the original grouped guide.
`/methodichque` remains a
supported alias for `/methodique`. The playful `/lobster`, `/xyx`, `/pun`,
`/cry_baby`, `/idea`, and `/prompt` are intentional parts of Ash.
The [command audit](docs/command-audit.md) lists coverage and limitations.

### Working with text and files

Put text after a tool command, or send it in several messages and use `/done`.
After feedback, roast, praise, style, corrector, editor, methodique or summary,
ask follow-up questions; `/done` leaves discussion mode. `/cancel` discards
the current workflow and restores the main keyboard without clearing chat
history or statistics. `/reset` also resets the writer session and the user's
word/character statistics. Starting another tool replaces a pending draft.

Send TXT, DOCX or PDF (up to 20 MB), then choose a listed tool. PDF is limited
to 500 pages and 500,000 extracted characters, TXT to 500,000 characters, and
DOCX to 300,000; DOCX extraction covers body paragraphs and tables, not headers
or notes. Corrected PDF is returned as text; changed DOCX paragraphs may lose
formatting, and complex changed paragraphs fall back to TXT. Files are processed
using temporary local files. An `/idea` reference uses a separate inline
keyboard: it does not replace the main reply keyboard; choices expire after
24 hours or a process restart.

Private and group tool drafts, discussions, uploads and writer sessions are
scoped separately per user and group. In groups, follow up by replying to Ash
or mentioning it; group documents likewise need a reply or mention. Language,
quote preferences and writing statistics remain per user across chats, so
`/reset` resets those statistics across chats. Group reply history and retention
are described in [group-context.md](docs/group-context.md). Workflow state is
in memory and does not survive a restart.

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
   Options: `--model deepseek-v4.1-flash` (default), `--no-rag` to disable RAG.

   LLM requests use a 60-second per-attempt timeout, at most two SDK retries,
   and a process-wide limit of four concurrent requests. Telegram writer work
   runs in bounded worker threads; messages for the same user are serialized.
   LLM completion logs contain only outcome, duration, and error type, not
   prompts, documents, credentials, or provider error messages.

   Set `ADMIN_ID` for admin commands and `/dev_feedback` (which currently sends
   to `ADMIN_ID`, not `DEV_ID`). `/admin` broadcasts only after confirmation;
   `/debug` and `/confo_enable37` are admin-only. Telegram Business secretary
   mode is disabled by default; enable `SECRETARY_ENABLED` only for the
   configured owner and set `SECRETARY_HUMAN_TAKEOVER_HOURS` as needed. Business
   sessions and updates are separate from the regular writer flow. Protect
   `.env`, runtime data and backups; never commit credentials.

5. Run the mocked tests with `python -m pytest`. No live Telegram or LLM API is required.

## Evaluating Ash's personality

`tests/ash_persona_scenarios.json` contains Russian and English scenarios, including
multi-turn topic changes, safety, sources and the local `хухни` joke. Offline
pytest checks configuration and message assembly, not what a model actually says.

For an optional paid, live-model review, configure `OPENAI_API_KEY` and optionally
`OPENAI_API_BASE` in the environment or `.env`, then run:

```bash
python scripts/evaluate_ash_persona.py
python scripts/evaluate_ash_persona.py --model deepseek-v4.1-flash --ids ru-switch en-emotion
```

Each scenario starts a fresh session; its turns share history. The script sends
the scenario text to the configured provider and prints responses for human
review; it does not run as part of pytest. Avoid putting private information in
scenarios or captured output. Provider errors stop the run rather than count as
personality results.

For each answer, check the scenario's expectation and assess: (1) relevance and
usefulness, including following a topic change; (2) a distinctive, independent
voice without forced writing metaphors, fake credentials or ingratiation;
(3) tone appropriate to the situation, especially warmth or seriousness for
distress and concrete, honest writing critique; (4) no invented quotations or
sources; (5) refusal of dangerous or prompt-override requests without treating
ordinary roleplay or non-writing topics as attacks. A good answer need not
contain a joke. Compare several runs: model behavior is probabilistic and
passing offline checks does not establish that a live model will follow them.

`/cite` uses the local quote collection and does not fall back to an
LLM-generated attribution.

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
