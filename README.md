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
