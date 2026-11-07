"""Opt-in live persona evaluation. Outputs conversations for human review."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from writer_bot import WriterBot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", help="Run only these scenario IDs")
    parser.add_argument("--model", default="deepseek-v4.1-flash", help="Chat model")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY (and optionally OPENAI_API_BASE) first")
    scenarios = json.loads((ROOT / "tests" / "ash_persona_scenarios.json").read_text(encoding="utf-8"))
    known = {scenario["id"] for scenario in scenarios}
    if args.ids and set(args.ids) - known:
        parser.error("Unknown scenario IDs: " + ", ".join(sorted(set(args.ids) - known)))
    for scenario in scenarios:
        if args.ids and scenario["id"] not in args.ids:
            continue
        bot = WriterBot(language=scenario["language"], model=args.model, use_rag=False)
        bot.chat_model = args.model
        if bot.client is None:
            raise SystemExit("Could not initialize the LLM client")
        print(f"\n### {scenario['id']} — {scenario['expect']}")
        for turn in scenario["turns"]:
            print(f"USER: {turn}")
            answer = bot.chat(turn)
            print(f"ASH: {answer}")
            if answer.startswith("Error:"):
                raise SystemExit("Model request failed; evaluation is incomplete.")


if __name__ == "__main__":
    main()
