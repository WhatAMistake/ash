"""Tests for lobster monologue and xyx (/porko) randomizer."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from i18n import t
from writer_bot import WriterBot


def test_xyx_porko_randomizer():
    bot = WriterBot(language="ru", use_rag=False)
    allowed = {
        "хух",
        "хухни",
        "давайте правильно общаться",
        "наука на твоей стороне",
        "бульк",
        "булькни",
    }
    seen = set()
    for _ in range(200):
        val = bot.porko()
        assert val in allowed
        seen.add(val)
    # With 200 draws from 6 items, probability of not seeing all is astronomically small
    assert len(seen) >= 4
    assert "давайте правильно общаться" in seen or "наука на твоей стороне" in seen
    assert "бульк" in seen or "булькни" in seen


def test_system_prompt_contains_xyx_rule():
    prompts_dir = project_root / "prompts"
    prompt_ru = (prompts_dir / "system_prompt.ru.md").read_text(encoding="utf-8")
    prompt_en = (prompts_dir / "system_prompt.en.md").read_text(encoding="utf-8")
    prompt_base = (prompts_dir / "system_prompt.md").read_text(encoding="utf-8")

    assert "хухни" in prompt_ru
    assert "хух" in prompt_ru
    assert "хухни" in prompt_en
    assert "хух" in prompt_en
    assert "хухни" in prompt_base
    assert "хух" in prompt_base


def test_lobster_adelina_prompt_metaphor_and_romance():
    ru_love = t("ru", "instr_lobster_love")
    assert "Аделина" in ru_love
    assert "метафора" in ru_love
    assert "романтическ" in ru_love or "романтич" in ru_love

    en_love = t("en", "instr_lobster_love")
    assert "Adeline" in en_love
    assert "metaphor" in en_love
    assert "romantic" in en_love


def test_lobster_instructions_and_themes():
    ru_lobster_instr = t("ru", "instr_lobster")
    assert "кинцуги" in ru_lobster_instr
    assert "икигай" in ru_lobster_instr
    assert "ваби-саби" in ru_lobster_instr
    assert "хоннэ и татэмаэ" in ru_lobster_instr
    assert "мандрейк" in ru_lobster_instr

    bot_ru = WriterBot(language="ru", use_rag=False)
    with patch("random.random", return_value=0.99), patch.object(
        bot_ru, "generate_response", return_value="Error: llm offline"
    ):
        messages = bot_ru.lobster()
        assert isinstance(messages, list)
        assert len(messages) >= 1
        assert not any("---MESSAGE---" in m for m in messages)


def test_lobster_adelina_roll_fallback():
    bot_ru = WriterBot(language="ru", use_rag=False)
    with patch("random.random", return_value=0.01), patch.object(
        bot_ru, "generate_response", return_value="Error: network fail"
    ):
        messages = bot_ru.lobster()
        assert len(messages) == 1
        assert "аделина" in messages[0].lower()
