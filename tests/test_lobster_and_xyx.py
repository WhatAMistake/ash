"""Tests for lobster monologue and xyx (/porko) randomizer."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
        "давайте лампово общаться",
        "наука на твоей стороне",
        "бульк",
        "булькни",
        "а могли бы лампироваться",
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


def test_lobster_isolated_from_ash_system_prompt():
    bot = WriterBot(language="ru", use_rag=False)
    from writer_bot import Message
    bot.history.append(Message(role="user", content="Я хочу написать роман"))
    bot.history.append(Message(role="assistant", content="Отлично, о чём он?"))

    built = bot._build_messages("Напиши монолог лобстера.", extra_system="Инструкция лобстера", command="lobster")
    # System prompt of Ash with jailbreak rules must NOT be in messages
    all_contents = " ".join(m["content"] for m in built)
    assert bot.system_prompt not in all_contents
    assert "Я хочу написать роман" not in all_contents
    assert "Инструкция лобстера" in all_contents


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_lobster_voice_and_format_instructions(lang):
    prompt = t(lang, "instr_lobster")
    love = t(lang, "instr_lobster_love")
    assert "---MESSAGE---" in prompt
    assert "3–5" in prompt
    assert "Ты — лобстер" not in prompt
    assert "You are a lobster" not in prompt
    if lang == "ru":
        assert all(word in prompt for word in (
            "маска", "мандрейк", "меланхолия", "радость", "раздражение",
            "нежность", "сарказм", "не обязательны", "глубокомысленной",
        ))
        assert "Аделина" in love and "350" in love
    else:
        assert all(word in prompt for word in (
            "mask", "mandrake", "melancholy", "joy", "irritation",
            "tenderness", "sarcasm", "not mandatory", "profound",
        ))
        assert "Adeline" in love and "350" in love


@pytest.mark.parametrize(
    "output,expected",
    [
        ("first\n---MESSAGE---\nsecond\n---MESSAGE---\nthird",
         ["first", "second", "third"]),
        ("first\ncontinued\n---MESSAGE---\nsecond\nline\n---MESSAGE---\nthird",
         ["first\ncontinued", "second\nline", "third"]),
        ("---MESSAGE---\nfirst---MESSAGE------MESSAGE---second\n---MESSAGE---",
         ["first", "second"]),
        ("first\n\nsecond\nthird", ["first", "second", "third"]),
        ("one sentence. another sentence", ["one sentence. another sentence"]),
        ("same\n---MESSAGE---\nsame", ["same", "same"]),
    ],
)
def test_lobster_parser_preserves_messages(output, expected):
    bot = WriterBot(language="en", use_rag=False)
    with patch("random.random", return_value=0.07), patch.object(
        bot, "generate_response", return_value=output
    ) as generate:
        assert bot.lobster() == expected
    instruction = generate.call_args.kwargs["temporary_system_instruction"]
    assert "Starting theme:" in instruction
    assert "not mandatory" in instruction
    assert generate.call_args.kwargs["command"] == "lobster"


@pytest.mark.parametrize("lang", ["ru", "en"])
@pytest.mark.parametrize("output", [None, "", "   ", "Error: offline", "  Error: offline", "---MESSAGE---"])
def test_lobster_regular_fallbacks_are_varied_and_formatted(lang, output):
    bot = WriterBot(language=lang, use_rag=False)
    packs = []
    for index in range(3):
        with patch("random.random", return_value=0.5), patch(
            "random.choice", side_effect=lambda choices: choices[index]
        ), patch.object(bot, "generate_response", return_value=output):
            packs.append(bot.lobster())
    assert len({tuple(pack) for pack in packs}) == 3
    for pack in packs:
        assert 3 <= len(pack) <= 5
        assert all(message and message == message.lower() and not message.endswith(".")
                   and "---MESSAGE---" not in message and "\n" not in message for message in pack)
    assert any("мандрейк" in line if lang == "ru" else "mandrake" in line
               for pack in packs for line in pack)


@pytest.mark.parametrize("lang,name", [("ru", "аделина"), ("en", "adeline")])
@pytest.mark.parametrize("output", [None, "", "   ", "Error: offline"])
def test_lobster_love_fallback(lang, name, output):
    bot = WriterBot(language=lang, use_rag=False)
    with patch("random.random", return_value=0.069999), patch.object(
        bot, "generate_response", return_value=output
    ) as generate:
        messages = bot.lobster()
    assert len(messages) == 1 and name in messages[0].lower()
    assert generate.call_args.kwargs["command"] == "lobster"
    assert "350" in generate.call_args.kwargs["temporary_system_instruction"]


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_lobster_love_success_is_one_letter(lang):
    bot = WriterBot(language=lang, use_rag=False)
    with patch("random.random", return_value=0.0), patch.object(
        bot, "generate_response", return_value="  a love letter  "
    ):
        assert bot.lobster() == ["a love letter"]
