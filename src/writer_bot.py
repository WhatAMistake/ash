"""
Ash — bot for writers: creative block, ideas, exercises, feedback.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass

from dotenv import load_dotenv
from i18n import t
from lang_utils import detect_language
from llm_completion import create_chat_completion, visible_completion_text, log_completion_error

load_dotenv()


@dataclass
class Message:
    role: str
    content: str


class WriterBot:
    """Ash: writing coach with RAG over craft books."""

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        use_rag: bool = True,
        data_dir: Optional[str] = None,
        language: str = "en",
    ):
        self.model = model
        self.chat_model = "deepseek-v4-pro"  # Secondary model for simple chat/chitchat
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.language = language
        self.use_rag = use_rag
        self.system_prompt = self._load_system_prompt()
        self.history: list[Message] = []
        self.session_summary: Optional[str] = None
        self.messages_since_summary: int = 0
        self.summary_trigger_count: int = 20

        self.rag = None
        if use_rag:
            self._init_rag(data_dir)

        self.client = None
        self._init_llm()

        self._prompts_data: Optional[dict] = None

    def _load_system_prompt(self) -> str:
        prompts_dir = Path(__file__).parent.parent / "prompts"
        lang_file = prompts_dir / f"system_prompt.{self.language}.md"
        if lang_file.exists():
            return lang_file.read_text(encoding="utf-8")
        generic = prompts_dir / "system_prompt.md"
        if generic.exists():
            return generic.read_text(encoding="utf-8")
        return self._default_prompt()

    def _default_prompt(self) -> str:
        return """You are Ash — a writer with twenty years of experience. You help with creative block, plot and character development, dialogue, and style. You draw on craft advice from authors like Stephen King, Anne Lamott, Ray Bradbury, John Truby, Robert McKee. Be concise, concrete, and direct. No fluff, no empty motivation."""

    # Context window limit: 128k tokens ≈ 512k characters (safe limit ~400k)
    MAX_CONTEXT_CHARS = 420000

    def _get_detail_level(self, text_length: int) -> str:
        """Determine detail level based on text length."""
        if text_length < 2000:
            return "brief"
        elif text_length < 7000:
            return "standard"
        elif text_length < 30000:
            return "detailed"
        else:
            return "full"

    def _truncate_for_context(self, text: str, max_chars: Optional[int] = None) -> tuple[str, bool]:
        """Truncate text if it exceeds context window limit.
        
        Returns: (truncated_text, was_truncated)
        """
        max_chars = max_chars or self.MAX_CONTEXT_CHARS
        if len(text) <= max_chars:
            return text, False
        
        # Truncate with notification
        truncated = text[:max_chars]
        return truncated, True

    def _init_rag(self, data_dir: Optional[str]):
        try:
            from writer_rag import WriterRAG
            self.rag = WriterRAG(data_dir=data_dir)
            print("Writer RAG initialized")
        except Exception as e:
            print(f"Writer RAG unavailable: {e}")
            self.rag = None

    def _init_llm(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            print(f"LLM client initialized: {self.model}")
        except ImportError:
            print("openai not installed. pip install openai")
            self.client = None

    def _clear_prompts_cache(self) -> None:
        """Clear the prompts data cache to force reload on next access."""
        self._prompts_data = None
        print(f"[PROMPTS] Cache cleared for language '{self.language}'")

    def _load_prompts_data(self) -> dict:
        if self._prompts_data is not None:
            return self._prompts_data
        
        # Try language-specific file first
        lang_suffix = f".{self.language}" if self.language != "en" else ""
        data_path = Path(__file__).parent.parent / "data" / f"writing_prompts{lang_suffix}.json"
        
        print(f"[PROMPTS] Trying to load: {data_path}")
        
        # Fall back to default file if language-specific doesn't exist
        if not data_path.exists():
            print(f"[PROMPTS] Language-specific file not found, falling back to default")
            data_path = Path(__file__).parent.parent / "data" / "writing_prompts.json"
            print(f"[PROMPTS] Trying default: {data_path}")
        
        try:
            if data_path.exists():
                print(f"[PROMPTS] Loading from: {data_path}")
                with open(data_path, "r", encoding="utf-8") as f:
                    self._prompts_data = json.load(f)
                print(f"[PROMPTS] Loaded successfully. Keys: {list(self._prompts_data.keys())}")
                # Check if idea keys exist
                for key in ["idea_genre", "idea_setting", "idea_conflict"]:
                    if key in self._prompts_data:
                        print(f"[PROMPTS] {key}: {len(self._prompts_data[key])} items")
                    else:
                        print(f"[PROMPTS] WARNING: {key} not found in prompts data!")
            else:
                print(f"[PROMPTS] File not found: {data_path}")
                print(f"[PROMPTS] Falling back to default prompts data")
                self._prompts_data = self._default_prompts_data()
        except json.JSONDecodeError as e:
            print(f"[PROMPTS] JSON parse error in {data_path}: {e}")
            print(f"[PROMPTS] Falling back to default prompts data")
            self._prompts_data = self._default_prompts_data()
        except Exception as e:
            print(f"[PROMPTS] Error loading {data_path}: {e}")
            print(f"[PROMPTS] Falling back to default prompts data")
            self._prompts_data = self._default_prompts_data()
        
        return self._prompts_data

    def _default_prompts_data(self) -> dict:
        if self.language == "ru":
            return {
                "prompts": [
                    "Напиши абзац без буквы 'о'.",
                    "Опиши комнату глазами человека, который только что кого-то убил.",
                    "Первое предложение: 'Дверь была открыта, хотя я точно помнил...'",
                    "Персонаж находит письмо, которое всё меняет.",
                    "Двое встречаются в лифте. Один лжёт. Напиши диалог.",
                    "Опиши один и тот же объект с точки зрения трёх разных людей.",
                    "Напиши сцену, где ничего не происходит, но всё меняется.",
                    "Персонаж просыпается с чужим воспоминанием.",
                    "Напиши диалог, где оба говорят о разном, но думают, что понимают друг друга.",
                    "Опиши запах, который вызывает травму.",
                    "Первое предложение: 'Она не знала, что это последний раз...'",
                    "Персонаж делает доброе дело с эгоистичными мотивами.",
                    "Напиши сцену только через звуки.",
                    "Два персонажа, которые ненавидят друг друга, вынуждены работать вместе.",
                    "Персонаж находит вещь, которую искал десять лет. И сжигает её.",
                ],
                "exercises": [
                    {"title": "Поток сознания", "duration_min": 5, "description": "Пиши без остановки 5 минут. Без редактуры. Пусть слова текут."},
                    {"title": "Диалог в лифте", "duration_min": 10, "description": "Два персонажа в лифте. Один знает секрет, который нельзя выдавать."},
                    {"title": "Дождь тремя способами", "duration_min": 15, "description": "Опиши звук дождя тремя способами: буквально, метафорически и через настроение персонажа."},
                    {"title": "Смерть предмета", "duration_min": 10, "description": "Опиши, как умирает какой-то обычный предмет (часы, растение, машина). Без патоса."},
                    {"title": "Обратный хронометраж", "duration_min": 15, "description": "Напиши сцену, где время идёт назад — от конца к началу."},
                    {"title": "Чужое окно", "duration_min": 10, "description": "Опиши, что видит персонаж, глядя в чужое окно. Только визуальные детали."},
                ],
                "idea_genre": [
                    "литература", "киберпанк", "фэнтези", "нуар", "романтика", 
                    "триллер", "исторический", "sci-fi", "хоррор", "вестерн",
                    "постапокалипсис", "магический реализм", "детектив", "боевик",
                    "психологическая драма", "сатира", "утопия", "антиутопия",
                    "путешествие во времени", "космоопера", "тёмное фэнтези",
                    "криминал", "семейная сага", "психологический триллер",
                    "ромфуд", "body horror", "метапроза", "афрофутуризм",
                    "соларпанк", "биопанк", "вестерн", "гаслампа",
                    "нью-вёрд", "документальная проза", "литРПГ",
                    "уютное фэнтези", "гримдарк", "прогресс-фэнтези",
                    "культовая фикция", "анти-романтика", "киндзайти",
                    "вуся", "сянься", "тики-нуар"
                ],
                "idea_setting": [
                    "монастырь", "космическая станция", "провинциальный городок", 
                    "отель", "корабль", "библиотека", "больница", "школа",
                    "заброшенный завод", "метро после закрытия", "пустыня",
                    "подводная станция", "арктическая станция", "высотка в тумане",
                    "портовый город", "граница двух миров", "виртуальная реальность",
                    "последний город на Земле", "остров без связи", "подземный бункер",
                    "поезд через всю страну", "аэропорт в снегопад", "лес, где теряется время",
                    "парящий город", "подземное метро", "разрушенный торговый центр",
                    "ночной цирк", "такси по городу", "брошенный космический корабль",
                    "рыбный рынок на рассвете", "пентхаус", "охотничий домик",
                    "тоннели под городом", "дом у замёрзшего озера", "вечеринка на крыше",
                    "заправка в пустыне", "магазин виниловых пластинок", "букинист",
                    "паром", "психиатрическая клиника", "заброшенный санаторий",
                    "стройка ночью"
                ],
                "idea_conflict": [
                    "главный герой не может лгать", "выбор между двумя людьми",
                    "тайна, которая разрушит семью", "гонка со временем", 
                    "предательство друга", "прошлое настигает",
                    "наследство, которое нельзя принять", "долг vs желание",
                    "свидетель преступления молчит", "возвращение в город, который ненавидит",
                    "персонаж узнаёт, что он — клон", "невозможное обещание",
                    "защита врага ради общего блага", "секрет, который нельзя рассказать даже близким",
                    "последний представитель рода", "двойник из параллельной жизни",
                    "память, которую нужно стереть", "встреча с молодым собой",
                    "наследственное проклятие", "технология, которая знает о тебе всё",
                    "потерянный ребёнок ищет родителей", "нужно уничтожить единственную вещь, связывающую с прошлым",
                    "незнакомец знает имя, но его не помнят", "каждый сон — пророчество, которое сбывается",
                    "слышит мысли окружающих", "должен солгать, чтобы спасти чью-то жизнь",
                    "забытый долг возвращается", "технология, которую создал персонаж, используется во зло",
                    "все, кого он любит, умирают", "только он помнит умершего",
                    "двойная жизнь раскрыта", "прошлое, которое пытались стереть, всплывает",
                    "нельзя доверять собственным воспоминаниям", "выбор между правдой и справедливостью",
                    "голос преследующего — его собственное будущее", "носит чужую вину",
                    "знает, кто убийца, но нет доказательств", "дар, который убивает любого, кого он любит",
                    "застрял в чужой жизни", "запрещено говорить правду"
                ],
            }
        return {
            "prompts": [
                "Write a paragraph without the letter 'e'.",
                "Describe a room through the eyes of someone who just killed someone.",
                "First sentence: 'The door was open, although I clearly remembered...'",
                "A character finds a letter that changes everything.",
                "Two people meet in a lift. One is lying. Write the dialogue.",
                "Describe the same object from three different people's perspectives.",
                "Write a scene where nothing happens, but everything changes.",
                "A character wakes up with someone else's memory.",
                "Write a dialogue where both talk about different things but think they understand each other.",
                "Describe a smell that triggers trauma.",
                "First sentence: 'She didn't know this was the last time...'",
                "A character does a good deed with selfish motives.",
                "Write a scene using only sounds.",
                "Two characters who hate each other forced to work together.",
                "A character finds an item they searched for ten years. And burns it.",
            ],
            "exercises": [
                {"title": "Stream of consciousness", "duration_min": 5, "description": "Write non-stop for 5 minutes. No editing. Let the words flow."},
                {"title": "Dialogue in a lift", "duration_min": 10, "description": "Two characters in an elevator. One knows a secret the other must not discover."},
                {"title": "Rain three ways", "duration_min": 15, "description": "Describe the sound of rain in three different ways: literal, metaphorical, and through a character's mood."},
                {"title": "Death of an object", "duration_min": 10, "description": "Describe an ordinary object dying (clock, plant, car). No pathos."},
                {"title": "Reverse chronology", "duration_min": 15, "description": "Write a scene where time runs backwards — from end to beginning."},
                {"title": "Someone else's window", "duration_min": 10, "description": "Describe what a character sees looking into someone else's window. Visual details only."},
            ],
            "idea_genre": [
                "literary", "cyberpunk", "fantasy", "noir", "romance", 
                "thriller", "historical", "sci-fi", "horror", "western",
                "post-apocalyptic", "magical realism", "detective", "action",
                "psychological drama", "satire", "utopia", "dystopia",
                "time travel", "space opera", "dark fantasy",
                "crime", "family saga", "psychological thriller",
                "romantic food fiction", "body horror", "metafiction", "afrofuturism",
                "solarpunk", "biopunk", "weird west", "gaslamp fantasy",
                "new weird", "documentary fiction", "litRPG",
                "cozy fantasy", "grimdark", "progression fantasy",
                "cult fiction", "anti-romance", "kindaichi mystery",
                "wuxia", "xianxia", "tiki noir"
            ],
            "idea_setting": [
                "monastery", "space station", "small town", 
                "hotel", "ship", "library", "hospital", "school",
                "abandoned factory", "subway after hours", "desert",
                "underwater station", "arctic station", "skyscraper in fog",
                "port city", "border between two worlds", "virtual reality",
                "last city on Earth", "island with no connection", "underground bunker",
                "train across the country", "airport in snowstorm", "forest where time is lost",
                "floating city", "underground metro", "ruined mall",
                "night circus", "taxi driving through the city", "abandoned spaceship",
                "fish market at dawn", "penthouse apartment", "hunting cabin",
                "tunnel system under the city", "frozen lake house", "crowded rooftop party",
                "desert gas station", "vintage record shop", "antique bookstore",
                "ferry boat", "psychiatric ward", "abandoned sanatorium",
                "construction site at night"
            ],
            "idea_conflict": [
                "protagonist cannot lie", "must choose between two people",
                "secret that would destroy the family", "race against time", 
                "betrayal of a friend", "past catches up",
                "inheritance that cannot be accepted", "duty vs desire",
                "witness to crime stays silent", "return to a hated hometown",
                "character learns they are a clone", "impossible promise",
                "protecting an enemy for common good", "secret that cannot be told even to loved ones",
                "last of the bloodline", "doppelganger from parallel life",
                "memory that must be erased", "meeting younger self",
                "hereditary curse", "technology that knows everything about you",
                "lost child searching for parents", "must destroy the only thing connecting them to the past",
                "stranger knows their name but they don't remember them", "every dream is a prophecy that comes true",
                "can hear other people's thoughts", "must lie to save someone's life",
                "forgotten debt returns", "technology they created is used for evil",
                "everyone they love dies", "only one who remembers the dead person",
                "double life exposed", "past they tried to erase resurfaces",
                "cannot trust their own memories", "must choose between truth and justice",
                "haunting voice belongs to their future self", "carrying someone else's guilt",
                "knows who the killer is but has no proof", "gift that kills anyone they love",
                "stuck in someone else's life", "forbidden to speak the truth"
            ],
        }

    def _build_messages(self, user_input: str, extra_system: Optional[str] = None, command: Optional[str] = None) -> list[dict]:
        # Keep system_prompt as separate message to ensure it's not ignored
        messages = [{"role": "system", "content": self.system_prompt}]
        if extra_system:
            messages.append({"role": "system", "content": extra_system})
        
        if self.rag and self.use_rag:
            context = self.rag.get_context_for_query(user_input, max_chunks=3, command=command)
            if context:
                messages.append({
                    "role": "system",
                    "content": t(self.language, "rag_context", context=context)
                })
        if self.session_summary:
            messages.append({
                "role": "system",
                "content": f"Brief summary of this session:\n{self.session_summary}"
            })
        
        # Skip history for summary command to avoid role confusion
        if command != "summary":
            for msg in self.history[-10:]:
                messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": user_input})
        return messages

    def generate_response(
        self,
        user_input: str,
        temporary_system_instruction: Optional[str] = None,
        command: Optional[str] = None,
    ) -> str:
        if not self.client:
            return "Error: LLM client not initialized."
        # Add clean text instruction to all responses
        clean_text_instr = "\n\nВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов, без эмодзи, без markdown-форматирования." if self.language == "ru" else "\n\nIMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements, no markdown formatting."
        if temporary_system_instruction:
            temporary_system_instruction = temporary_system_instruction + clean_text_instr
        else:
            temporary_system_instruction = clean_text_instr.strip()
        messages = self._build_messages(user_input, extra_system=temporary_system_instruction, command=command)
        try:
            r = create_chat_completion(
                self.client,
                reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            return visible_completion_text(r)
        except Exception as e:
            log_completion_error(e)
            return "Error: " + t(self.language, "error_llm")

    def chat(self, user_input: str) -> str:
        if not self.client:
            return "Error: LLM client not initialized."
        # Add clean text instruction
        clean_text_instr = "ВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов, без эмодзи, без markdown-форматирования." if self.language == "ru" else "IMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements, no markdown formatting."
        messages = self._build_messages(user_input, extra_system=clean_text_instr)
        try:
            r = create_chat_completion(
                self.client,
                reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                model=self.chat_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            reply = visible_completion_text(r)
            self.history.append(Message(role="user", content=user_input))
            self.history.append(Message(role="assistant", content=reply))
            self.messages_since_summary += 2
            if self.messages_since_summary >= self.summary_trigger_count:
                self._maybe_summarize()
            return reply
        except Exception as e:
            log_completion_error(e)
            return "Error: " + t(self.language, "error_llm")

    def _maybe_summarize(self):
        if not self.client or len(self.history) < 10:
            return
        history_text = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in self.history[:-10]
        )
        clean_suffix = " Без эмодзи, без звёздочек, без декоративных элементов." if self.language == "ru" else " No emojis, no asterisks, no decorative elements."
        prompt = f"Summarize this writing-coach session in under 150 words (themes, goals, progress). Use {self.language}.\n\n{history_text}"
        try:
            r = create_chat_completion(
                self.client,
                reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                model=self.model,
                messages=[
                    {"role": "system", "content": "You summarize writing sessions briefly." + clean_suffix},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            self.session_summary = visible_completion_text(r) or None
            self.messages_since_summary = 0
        except Exception as e:
            log_completion_error(e)

    def handle_block(self, user_message: str) -> str:
        # Define all 12 tips and randomly select 2 to ensure variety
        all_tips_ru = [
            "Напиши концовку сцены первой — потом вернись к началу",
            "Смени шрифт на Comic Sans или моноширинный — сломай привычный визуал",
            "Пиши от второго лица ('ты') или множественного числа ('мы')",
            "Опиши сцену только через запахи — без зрения",
            "Убей персонажа неожиданно — разберись с последствиями позже",
            "Напиши сцену как письмо врагу или бывшему",
            "Начни с погоды, но сделай её зловещей (не 'солнечно', а 'солнце как приговор')",
            "Только диалоги — без тегов, без описаний, голый разговор",
            "Инвертируй обстановку: пустыня → арктика, квартира → лифт",
            "Напиши сцену как полицейский протокол",
            "Начни с ломающегося предмета — звук разрушения",
            "Смени время: если писал в прошлом — перепиши в настоящем"
        ]
        all_tips_en = [
            "Write the scene's ending first — then go back to the beginning",
            "Change font to Comic Sans or monospace — break the familiar visual",
            "Write in second person ('you') or plural ('we')",
            "Describe the scene through smells only — no vision",
            "Kill a character unexpectedly — deal with consequences later",
            "Write the scene as a letter to an enemy or ex",
            "Start with weather but make it ominous (not 'sunny', but 'sun like a verdict')",
            "Dialogue only — no tags, no descriptions, bare conversation",
            "Invert the setting: desert → arctic, apartment → elevator",
            "Write the scene as a police report",
            "Start with a breaking object — sound of destruction",
            "Change tense: if you wrote in past, rewrite in present"
        ]
        
        tips = all_tips_ru if self.language == "ru" else all_tips_en
        selected = random.sample(tips, 2)
        
        base_instr = t(
            self.language,
            "instr_block",
            default="Writer is blocked. Give 1-2 concrete actions. You MUST cite at least one source from the provided context (e.g., 'As King says...' or 'По Ламотт...'). If no relevant advice in context, say so explicitly."
        )
        
        # Inject randomly selected tips into the instruction
        if self.language == "ru":
            specific_instr = f"{base_instr}\n\nИспользуй ТОЛЬКО эти два совета (или один из них):\n1) {selected[0]}\n2) {selected[1]}\n\nНе используй другие советы. Выбери один из этих двух и дай конкретное действие."
        else:
            specific_instr = f"{base_instr}\n\nUse ONLY these two tips (or one of them):\n1) {selected[0]}\n2) {selected[1]}\n\nDo not use other tips. Pick one of these two and give a concrete action."
        
        return self.generate_response(user_message, temporary_system_instruction=specific_instr, command="block")

    def develop_idea(self, user_message: str) -> str:
        instr = t(
            self.language,
            "instr_develop",
            default="Develop the idea. Structure, stakes, twist. You MUST reference at least one specific technique from the provided books with author name (e.g., 'Truby's 22 steps...' or 'По Макки, инцидент...'). Cite sources explicitly."
        )
        return self.generate_response(user_message, temporary_system_instruction=instr, command="develop")

    def character_help(self, user_message: str) -> str:
        instr = t(
            self.language,
            "instr_character",
            default="Character. Want vs need, flaw, voice. You MUST cite at least one specific method from the provided context with author name (e.g., 'As McKee defines need...' or 'По Труби, недостаток...'). Always name your source."
        )
        return self.generate_response(user_message, temporary_system_instruction=instr, command="character")

    def dialogue_help(self, user_message: str) -> str:
        instr = t(
            self.language,
            "instr_dialogue",
            default="Dialogue. Subtext, conflict, voice differentiation. You MUST reference specific advice from the provided books with author name (e.g., 'McKee on subtext...' or 'По Есенгазиеву...'). Cite your source every time."
        )
        return self.generate_response(user_message, temporary_system_instruction=instr, command="dialogue")

    def _local_random_prompt(self) -> str:
        data = self._load_prompts_data()
        prompts = data.get("prompts", [])
        if not prompts:
            return self._default_prompts_data()["prompts"][0]
        return random.choice(prompts)

    def get_random_prompt(self) -> str:
        # 80% chance to generate fresh, 20% to use pre-made
        if random.random() < 0.8 and self.client:
            return self.generate_fresh_prompt()
        return self._local_random_prompt()

    def get_random_exercise(self) -> dict:
        data = self._load_prompts_data()
        exercises = data.get("exercises", [])
        if not exercises:
            return self._default_prompts_data()["exercises"][0]
        return random.choice(exercises)

    def generate_idea(self) -> str:
        # Load prompts data from file (language-specific if available)
        data = self._load_prompts_data()
        genre = random.choice(data.get("idea_genre", ["literary"]))
        setting = random.choice(data.get("idea_setting", ["small town"]))
        conflict = random.choice(data.get("idea_conflict", ["secret that would destroy the family"]))
        return t(
            self.language,
            "idea_format",
            genre=genre,
            setting=setting,
            conflict=conflict,
            default=f"{genre} + {setting} + {conflict}",
        )

    def feedback_on_text(self, text: str) -> str:
        # Check for context limit and truncate if needed
        text, was_truncated = self._truncate_for_context(text)
        truncation_msg = f"\n\n[ВНИМАНИЕ: Текст обрезан до {self.MAX_CONTEXT_CHARS} символов из-за ограничения контекста.]" if self.language == "ru" else f"\n\n[NOTE: Text truncated to {self.MAX_CONTEXT_CHARS} chars due to context limit.]"
        
        level = self._get_detail_level(len(text))
        instr_key = f"instr_feedback_{level}"
        instr = t(
            self.language,
            instr_key,
            default=t(
                self.language,
                "instr_feedback",
                default="Text critique. 1) What works. 2) What breaks. 3) How to fix. You MUST cite at least one specific principle from the provided books with author name."
            )
        )
        result = self.generate_response(text, temporary_system_instruction=instr, command="feedback")
        return result + truncation_msg if was_truncated else result

    def analyze_style(self, text: str) -> str:
        # Check for context limit and truncate if needed
        text, was_truncated = self._truncate_for_context(text)
        truncation_msg = f"\n\n[ВНИМАНИЕ: Текст обрезан до {self.MAX_CONTEXT_CHARS} символов из-за ограничения контекста.]" if self.language == "ru" else f"\n\n[NOTE: Text truncated to {self.MAX_CONTEXT_CHARS} chars due to context limit.]"
        
        level = self._get_detail_level(len(text))
        instr_key = f"instr_style_{level}"
        instr = t(
            self.language,
            instr_key,
            default=t(
                self.language,
                "instr_style",
                default="Style check: sentence length (monotony?), POV shifts, verbs (passive vs active), word repetition. Name one weak spot and how to fix it."
            )
        )
        result = self.generate_response(text, temporary_system_instruction=instr, command="style")
        return result + truncation_msg if was_truncated else result

    def roast(self, text: str) -> str:
        """Harsh, cynical, but objective critique."""
        # Check for context limit and truncate if needed
        text, was_truncated = self._truncate_for_context(text)
        truncation_msg = f"\n\n[ВНИМАНИЕ: Текст обрезан до {self.MAX_CONTEXT_CHARS} символов из-за ограничения контекста.]" if self.language == "ru" else f"\n\n[NOTE: Text truncated to {self.MAX_CONTEXT_CHARS} chars due to context limit.]"
        
        level = self._get_detail_level(len(text))
        instr_key = f"instr_roast_{level}"
        instr = t(
            self.language,
            instr_key,
            default=t(
                self.language,
                "instr_roast",
                default="Harsh, cynical, but objective critique. No mercy. Point out specific problems: what breaks, why it's bad, how to fix."
            )
        )
        result = self.generate_response(text, temporary_system_instruction=instr, command="roast")
        return result + truncation_msg if was_truncated else result

    def praise(self, text: str) -> str:
        """Specific, precise praise for what works in the text."""
        # Check for context limit and truncate if needed
        text, was_truncated = self._truncate_for_context(text)
        truncation_msg = f"\n\n[ВНИМАНИЕ: Текст обрезан до {self.MAX_CONTEXT_CHARS} символов из-за ограничения контекста.]" if self.language == "ru" else f"\n\n[NOTE: Text truncated to {self.MAX_CONTEXT_CHARS} chars due to context limit.]"
        
        level = self._get_detail_level(len(text))
        instr_key = f"instr_praise_{level}"
        instr = t(
            self.language,
            instr_key,
            default=t(
                self.language,
                "instr_praise",
                default="Specific praise. Show WHAT works and WHY. No vague compliments, only concrete analysis with examples."
            )
        )
        result = self.generate_response(text, temporary_system_instruction=instr, command="praise")
        return result + truncation_msg if was_truncated else result

    def reset(self):
        self.history = []
        self.session_summary = None
        self.messages_since_summary = 0
        print("Writer session reset")

    # --- Corrector: typography, grammar, orthography ---
    def correct_text(self, text: str) -> str:
        if not self.client:
            return "Error: LLM client not initialized."

        text, was_truncated = self._truncate_for_context(text)
        truncation_msg = (
            f"\n\n[ВНИМАНИЕ: Текст обрезан до {self.MAX_CONTEXT_CHARS} символов из-за ограничения контекста.]"
            if self.language == "ru"
            else f"\n\n[NOTE: Text truncated to {self.MAX_CONTEXT_CHARS} chars due to context limit.]"
        )
        
        if self.language == "ru":
            instr = """Ты — профессиональный корректор. Твоя задача — исправить текст, НЕ меняя стиль автора.
            
            ЧТО ДЕЛАТЬ:
            1. Исправь орфографические и пунктуационные ошибки.
            2. Типографика: замени дефисы на тире (—) там, где это нужно; используй кавычки «ёлочки» (вложенные — „лапки“).
            3. Убери лишние пробелы.
            
            ЧЕГО НЕ ДЕЛАТЬ:
            1. Не меняй слова и формулировки, если они не ошибочны.
            2. Не "улучшай" стиль.
            3. Не добавляй комментариев.
            
            Выведи ТОЛЬКО исправленный текст."""
        else:
            instr = """You are a professional proofreader. Your task is to correct the text WITHOUT changing the author's style.
            
            WHAT TO DO:
            1. Fix spelling and punctuation errors.
            2. Typography: use proper dashes (—) and quotes.
            3. Remove extra spaces.
            
            WHAT NOT TO DO:
            1. Do not change words or phrasing unless they are erroneous.
            2. Do not "improve" the style.
            3. Do not add commentary.
            
            Output ONLY the corrected text."""
            
        clean_suffix = "\n\nВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов." if self.language == "ru" else "\n\nIMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements."
        instr += clean_suffix
        
        try:
            r = create_chat_completion(
                self.client,
                reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                model=self.model,
                messages=[
                    {"role": "system", "content": instr},
                    {"role": "user", "content": text},
                ],
                temperature=0.1, # Low temp for precision
                max_tokens=min(8000, max(2000, len(text) // 2)),
            )
            result = visible_completion_text(r)
            return result + truncation_msg if was_truncated else result
        except Exception as e:
            log_completion_error(e)
            return "Error: " + t(self.language, "error_llm")

    # --- Editor: clichés and bad expressions, 1–3 alternatives ---
    def edit_text(self, text: str) -> str:
        if not self.client:
            return "Error: LLM client not initialized."
        text, was_truncated = self._truncate_for_context(text)
        truncation_msg = (
            f"\n\n[ВНИМАНИЕ: Текст обрезан до {self.MAX_CONTEXT_CHARS} символов из-за ограничения контекста.]"
            if self.language == "ru"
            else f"\n\n[NOTE: Text truncated to {self.MAX_CONTEXT_CHARS} chars due to context limit.]"
        )
        clean_suffix = "\n\nВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов." if self.language == "ru" else "\n\nIMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements."
        instr = """You are a sharp editor. For the given text: (1) Identify 1–3 clichés, dead metaphors, or weak expressions. (2) For each, offer 1–3 concrete alternative phrasings that are fresh, precise, and natural. Avoid awkward or overly poetic alternatives—suggest what a good writer would actually write. The alternatives should sound like real prose, not like a thesaurus threw up. Be concise. Use the same language as the input. Format clearly (e.g. "Original: ... → Alternatives: ..."). You MUST cite at least one principle from the provided editorial books with author name (e.g., 'As Nora Gal notes on dead metaphors...' or 'По Галь, штамп...'). Always name your source.""" + clean_suffix
        result = self.generate_response(text, temporary_system_instruction=instr, command="editor")
        return result + truncation_msg if was_truncated else result

    # --- Methodique: RAG + analyze text against method book canon ---
    def methodique(self, text_or_idea: str) -> str:
        text_or_idea, was_truncated = self._truncate_for_context(text_or_idea)
        truncation_msg = (
            f"\n\n[ВНИМАНИЕ: Текст обрезан до {self.MAX_CONTEXT_CHARS} символов из-за ограничения контекста.]"
            if self.language == "ru"
            else f"\n\n[NOTE: Text truncated to {self.MAX_CONTEXT_CHARS} chars due to context limit.]"
        )
        level = self._get_detail_level(len(text_or_idea))
        instr_key = f"instr_methodique_{level}"
        base_instr = t(
            self.language,
            instr_key,
            default=t(
                self.language,
                "instr_methodique",
                default="Analyze the text using method book canon (King, Truby, McKee, Vogler, Snyder). Point out: 1) What follows the canon (with quote from book). 2) What violates it (with quote). 3) How to fix it. Cite specific books and authors."
            )
        )
        
        # Build diverse context by explicitly querying for different authors to avoid Vogler bias
        diverse_contexts = []
        if self.rag and self.use_rag:
            author_queries = [
                "Stephen King writing craft",
                "John Truby plot structure", 
                "Robert McKee story principles",
                "Christopher Vogler hero journey",
                "Blake Snyder save the cat beats"
            ]
            for query in author_queries:
                ctx = self.rag.get_context_for_query(query, max_chunks=1, command=None)
                if ctx:
                    diverse_contexts.append(ctx)
        
        # If no diverse contexts, use whatever is available from regular RAG
        if not diverse_contexts and self.rag and self.use_rag:
            ctx = self.rag.get_context_for_query(text_or_idea, max_chunks=3, command="methodique")
            if ctx:
                diverse_contexts = [ctx]
        
        # If still no context, fall back to regular generate_response with automatic RAG
        if not diverse_contexts:
            result = self.generate_response(text_or_idea, temporary_system_instruction=base_instr, command="methodique")
            return result + truncation_msg if was_truncated else result
        
        # Combine contexts and create instruction
        combined = "\n\n=== КОНТЕКСТ ИЗ МЕТОДИЧЕК ===\n\n" + "\n\n".join(diverse_contexts[:4]) + "\n\n=== КОНЕЦ КОНТЕКСТА ==="
        
        # Build instruction that uses context when available, but allows general knowledge
        if self.language == "ru":
            strict_instr = f"""{combined}

Ты — литературный редактор с двадцатилетним стажем. Твоя задача — анализировать текст через призму методичек (Кинг, Труби, Макки, Воглер, Снайдер и др.).

ВАЖНО: Всегда давай конкретный анализ текста. НИКОГДА не говори "в контексте нет цитат" или подобное. Если контекст не подходит — используй общие знания о методичках.

ПРАВИЛА:
1. Всегда анализируй текст конкретно: что работает, что нарушает канон, как исправить
2. Если в контексте есть подходящие цитаты — приводи их с автором и книгой
3. Если цитат нет — используй общие принципы методичек без лишних оправданий
4. Не давай пустых определений — только конкретные советы по тексту
5. Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов, без эмодзи, без markdown-форматирования.

Инструкция: {base_instr}"""
        else:
            strict_instr = f"""{combined}

You are a literary editor with twenty years of experience. Your task is to analyze text through the lens of method books (King, Truby, McKee, Vogler, Snyder, etc.).

IMPORTANT: Always provide specific text analysis. NEVER say "no quotes in context" or similar. If context doesn't fit — use general knowledge about method books.

RULES:
1. Always analyze the text specifically: what works, what violates canon, how to fix
2. If context has relevant quotes — cite them with author and book
3. If no quotes — use general method book principles without unnecessary excuses
4. No empty definitions — only concrete advice for the text
5. Clean text only. No asterisks, no emojis, no decorative elements, no markdown formatting.

Instruction: {base_instr}"""

        
        # Build messages manually to avoid automatic RAG fetching additional context
        if not self.client:
            return "Error: LLM client not initialized."
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": strict_instr},
            {"role": "user", "content": text_or_idea}
        ]
        
        try:
            r = create_chat_completion(
                self.client,
                reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                model=self.model,
                messages=messages,
                temperature=0.2,  # Lower temperature to reduce hallucination
                max_tokens=1000,
            )
            result = visible_completion_text(r)
            return result + truncation_msg if was_truncated else result
        except Exception as e:
            log_completion_error(e)
            return "Error: " + t(self.language, "error_llm")

    # --- Methodique Random: random RAG from craft books when user sends just "Методичка" ---
    def methodique_random(self) -> str:
        """Generate random method book insights - when user sends just 'Методичка'."""
        if not self.client:
            return "Error: LLM client not initialized."
        
        # Load craft data and pick random chunk
        data_path = Path(__file__).parent.parent / "data" / "craft.json"
        if not data_path.exists():
            return "Error: Method books data not found."
        
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                craft_data = json.load(f)
            
            if not craft_data:
                return "Error: No method books data available."
            
            # Pick random chunk
            random_chunk = random.choice(craft_data)
            context = random_chunk.get("text", "")
            author = random_chunk.get("author", "Unknown")
            book_title = random_chunk.get("book_title", "Unknown")
            
            if not context:
                return "Error: Empty context selected."
            
            # Truncate if too long
            if len(context) > 1500:
                context = context[:1500] + "..."
            
            # Create instruction for generating insights
            if self.language == "ru":
                instruction = f"""Ты — литературный редактор с двадцатилетним стажем. 
Тебе выпал случайный отрывок из методички по писательскому мастерству.

Отрывок:
{context}

Твоя задача — извлечь из этого отрывка ОДИН конкретный урок или принцип (2-3 предложения максимум).
Напиши кратко: что это за урок и как писатель может применить его прямо сейчас.

Тон: профессиональный, конкретный. Без воды.

ВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов, без эмодзи, без markdown-форматирования типа **жирный** или *курсив*. Просто текст."""
            else:
                instruction = f"""You are a literary editor with twenty years of experience.
You've received a random excerpt from a writing craft book.

Excerpt:
{context}

Your task: extract ONE specific lesson or principle from this excerpt (2-3 sentences max).
Write briefly: what is the lesson and how can a writer apply it right now.

Tone: professional, specific. No fluff.

IMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements, no markdown formatting like **bold** or *italic*. Just plain text."""

            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": instruction},
                {"role": "user", "content": "Share this method book wisdom."}
            ]
            
            # Call LLM
            r = create_chat_completion(
                self.client,
                reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=400,
            )
            content = visible_completion_text(r)
            
            # Add author and book citation at the end
            # Translate known authors and books to English if needed
            author_translations = {
                "Стивен Кинг": "Stephen King",
                "Джон Труби": "John Truby",
                "Роберт Макки": "Robert McKee",
                "Кристофер Воглер": "Christopher Vogler",
                "Блейк Снайдер": "Blake Snyder",
                "Анна Ламотт": "Anne Lamott",
                "Рэй Брэдбери": "Ray Bradbury",
                "Нора Галь": "Nora Gal",
                "Михаил Есенгазиев": "Amir Esenaziev",
                "Никита Ротару": "Nichita Rotaru",
            }
            book_translations = {
                "Как писать книги": "On Writing",
                "Анатомия истории": "The Anatomy of Story",
                "История на миллион долларов": "Story",
                "Путешествие писателя": "The Writer's Journey",
                "Спасите котика!": "Save the Cat!",
                "Птица за птицей": "Bird by Bird",
                "Дзен в искусстве написания книг": "Zen in the Art of Writing",
                "Слово живое и мёртвое": "The Living and the Dead Word",
                "Писательство. Техника, опыт, мастерство": "Writing. Technique, Experience, Mastery",
                "Техника писательского мастерства": "Writing Technique",
            }
            
            if self.language == "ru":
                citation = f"\n\n— {author}, «{book_title}»"
            else:
                # Translate to English
                author_en = author_translations.get(author, author)
                book_en = book_translations.get(book_title, book_title)
                citation = f"\n\n— {author_en}, \"{book_en}\""
            
            return content + citation
            
        except Exception as e:
            log_completion_error(e)
            return "Error: " + t(self.language, "error_llm")

    # --- Cite: random writer's diary citation + writer name ---
    def cite(self) -> tuple[str, str]:
        """Return (quote, writer_name) filtered by language."""
        data_path = Path(__file__).parent.parent / "data" / "writer_diary_citations.json"
        if data_path.exists():
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                if isinstance(items, list) and items:
                    # Filter by language based on quote content
                    def is_russian_quote(item):
                        quote = item.get("quote", "") if isinstance(item, dict) else item
                        # Check for Cyrillic characters
                        return any('\u0400' <= c <= '\u04FF' for c in quote)
                    
                    if self.language == "ru":
                        # Filter for Russian quotes
                        ru_items = [item for item in items if is_russian_quote(item)]
                        if ru_items:
                            items = ru_items
                    else:
                        # Filter for English quotes (non-Russian)
                        en_items = [item for item in items if not is_russian_quote(item)]
                        if en_items:
                            items = en_items
                    
                    item = random.choice(items)
                    if isinstance(item, dict):
                        return (item.get("quote", ""), item.get("writer", "Unknown"))
                    if isinstance(item, str):
                        return (item, "Unknown")
            except Exception as e:
                print(f"Cite load error: {e}")
        # Fallback: localized quotes from i18n
        fallback_quotes = t(self.language, "cite_fallback_quotes", default=None)
        if fallback_quotes and isinstance(fallback_quotes, list) and fallback_quotes:
            item = random.choice(fallback_quotes)
            if isinstance(item, dict):
                return (item.get("quote", ""), item.get("writer", "Unknown"))
        # Fallback: LLM-generated citation-style line
        if self.client:
            try:
                lang_prompt = "на русском языке" if self.language == "ru" else "in English"
                clean_suffix = " No emojis, no asterisks, no decorative elements." if self.language != "ru" else " Без эмодзи, без звёздочек, без декоративных элементов."
                r = create_chat_completion(
                    self.client,
                    reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"You output a single short, poignant sentence that sounds like a line from a writer's diary {lang_prompt}. Then a dash and the writer's name. Example: 'The blank page is a mirror — Kafka.' Output only that one line, no quotes or extra text.{clean_suffix}"},
                        {"role": "user", "content": "One diary-style citation and writer name."},
                    ],
                    temperature=0.9,
                    max_tokens=150,
                )
                raw = visible_completion_text(r)
                if " — " in raw:
                    quote, writer = raw.rsplit(" — ", 1)
                    return (quote.strip(), writer.strip())
                if raw:
                    return (raw, "Unknown")
            except Exception as e:
                log_completion_error(e)
        if self.language == "ru":
            return ("Чистый лист — это зеркало", "Неизвестно")
        return ("The blank page is a mirror", "Unknown")

    # --- Lobster: suffocating pessimistic speech (or small chance love for Adelina) ---
    def lobster(self) -> list[str]:
        """Returns a list of message strings to send one by one."""
        roll = random.random()
        love_for_adelina = roll < 0.07  # ~7% chance
        print(f"[LOBSTER] love_for_adelina={love_for_adelina}, roll={roll:.4f}")
        if love_for_adelina:
            # Use i18n instruction that explicitly requires mentioning "Adeline"
            clean_suffix = "\n\nВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов." if self.language == "ru" else "\n\nIMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements."
            instr = t(self.language, "instr_lobster_love", default="Write a love letter from the lobster to Adeline. Explicitly mention 'Adeline' by name in the text. Themes: doomed love, the fragility of beauty, the approaching end. Tone: romantic, melancholic, desperate. Maximum 300 characters. Output only the letter, nothing else.") + clean_suffix
            out = self.generate_response("Love letter to Adeline.", temporary_system_instruction=instr)
            print(f"[LOBSTER] Adelina response: {out[:100] if out else 'None'}...")
            if self.language == "ru":
                fallback = ["аделина, ты — единственный свет в этой тьме"]
            else:
                fallback = ["adelina, you are the only light in this darkness"]
            return [out] if out and not out.startswith("Error") else fallback
        
        # Regular lobster monologue - pick a random theme for the whole pack
        themes_ru = [
            "смертность и конечность бытия",
            "абсурд существования",
            "тяжесть сознания",
            "научная фантастика и бессилие разума перед космосом",
            "Стругацкие: прогрессоры, малыши и безысходность понимания",
            "тепловая смерть вселенной",
            "забвение и бессмысленность памяти",
            "японская культура: ваби-саби, моно-но аварэ, красота увядания",
            "антинатализм: грех рождения, лучше не родиться",
            "обречённость человечества: мы сами себе конец",
            "бездушные нейронки заменят нас всех"
        ]
        themes_en = [
            "mortality and finitude of being",
            "absurdity of existence",
            "weight of consciousness",
            "science fiction and helplessness of mind before cosmos",
            "Strugatsky: progressors, little ones and hopelessness of understanding",
            "heat death of the universe",
            "oblivion and meaninglessness of memory",
            "japanese culture: wabi-sabi, mono no aware, beauty of decay",
            "antinatalism: the sin of birth, better never to have been",
            "doom of humanity: we are our own end",
            "soulless neural networks will replace us all"
        ]
        
        themes = themes_ru if self.language == "ru" else themes_en
        chosen_theme = random.choice(themes)
        print(f"[LOBSTER] Theme: {chosen_theme}")
        
        base_instr = t(self.language, "instr_lobster", default="You are a lobster. Write 3–5 short messages. Each message is EXACTLY ONE sentence, lowercase, no capitals, NO DOTS at the end. Separate messages with the line: ---MESSAGE---\n\nExample format:\nwe are all already dead, we just haven't stopped moving\n---MESSAGE---\nevery breath is a debt we pay to the void\n---MESSAGE---\nwhen the pot lid slams shut you'll remember my words\n\nThemes: mortality, absurdity, weight of consciousness, inevitability of the pot. Tone: bleak, lyrical. Only the monologue, nothing else.")
        
        # Inject the chosen theme into the instruction
        instr = base_instr.replace(
            "Themes: mortality, absurdity, weight of consciousness, inevitability of the pot.",
            f"Theme for this monologue (all messages on this one topic): {chosen_theme}."
        ).replace(
            "Темы: смертность, абсурд, тяжесть сознания, неизбежность кастрюли.",
            f"Тема этого монолога (все сообщения на эту одну тему): {chosen_theme}."
        )
        
        # Add clean text instruction
        clean_suffix = "\n\nВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов." if self.language == "ru" else "\n\nIMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements."
        instr = instr + clean_suffix
        if self.language == "ru":
            fallback = [
                "мы все уже мертвы; мы просто ещё не остановились",
                "---MESSAGE---",
                "каждый вздох — это долг, который мы выплачиваем пустоте",
                "---MESSAGE---",
                "когда крышка кастрюли захлопнется — ты вспомнишь мои слова"
            ]
        else:
            fallback = [
                "we are all already dead; we just haven't stopped moving yet",
                "---MESSAGE---",
                "every breath is a debt we pay to the void",
                "---MESSAGE---",
                "when the pot lid slams shut — you'll remember my words"
            ]
        
        out = self.generate_response("Lobster monologue.", temporary_system_instruction=instr)
        print(f"[LOBSTER] Regular response: {out[:150] if out else 'None'}...")
        if not out or out.startswith("Error"):
            # Extract actual messages from fallback list (elements at even indices: 0, 2, 4...)
            messages = [fallback[i].strip() for i in range(0, len(fallback), 2) if fallback[i].strip()]
            return messages if messages else ["the lobster is silent..."]  # Last resort fallback
        
        parts = [p.strip() for p in out.split("---MESSAGE---") if p.strip()]
        print(f"[LOBSTER] Parsed parts: {len(parts)} messages")
        return parts if parts else [out]

    # --- XYX: short random interjection ---
    def porko(self) -> str:
        """Return random short interjection."""
        return random.choice(["хух", "хухни"])

    # --- Pun: wordplay on a single word from the last message ---
    def pun(self, last_message: str) -> str:
        if not self.client or not last_message.strip():
            return "Send me a message first, then /pun."
        # Pick a random word from the message (skip short words)
        words = [w.strip(".,!?;:\"'()[]{}") for w in last_message.split() if len(w.strip(".,!?;:\"'()[]{}")) >= 3]
        target_word = random.choice(words) if words else last_message.split()[0] if last_message.split() else "word"
        
        # Pre-defined wordplay patterns for common words to avoid LLM nonsense
        simple_puns_ru = {
            "писатель": ["писатель — глазатель", "писатель — казатель", "писатель — спасатель (нет)"],
            "книга": ["книга — фига", "книга — интрига", "книга — верига"],
            "текст": ["текст — квест", "текст — арест", "текст — манифест"],
            "сюжет": ["сюжет — винегрет", "сюжет — паштет", "сюжет — бюджет"],
            "герой": ["герой — геморрой", "герой — отстой", "герой — ковбой"],
            "автор": ["автор — завтрак", "автор — призрак", "автор — диктатор"],
            "читатель": ["читатель — мечтатель", "читатель — пытатель", "читатель — глотатель"],
            "роман": ["роман — карман", "роман — дурман", "роман — баклан"],
            "страница": ["страница — девица", "страница — темница", "страница — горчица"],
            "глава": ["глава — забава", "глава — отрава", "глава — канава"],
            "слово": ["слово — олово", "слово — полово", "слово — сурово"],
            "финал": ["финал — скандал", "финал — провал", "финал — обвал"],
            "конец": ["конец — холодец", "конец — венец", "конец — капец"],
        }
        
        simple_puns_en = {
            "writer": ["writer — fighter", "writer — lighter"],
            "book": ["book — cook", "book — look"],
            "text": ["text — next", "text — vexed"],
            "plot": ["plot — not", "plot — forgot"],
            "hero": ["hero — zero", "hero — weirdo"],
            "author": ["author — offer", "author — bother"],
            "reader": ["reader — leader", "reader — feeder"],
            "novel": ["novel — grovel", "novel — hovel"],
            "page": ["page — rage", "page — cage"],
            "chapter": ["chapter — rapture", "chapter — capture"],
            "word": ["word — absurd", "word — heard"],
            "end": ["end — friend", "end — bend"],
            "story": ["story — glory", "story — gory"],
        }
        
        # Check if we have a pre-defined pun for this word
        target_lower = target_word.lower()
        if self.language == "ru" and target_lower in simple_puns_ru:
            return random.choice(simple_puns_ru[target_lower])
        if self.language == "en" and target_lower in simple_puns_en:
            return random.choice(simple_puns_en[target_lower])
        
        # Fallback to LLM with lower temperature and stricter prompt
        clean_suffix = "\n\nВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов." if self.language == "ru" else "\n\nIMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements."
        instr = f"""Create a simple, natural-sounding pun or rhyme for the word "{target_word}".

Rules:
- Use common, everyday words only
- Keep it to 2-4 words maximum
- Must actually sound like the original word (phonetic similarity)
- No forced metaphors, no abstract nonsense
- Examples of good puns: "deadline — breadline", "coffee — cough-fee", "night — knight"

Bad examples (avoid): "coffee — philosophical awakening", "deadline — temporal boundary"

Output ONLY the pun, no explanation.""" + clean_suffix
        try:
            r = create_chat_completion(
                self.client,
                reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                model=self.model,
                messages=[
                    {"role": "system", "content": instr},
                    {"role": "user", "content": f'Create a pun for the word: "{target_word}"'},
                ],
                temperature=0.4,  # Lower temperature for more controlled output
                max_tokens=50,    # Shorter response
            )
            result = visible_completion_text(r) or "No pun today."
            # Clean up quotes and limit length
            result = result.strip('"\'').split('\n')[0][:100]
            return result
        except Exception as e:
            log_completion_error(e)
            return "Error: " + t(self.language, "error_llm")

    # --- Generate new prompt via LLM ---
    def generate_fresh_prompt(self) -> str:
        """Generate a new writing prompt using LLM based on existing patterns."""
        if not self.client:
            return self._local_random_prompt()
        
        # Get some existing prompts as examples - use language-specific data
        data = self._load_prompts_data()
        existing = data.get("prompts", [])[:5]
        examples = "\n".join(f"- {p}" for p in existing)
        
        clean_suffix = "\n\nВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов." if self.language == "ru" else "\n\nIMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements."
        if self.language == "ru":
            instr = f"""Ты — писатель с двадцатилетним стажем. Придумай один новый креативный промпт для писателей.

Примеры существующих:
{examples}

Требования к новому промпту:
- Конкретная, необычная задача
- Один-два предложения
- Без объяснений, что делать — только сам промпт
- Может включать ограничение (например, без какой-то буквы), точку зрения, начальную фразу, или странную ситуацию

Выведи ТОЛЬКО сам промпт, без кавычек вокруг.""" + clean_suffix
        else:
            instr = f"""You are a writer with twenty years of experience. Create one new creative writing prompt.

Existing examples:
{examples}

Requirements for the new prompt:
- Specific, unusual task
- One or two sentences
- No explanations of what to do — just the prompt itself
- Can include constraints (e.g., without a certain letter), point of view, opening line, or strange situation

Output ONLY the prompt itself, no quotes around it.""" + clean_suffix
        
        try:
            r = create_chat_completion(
                self.client,
                reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                model=self.model,
                messages=[
                    {"role": "system", "content": instr},
                    {"role": "user", "content": "Generate a fresh writing prompt."},
                ],
                temperature=0.9,
                max_tokens=200,
            )
            result = visible_completion_text(r)
            # Clean up quotes if LLM added them
            result = result.strip('"\'')
            return result if result else self._local_random_prompt()
        except Exception:
            return self._local_random_prompt()

    # --- Cry_baby: offer to cry, then reply with "0 words saved" etc ---
    def cry_baby_reply(self) -> str:
        """Cunning reply after user 'cried' (used when they answer after /cry_baby)."""
        # Try language-specific file first
        lang_suffix = f".{self.language}" if self.language != "en" else ""
        data_path = Path(__file__).parent.parent / "data" / f"cry_baby_replies{lang_suffix}.json"
        
        # Fallback to default file if language-specific doesn't exist
        if not data_path.exists():
            data_path = Path(__file__).parent.parent / "data" / "cry_baby_replies.json"
        
        if data_path.exists():
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    variants = json.load(f)
                if isinstance(variants, list) and variants:
                    return random.choice(variants)
            except Exception:
                pass
        
        # Fallback to hardcoded i18n strings
        if self.language == "ru":
            variants = [
                "0 слов в статистику. Но иногда это именно то, что нужно душе.",
                "0 слов в статистику. Страница всё ещё пуста — и это нормально.",
                "0 слов в статистику. Плач считается за пунктуацию.",
                "0 слов в статистику. Ты пришёл. Это уже предложение.",
            ]
        else:
            variants = [
                "0 words saved to Stats. But sometimes that's exactly what the soul needed.",
                "0 words saved to Stats. The page is still blank — and that's okay.",
                "0 words saved to Stats. Crying counts as punctuation.",
                "0 words saved to Stats. You showed up. That's already a sentence.",
            ]
        return random.choice(variants)
