"""Public command catalogue used by Telegram's menu."""

# Command, English description, Russian description. Keep distinct voices visible.
COMMANDS = (
    ("start", "Start Ash", "Начать"),
    ("help", "Commands and usage", "Команды и подсказки"),
    ("block", "Get unstuck", "Преодолеть творческий блок"),
    ("develop", "Develop an idea", "Развить идею"),
    ("character", "Develop a character", "Разработать персонажа"),
    ("dialogue", "Work on dialogue", "Работа над диалогом"),
    ("idea", "Random story idea and optional reference", "Случайная идея и необязательный референс"),
    ("prompt", "Random writing prompt", "Случайное писательское задание"),
    ("feedback", "Constructive critique", "Конструктивная критика"),
    ("roast", "Sharp, ironic roast", "Жёсткая ироничная прожарка"),
    ("praise", "Reasoned praise", "Аргументированная похвала"),
    ("style", "Analyze style", "Анализ стиля"),
    ("corrector", "Proofread text", "Корректура текста"),
    ("editor", "Find clichés and alternatives", "Штампы и варианты"),
    ("methodique", "Analyze using available craft books", "Разбор по доступным методичкам"),
    ("summary", "Summarize text", "Краткое содержание"),
    ("upload", "Upload TXT, DOCX or PDF", "Загрузить TXT, DOCX или PDF"),
    ("count_me", "Count and record words and characters", "Посчитать и записать слова и знаки"),
    ("stats", "Writing statistics", "Статистика письма"),
    ("done", "Process text or exit discussion", "Обработать текст или выйти из обсуждения"),
    ("cancel", "Cancel current workflow", "Отменить текущий сценарий"),
    ("reset", "Reset chat and writing statistics", "Сбросить диалог и статистику"),
    ("lang", "Show or set language: ru / en", "Показать или задать язык: ru / en"),
    ("switchlang", "Switch language", "Сменить язык"),
    ("lobster", "Lobster monologue", "Монолог лобстера"),
    ("xyx", "Хух", "Хух"),
    ("pun", "Pun on a reply or last message", "Каламбур по ответу или последнему сообщению"),
    ("cry_baby", "Have a cry", "Поплакать"),
    ("cite", "Writer quote", "Цитата писателя"),
    ("cite_on", "Enable daily quotes", "Включить ежедневные цитаты"),
    ("cite_off", "Disable daily quotes", "Отключить ежедневные цитаты"),
    ("cite_when", "When is the next quote?", "Когда следующая цитата?"),
    ("dev_feedback", "Message the developer", "Написать разработчику"),
)


def descriptions(lang):
    index = 2 if lang == "ru" else 1
    return [(row[0], row[index]) for row in COMMANDS]


def command_help(lang):
    lines = ["Ash", ""]
    lines += [f"/{name} — {description}" for name, description in descriptions(lang)]
    if lang == "ru":
        lines += ["", "Текст можно добавить после команды или отправить частями и нажать /done. "
                  "После разбора задай уточнение; /done завершит обсуждение. /cancel отменяет текущий сценарий.",
                  "Файл: .txt, .docx или .pdf до 20 МБ; после загрузки выбери инструмент. "
                  "/methodichque — прежний алиас /methodique.",
                  "В группе продолжай сценарий ответом боту или упоминанием; файлы тоже требуют ответа или упоминания."]
    else:
        lines += ["", "Put text after a command or send it in parts and use /done. "
                  "Ask follow-up questions after a result; /done exits discussion. /cancel cancels the current workflow.",
                  "Upload a .txt, .docx or .pdf up to 20 MB, then choose a tool. "
                  "/methodichque remains an alias for /methodique.",
                  "In groups, continue a workflow by replying to or mentioning Ash; documents also need a reply or mention."]
    return "\n".join(lines)
