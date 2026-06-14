#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'src')

# Set dummy API key to avoid initialization error
os.environ['OPENAI_API_KEY'] = 'dummy-key-for-testing'

from writer_bot import WriterBot


print("Testing Russian prompts...")
bot_ru = WriterBot(language='ru', use_rag=False)
data_ru = bot_ru._load_prompts_data()

print(f"Keys: {list(data_ru.keys())}")
print(f"idea_genre: {len(data_ru.get('idea_genre', []))} items")
print(f"idea_setting: {len(data_ru.get('idea_setting', []))} items")  
print(f"idea_conflict: {len(data_ru.get('idea_conflict', []))} items")

print("\nTesting English prompts...")
bot_en = WriterBot(language='en', use_rag=False)
data_en = bot_en._load_prompts_data()

print(f"Keys: {list(data_en.keys())}")
print(f"idea_genre: {len(data_en.get('idea_genre', []))} items")

print("\nGenerating ideas:")
print("RU:", bot_ru.generate_idea())
print("EN:", bot_en.generate_idea())
