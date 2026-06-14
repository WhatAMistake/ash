import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.idea_generator import IdeaGenerator
from src.core.text_generator import TextGenerator

def test_idea_generation():
    print("Testing Idea Generation...")
    generator = IdeaGenerator(language="ru")
    idea = generator.generate()
    print(f"Generated Idea: {idea}")
    assert idea.genre
    assert idea.setting
    assert idea.conflict
    print("Idea Generation OK")

def test_text_generation():
    print("\nTesting Text Generation (Mock)...")
    # We won't actually call the API in this test to save tokens/time, 
    # but we'll initialize the class
    generator = TextGenerator(language="ru")
    if generator.client:
        print("Text Generator Client Initialized OK")
    else:
        print("Text Generator Client Failed (openai not installed or no key)")

if __name__ == "__main__":
    test_idea_generation()
    test_text_generation()
