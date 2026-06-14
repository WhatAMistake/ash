"""
Core logic for Writer's Tears idea generation.
Separated from bot logic to be used by both Telegram bot and Web API.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class StoryIdea:
    genre: str
    setting: str
    conflict: str
    
    def __str__(self) -> str:
        return f"{self.genre} + {self.setting} + {self.conflict}"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "genre": self.genre,
            "setting": self.setting,
            "conflict": self.conflict
        }

class IdeaGenerator:
    """Generates creative writing prompts based on Genre + Setting + Conflict formula."""
    
    def __init__(self, data_dir: Optional[Path] = None, language: str = "ru"):
        self.language = language
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"
        self._prompts_data: Optional[Dict] = None
        
    def _load_data(self) -> Dict:
        """Load prompts data from JSON file."""
        if self._prompts_data is not None:
            return self._prompts_data
            
        # Try language-specific file first
        lang_suffix = f".{self.language}" if self.language != "en" else ""
        data_path = self.data_dir / f"writing_prompts{lang_suffix}.json"
        
        if not data_path.exists():
            # Fallback to default
            data_path = self.data_dir / "writing_prompts.json"
            
        try:
            if data_path.exists():
                with open(data_path, "r", encoding="utf-8") as f:
                    self._prompts_data = json.load(f)
            else:
                self._prompts_data = self._default_data()
        except Exception as e:
            print(f"Error loading prompts data: {e}")
            self._prompts_data = self._default_data()
            
        return self._prompts_data

    def _default_data(self) -> Dict:
        """Fallback data if files are missing."""
        if self.language == "ru":
            return {
                "idea_genre": ["драма", "комедия", "триллер"],
                "idea_setting": ["город", "деревня", "космос"],
                "idea_conflict": ["любовь", "смерть", "деньги"]
            }
        return {
            "idea_genre": ["drama", "comedy", "thriller"],
            "idea_setting": ["city", "village", "space"],
            "idea_conflict": ["love", "death", "money"]
        }

    def generate(self) -> StoryIdea:
        """Generate a random story idea."""
        data = self._load_data()
        
        genre = random.choice(data.get("idea_genre", ["literary"]))
        setting = random.choice(data.get("idea_setting", ["small town"]))
        conflict = random.choice(data.get("idea_conflict", ["secret"]))
        
        return StoryIdea(genre=genre, setting=setting, conflict=conflict)

    def get_random_prompt(self) -> str:
        """Get a random writing prompt (one-liner)."""
        data = self._load_data()
        prompts = data.get("prompts", [])
        if not prompts:
            return "Write something."
        return random.choice(prompts)

    def get_random_exercise(self) -> Dict:
        """Get a random writing exercise."""
        data = self._load_data()
        exercises = data.get("exercises", [])
        if not exercises:
            return {"title": "Free writing", "description": "Write for 5 minutes."}
        return random.choice(exercises)
