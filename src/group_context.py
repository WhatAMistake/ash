"""Bounded, restart-safe links between group posts and individual discussions."""

import json
import sqlite3
import time
import uuid
from pathlib import Path


class GroupContextStore:
    TTL_SECONDS = 7 * 24 * 3600
    MAX_MESSAGES = 5000
    MAX_CONVERSATIONS = 1000
    MAX_TURNS = 10
    MAX_TEXT = 4000
    MAX_TURN_TEXT = 24000

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
                    kind TEXT NOT NULL, text TEXT NOT NULL, created REAL NOT NULL,
                    episode TEXT NOT NULL, owner INTEGER, conversation TEXT,
                    PRIMARY KEY (chat_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS message_episode ON messages(chat_id, episode, message_id);
                CREATE TABLE IF NOT EXISTS conversations (
                    chat_id INTEGER NOT NULL, owner INTEGER NOT NULL,
                    conversation TEXT NOT NULL, turns TEXT NOT NULL, updated REAL NOT NULL,
                    PRIMARY KEY (chat_id, owner, conversation)
                );
            """)
            self._prune(db)

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=5)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def new_id():
        return uuid.uuid4().hex

    def _prune(self, db):
        cutoff = time.time() - self.TTL_SECONDS
        db.execute("DELETE FROM messages WHERE created < ?", (cutoff,))
        db.execute("DELETE FROM conversations WHERE updated < ?", (cutoff,))
        db.execute("DELETE FROM messages WHERE rowid IN (SELECT rowid FROM messages "
                   "ORDER BY created DESC, rowid DESC LIMIT -1 OFFSET ?)", (self.MAX_MESSAGES,))
        db.execute("DELETE FROM conversations WHERE rowid IN (SELECT rowid FROM conversations "
                   "ORDER BY updated DESC, rowid DESC LIMIT -1 OFFSET ?)", (self.MAX_CONVERSATIONS,))

    def record(self, chat_id, message_id, kind, text, episode, owner=None, conversation=None):
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (chat_id, message_id, kind, str(text)[:self.MAX_TEXT], time.time(),
                        episode, owner, conversation))
            self._prune(db)

    def resolve(self, chat_id, message_id, user_id):
        with self._connect() as db:
            self._prune(db)
            row = db.execute("SELECT * FROM messages WHERE chat_id=? AND message_id=?",
                             (chat_id, message_id)).fetchone()
            if row is None:
                return None
            parts = db.execute("SELECT message_id, text FROM messages WHERE chat_id=? AND episode=? "
                               "ORDER BY message_id", (chat_id, row["episode"])).fetchall()
            conversation = row["conversation"] if row["owner"] == user_id else None
            turns = []
            if conversation:
                saved = db.execute("SELECT turns FROM conversations WHERE chat_id=? AND owner=? "
                                   "AND conversation=?", (chat_id, user_id, conversation)).fetchone()
                if saved:
                    turns = json.loads(saved["turns"])
                else:
                    conversation = None
            return {
                "kind": row["kind"], "text": row["text"],
                "parts": [part["text"] for part in parts],
                "part": next(i + 1 for i, part in enumerate(parts) if part["message_id"] == message_id),
                "conversation": conversation, "turns": turns,
            }

    def save_turns(self, chat_id, user_id, conversation, turns):
        bounded = [{"role": turn["role"], "content": turn["content"][:self.MAX_TURN_TEXT]}
                   for turn in turns[-self.MAX_TURNS:]]
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?, ?)",
                       (chat_id, user_id, conversation, json.dumps(bounded, ensure_ascii=False), time.time()))
            self._prune(db)
