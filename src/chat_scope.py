"""Isolate transient writer state by group and user while retaining private keys."""

from contextvars import ContextVar

current_group = ContextVar("ash_current_group", default=None)


class ChatScopedDict(dict):
    """A user-keyed dict whose group keys are (chat_id, user_id)."""

    @staticmethod
    def _key(key):
        chat = current_group.get()
        return (chat, key) if chat is not None and isinstance(key, int) else key

    def __getitem__(self, key):
        return super().__getitem__(self._key(key))

    def __setitem__(self, key, value):
        super().__setitem__(self._key(key), value)

    def __delitem__(self, key):
        super().__delitem__(self._key(key))

    def __contains__(self, key):
        return super().__contains__(self._key(key))

    def get(self, key, default=None):
        return super().get(self._key(key), default)

    def pop(self, key, *default):
        return super().pop(self._key(key), *default)
