"""The unit suite never loads .env files or opens network connections."""
import os
import socket

os.environ['PYTHON_DOTENV_DISABLED'] = '1'

import pytest


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network is forbidden in unit tests')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    for name in ('OPENAI_API_KEY', 'INTENT_GUARD_API_KEY', 'TELEGRAM_BOT_TOKEN'):
        monkeypatch.delenv(name, raising=False)
