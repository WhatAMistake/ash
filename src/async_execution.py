"""Bounded offloading and keyed serialization for synchronous writer work."""

import asyncio
from functools import partial


class AsyncExecution:
    def __init__(self, max_workers: int = 4):
        self._slots = asyncio.Semaphore(max_workers)
        self._locks = {}

    async def run(self, function, /, *args, **kwargs):
        async with self._slots:
            task = asyncio.create_task(asyncio.to_thread(partial(function, *args, **kwargs)))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A thread cannot be cancelled. Keep its slot and the caller's
                # session lock until it has stopped mutating shared state, even
                # if the awaiting task is cancelled again during shutdown.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if task.done() and not task.cancelled():
                    task.exception()  # Consume any failure from the worker.
                raise

    async def serialized(self, key, function, /, *args, **kwargs):
        entry = self._locks.get(key)
        if entry is None:
            entry = [asyncio.Lock(), 0]
            self._locks[key] = entry
        entry[1] += 1
        try:
            async with entry[0]:
                return await function(*args, **kwargs)
        finally:
            entry[1] -= 1
            if not entry[1]:
                del self._locks[key]
