import asyncio
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from async_execution import AsyncExecution


@pytest.mark.asyncio
async def test_worker_does_not_block_event_loop_and_limits_concurrency():
    execution = AsyncExecution(max_workers=2)
    started = threading.Event()
    release = threading.Event()
    guard = threading.Lock()
    active = 0
    peak = 0

    def work():
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
            if active == 2:
                started.set()
        release.wait(5)
        with guard:
            active -= 1

    tasks = [asyncio.create_task(execution.run(work)) for _ in range(4)]
    try:
        assert await asyncio.to_thread(started.wait, 3)
        # This coroutine can still run while both worker threads are blocked.
        await asyncio.wait_for(asyncio.sleep(0.01), 1)
        assert peak == 2
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert not execution._locks


@pytest.mark.asyncio
async def test_cancelled_worker_retains_session_lock_until_thread_finishes():
    execution = AsyncExecution()
    started = threading.Event()
    release = threading.Event()
    events = []

    def work():
        events.append("start")
        started.set()
        release.wait(5)
        events.append("end")

    async def first():
        await execution.run(work)

    async def second():
        events.append("second")

    task = asyncio.create_task(execution.serialized(1, first))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        other = asyncio.create_task(execution.serialized(1, second))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        assert events == ["start"]
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await other
    assert events == ["start", "end", "second"]
    assert not execution._locks


@pytest.mark.asyncio
async def test_different_users_are_not_serialized():
    execution = AsyncExecution()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow():
        entered.set()
        await release.wait()

    first = asyncio.create_task(execution.serialized(1, slow))
    await entered.wait()
    try:
        assert await asyncio.wait_for(execution.serialized(2, asyncio.sleep, 0, result="ok"), 1) == "ok"
    finally:
        release.set()
        await first
