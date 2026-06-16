"""A small pool of pre-warmed rppg.Model instances.

Model construction is expensive (weight load + JAX JIT warm-up) and a single
instance is NOT concurrency-safe (its `self.alive` guard rejects a second
task). So we build N instances up front and hand them out one-at-a-time.
"""
import asyncio
import logging

from engine import get_rppg, DEFAULT_MODEL

log = logging.getLogger("rppg.pool")


class ModelPool:
    def __init__(self, size: int = 1, model_name: str = DEFAULT_MODEL):
        self.size = size
        self.model_name = model_name
        self._queue: asyncio.Queue = asyncio.Queue()
        self._ready = False

    async def warmup(self) -> None:
        """Build and warm the model instances (runs blocking work in a thread)."""
        for i in range(self.size):
            log.info("Loading model %d/%d (%s)...", i + 1, self.size, self.model_name)
            model = await asyncio.to_thread(get_rppg().Model, self.model_name)
            await self._queue.put(model)
        self._ready = True
        log.info("Model pool ready (%d instance(s)).", self.size)

    @property
    def ready(self) -> bool:
        return self._ready

    def try_acquire(self):
        """Return a free model immediately, or None if all are busy."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def release(self, model) -> None:
        self._queue.put_nowait(model)
