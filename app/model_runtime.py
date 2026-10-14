"""Bound synchronous provider calls without blocking the ASGI event loop."""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


async def call_model(model, request, *, native=False):
    fn = getattr(model, 'generate_tools', None) if native else model.generate
    if fn is None:
        raise ValueError('native_tools_unavailable')
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nba-native-model')
    future = executor.submit(fn, request)
    try:
        deadline = time.monotonic() + request.timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError('model deadline')
            await asyncio.sleep(.01)
        return future.result()
    finally:
        future.cancel()
        executor.shutdown(wait=future.done(), cancel_futures=True)
