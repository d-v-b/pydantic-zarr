from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Generator


if TYPE_CHECKING:
    from pydantic_zarr.types import ArrayLike, ArrayV2Config, GroupV2Config


class ZarrV2Writer(abc.ABC):
    @abc.abstractmethod
    async def write_array(self, *, path: str, metadata: ArrayV2Config) -> ArrayLike[tuple[int, ...]]: ...

    @abc.abstractmethod
    async def write_group(self, *, path: str, metadata: GroupV2Config) -> None: ...

    @classmethod
    @abc.abstractmethod
    def from_url(cls, url: str) -> ZarrV2Writer:
        raise NotImplementedError


import asyncio
import functools
from typing import TypeVar, Generic, Any, Coroutine

T = TypeVar('T')

class Future(Generic[T]):
    """A wrapper for coroutines that adds a blocking .result() method."""
    def __init__(self, coro: Coroutine[Any, Any, T]) -> None:
        self.coro = coro
        self._loop = None
        self._result = None
        self._exception = None
        self._done = False

    def result(self, *, loop: asyncio.AbstractEventLoop | None = None) -> T:
        """Execute the coroutine and return its result. Blocks until completion."""
        if self._done:
            if self._exception:
                raise self._exception
            return self._result

        # Get or create an event loop
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in this thread, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        self._loop = loop

        try:
            # Run the coroutine and get the result
            if loop.is_running():
                # If we're already in an event loop, we need a different approach
                future = asyncio.run_coroutine_threadsafe(self.coro, loop)
                self._result = future.result()
            else:
                # If no loop is running, we can just run the coroutine
                self._result = loop.run_until_complete(self.coro)
        except Exception as e:
            self._exception = e
            raise
        finally:
            self._done = True

        return self._result

    def __await__(self) -> Generator[Any, Any, T]:
        """Make this class awaitable, so it can be used in other async code."""
        return self.coro.__await__()