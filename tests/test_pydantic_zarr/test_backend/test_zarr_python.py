from __future__ import annotations

from typing import TYPE_CHECKING
from zarr.storage import MemoryStore
if TYPE_CHECKING:
    from pydantic_zarr.types import StoreSpec

from pydantic_zarr.zarr_v2 import ArraySpec


async def test_arrayspec_write() -> None:
    """
    Test that an ArraySpec can be stored using the zarr backend
    """
    arr = ArraySpec(
        shape=(10, 10),
        chunks=(2, 2),
        dtype=">i2",
        fill_value=0,
        order="C",
        compressor={"id": "gzip", "level": 5},
        filters=None,
    )
    # store_spec: StoreSpec = {'engine': 'zarr-python', 'url': 'memory://foo'}
    store = MemoryStore({})
    observed_arr = arr.persist(store=store).result()
    assert observed_tstore.spec().to_json() == expected_spec
