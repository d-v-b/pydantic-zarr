import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorstore as ts

import pydantic_zarr.zarr_v2.array as array
from pydantic_zarr.backend._tensorstore.models import (
    FileDriverParams,
    KVStoreParams,
    MemoryDriverParams,
)
from pydantic_zarr.backend._tensorstore.v2 import (
    SEPARATOR,
    V2_ARRAY_KEY,
    V2_GROUP_KEY,
    get_member_keys,
    read_array_v2,
    read_group_v2,
    read_members_v2,
    write_array_metadata_v2,
    write_group_v2,
)
import pydantic_zarr.zarr_v2.group

if TYPE_CHECKING:
    from pydantic_zarr.types import GroupV2Config, StoreSpec


@pytest.fixture
def kvstore(request: pytest.FixtureRequest, tmp_path: Path) -> FileDriverParams | MemoryDriverParams:
    if request.param == "file":
        return {"driver": "file", "path": str(tmp_path) + "/"}
    elif request.param == "memory":
        return {"driver": "memory", "path": ""}
    raise ValueError(f"Invalid request: {request.param}")


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
@pytest.mark.parametrize("attrs", [None, {}, {"foo": "bar"}])
async def test_read_group_v2(kvstore: KVStoreParams, attrs: dict[str, str] | None) -> None:
    store = await ts.KvStore.open(kvstore)
    metadata = {"zarr_format": 2}
    _ = await store.write(".zgroup", json.dumps(metadata))
    if attrs is not None:
        _ = await store.write(".zattrs", json.dumps(attrs))
        attrs_expected = attrs
    else:
        attrs_expected = {}
    group_meta = await read_group_v2(store)
    assert group_meta == {'zarr_format': 2, 'attributes': attrs_expected}


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
@pytest.mark.parametrize("attrs", [None, {}, {"foo": "bar"}])
async def test_read_array_v2(kvstore: KVStoreParams, attrs: dict[str, str] | None) -> None:
    store = await ts.KvStore.open(kvstore)
    metadata = array.ArrayMetadataSpec(
        shape=(10,), chunks=(2,), dtype=">i2", order="C", zarr_format=2
    ).model_dump(exclude_none=True)
    _ = await store.write(".zarray", json.dumps(metadata))

    if attrs is not None:
        _ = await store.write(".zattrs", json.dumps(attrs))
        attrs_expected = attrs
    else:
        attrs_expected = {}
    array_meta = await read_array_v2(store)
    assert array_meta == metadata | {'attributes' : attrs_expected}


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
async def test_read_members_v2(kvstore: KVStoreParams) -> None:
    """
    Test that reading the members of a group works as expected.
    """
    store = await ts.KvStore.open(kvstore)
    g_metadata: GroupV2Config = {"zarr_format": 2, "attributes": {}}
    a_metadata = array.ArrayMetadataSpec(
        shape=(10,), chunks=(2,), dtype=">i2", order="C", zarr_format=2
    ).model_dump(exclude_none=True)
    a_attrs = {"foo": "bar"}
    _ = store.write(".zgroup", json.dumps(g_metadata)).result()
    _ = store.write("array/.zarray", json.dumps(a_metadata)).result()
    _ = store.write("array/.zattrs", json.dumps(a_attrs)).result()

    observed = await read_members_v2(store)
    expected = {b"": g_metadata, b"array": a_metadata | {'attributes' : a_attrs}}
    assert observed == expected


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
@pytest.mark.parametrize("dtype", ["int32", "float32"])
@pytest.mark.parametrize("shape", [(10,), (10, 10)])
async def test_create_array_v2(kvstore: KVStoreParams, dtype: str, shape: tuple[int, ...]) -> None:
    """
    Test that creating an array with a given shape and dtype works as expected.
    """
    template = np.zeros(shape, dtype=dtype)
    model = array.ArraySpec.from_array(template)
    store = await ts.KvStore.open(kvstore)
    arr = await write_array_metadata_v2(
        model.model_dump(exclude={'attributes'}), kvstore=store, open=False, create=True, delete_existing=True
    )
    assert arr.shape == shape
    assert arr.dtype.numpy_dtype == np.dtype(dtype)
    assert arr.schema.chunk_layout.read_chunk.shape == model.chunks
    assert arr.schema.chunk_layout.write_chunk.shape == model.chunks


@pytest.mark.parametrize("kvstore", ["memory", "file"], indirect=True)
@pytest.mark.parametrize("attrs", [{}, {"foo": "bar"}])
async def test_create_group_v2(kvstore: KVStoreParams, attrs: dict[str, str]) -> None:
    """
    Test that creating a group with attributes works as expected.
    """
    model = pydantic_zarr.zarr_v2.group.LeafGroupSpec(attributes=attrs)
    store = await ts.KvStore.open(kvstore)
    _ = await write_group_v2(model.model_dump(), kvstore=store)
    fetched = await read_group_v2(store)
    assert type(model)(**fetched) == model


def test_get_member_keys() -> None:
    node_keys = (
        V2_GROUP_KEY,
        SEPARATOR.join([b"foo", V2_GROUP_KEY]),
        SEPARATOR.join([b"bar", V2_GROUP_KEY]),
        SEPARATOR.join([b"bar", b"wam", V2_ARRAY_KEY]),
        SEPARATOR.join([b"foo", b"baz", V2_GROUP_KEY]),
    )
    extra = (
        SEPARATOR.join([b"foo", b"bar", b"non_group", b"baz", V2_GROUP_KEY]),
        SEPARATOR.join([b"bar", b"wam", b"sub_array", V2_ARRAY_KEY]),
    )
    assert node_keys == get_member_keys(node_keys + extra)


def test_arrayspec_write() -> None:
    """
    Test that an ArraySpec can be stored using the tensorstore backend
    """
    arr = array.ArraySpec(
        shape=(10, 10),
        chunks=(2, 2),
        dtype=">i2",
        fill_value=0,
        order="C",
        compressor={"id": "gzip", "level": 5},
        filters=None,
    )
    store_spec: StoreSpec = {'engine': 'tensorstore', 'url': 'memory://foo'}
    observed_tstore = arr.persist(store=store_spec).result()
    expected_spec = {
        'driver': 'zarr',
        'dtype': 'int16',
        'kvstore': {'driver': 'memory', 'path': 'foo/'}, 
        'metadata': {
            'chunks': [2, 2],
            'compressor': {'id': 'gzip', 'level': 5},
            'dimension_separator': '/',
            'dtype': '>i2',
            'fill_value': 0,
            'filters': None,
            'order': 'C',
            'shape': [10, 10], 'zarr_format': 2},
            'transform': {
                'input_exclusive_max': [[10], [10]], 'input_inclusive_min': [0, 0]}
                }
    assert observed_tstore.spec().to_json() == expected_spec
