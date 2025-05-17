from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from pydantic import AnyUrl, BaseModel, Field
from pydantic_zarr.types import ArrayV2Config


DataType = Literal[
    "bool",
    "char",
    "byte",
    "int4",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e4m3b11fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "string",
    "ustring",
    "json",
]


class IndexTransformParams(TypedDict):
    input_rank: int
    input_exclusive_min: int | tuple[int, ...]
    input_exclusive_max: int | tuple[int, ...]
    input_shape: int | tuple[int, ...]
    input_labels: tuple[str, ...]
    output: tuple[OutputIndexMap, ...]


class OutputIndexMap(TypedDict):
    input_dimension: int
    output_dimension: int
    strides: tuple[int, ...]
    offset: int
    size: int
    broadcast_sizes: tuple[int, ...] | None
    broadcast_dimensions: tuple[int, ...] | None


KVStoreDriver = Literal["memory", "file"]


class KVStoreParams(TypedDict):
    driver: KVStoreDriver
    path: str
    context: NotRequired[dict[str, object]]


class FileDriverParams(KVStoreParams):
    driver: Literal["file"]


class MemoryDriverParams(KVStoreParams):
    driver: Literal["memory"]


class ZarrDriverParams(TypedDict):
    driver: Literal["zarr"]
    kvstore: str | KVStoreParams
    context: NotRequired[dict[str, Any]]
    dtype: NotRequired[DataType]
    rank: NotRequired[int]
    transform: NotRequired[IndexTransformParams]
    open: bool
    create: bool
    delete_existing: bool
    assume_metadata: bool
    assume_cached_metadata: bool
    metadata: ArrayV2Config

