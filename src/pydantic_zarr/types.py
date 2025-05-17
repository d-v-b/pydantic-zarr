from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, NotRequired, Protocol, TypeVar, runtime_checkable

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping


class StoreSpec(TypedDict):
    url: str
    engine: NotRequired[Literal["tensorstore", "zarr-python"]]
    configuration: NotRequired[Mapping[str, Any]]

TShape = TypeVar("TShape", bound=tuple[int, ...])

@runtime_checkable
class ArrayLike(Protocol, Generic[TShape]):
    # shape has to be annotated with a generic type parameter for numpy
    shape: TShape
    dtype: Any


class GroupLike(Protocol):
    @property
    def attrs(self) -> Mapping[str, Any]: ...

    def members(
        self, max_depth: int | None
    ) -> tuple[tuple[str, ArrayLike[Any] | GroupLike], ...]: ...


class CodecConfigV2(TypedDict, total=False):
    id: str


class ArrayMetadataV2Config(TypedDict):
    zarr_format: Literal[2]
    shape: tuple[int, ...]
    dtype: str | tuple[object, ...]
    chunks: tuple[int, ...]
    fill_value: object
    order: Literal["C", "F"]
    compressor: CodecConfigV2 | None
    filters: tuple[CodecConfigV2, ...] | None
    dimension_separator: Literal["/", "."]


class ArrayV2Config(ArrayMetadataV2Config):
    attributes: NotRequired[Mapping[str, object]]


class GroupMetadataV2Config(TypedDict):
    zarr_format: Literal[2]


class GroupV2Config(GroupMetadataV2Config):
    attributes: NotRequired[Mapping[str, object]]
    members: NotRequired[Mapping[str, ArrayV2Config | GroupV2Config]]


class NamedConfig(TypedDict):
    name: str
    configuration: NotRequired[Mapping[str, object]]


class ArrayV3Config(TypedDict):
    zarr_format: Literal[3]
    node_type: Literal["array"]
    shape: tuple[int, ...]
    data_type: str | NamedConfig
    fill_value: object
    chunk_grid: NamedConfig
    chunk_key_encoding: NamedConfig
    codecs: tuple[NamedConfig, ...]
    attributes: Mapping[str, object]
    dimension_names: tuple[str, ...] | None


class GroupV3Config(TypedDict):
    zarr_format: Literal[3]
    node_type: Literal["group"]
    attributes: Mapping[str, object]


class RegularChunks(TypedDict):
    read_shape: tuple[int, ...]
    write_shape: tuple[int, ...] | None


class RectilinearChunks(TypedDict):
    read_shape: tuple[tuple[int, ...], ...]
    write_shape: tuple[tuple[int, ...], ...] | None