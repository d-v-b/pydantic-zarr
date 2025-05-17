from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Iterable,
    Literal,
    Protocol,
    Self,
    Sequence,
    TypeVar,
    cast,
    get_args,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
from pydantic import model_validator
from pydantic.functional_validators import BeforeValidator

from pydantic_zarr._io import Future
from pydantic_zarr.backend import get_storage_engine, zarrify_array_v2
from pydantic_zarr.base import (
    IncEx,
    StrictBase,
    model_like,
)
from pydantic_zarr.zarr_v2.base import TAttr

if TYPE_CHECKING:
    from pydantic_zarr.base import StoreLike
    from pydantic_zarr.types import ArrayLike, ArrayV2Config, TShape


DimensionSeparator = Literal["/", "."]


def stringify_dtype(value: npt.DTypeLike) -> str:
    """
    Convert a `numpy.dtype` object into a `str`.

    Parameters
    ---------
    value: `npt.DTypeLike`
        Some object that can be coerced to a numpy dtype

    Returns
    -------

    A numpy dtype string representation of `value`.
    """
    # TODO: handle string dtypes and structured dtypes
    return np.dtype(value).str  # type: ignore[no-any-return]


DTypeString = Annotated[str, BeforeValidator(stringify_dtype)]


@runtime_checkable
class CodecProtocol(Protocol):
    def get_config(self) -> dict[str, Any]: ...


def dictify_codec(value: dict[str, Any] | CodecProtocol) -> dict[str, Any]:
    """
    Ensure that a `numcodecs.abc.Codec` is converted to a `dict`. If the input is not an
    instance of `numcodecs.abc.Codec`, then it is assumed to be a `dict` with string keys
    and it is returned unaltered.

    Parameters
    ---------

    value : dict[str, Any] | numcodecs.abc.Codec
        The value to be dictified if it is not already a dict.

    Returns
    -------
    dict[str, Any]
        If the input was a `Codec`, then the result of calling `get_config()` on that
        object is returned. This should be a dict with string keys. All other values pass
        through unaltered.
    """
    if isinstance(value, CodecProtocol):
        return value.get_config()
    return value


def parse_dimension_separator(data: object) -> DimensionSeparator:
    """
    Parse the dimension_separator metadata as per the Zarr version 2 specification.
    If the input is `None`, this returns ".".
    If the input is either "." or "/", this returns it.
    Otherwise, raises a ValueError.

    Parameters
    ----------
    data: Any
        The input data to parse.

    Returns
    -------
    Literal["/", "."]
    """
    if data is None:
        return "."
    if data in get_args(DimensionSeparator):
        return cast(DimensionSeparator, data)
    raise ValueError(f'Invalid data, expected one of ("/", ".", None), got {data}')


CodecDict = Annotated[dict[str, Any], BeforeValidator(dictify_codec)]

T = TypeVar("T")


def nullify_empty_list(value: list[T] | None) -> list[T] | None:
    if value is not None and len(value) == 0:
        return None
    return value


class ArrayMetadataSpec(StrictBase):
    zarr_format: Literal[2] = 2
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: DTypeString
    fill_value: int | float | None = 0
    order: Literal["C", "F"] = "C"
    filters: Annotated[list[CodecDict] | None, BeforeValidator(nullify_empty_list)] = None
    dimension_separator: Annotated[
        Literal["/", "."], BeforeValidator(parse_dimension_separator)
    ] = "/"
    compressor: CodecDict | None = None

    @model_validator(mode="after")
    def check_ndim(self) -> Self:
        """
        Check that the `shape` and `chunks` and attributes have the same length.
        """
        if (lshape := len(self.shape)) != (lchunks := len(self.chunks)):
            msg = (
                f"Length of shape must match length of chunks. Got {lshape} elements",
                f"for shape and {lchunks} elements for chunks.",
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_array(
        cls,
        array: ArrayLike[TShape] | ArraySpec[TAttr],
        *,
        chunks: Literal["auto"] | Iterable[int] = "auto",
        fill_value: Literal["auto"] | float | None = "auto",
        order: Literal["auto", "C", "F"] = "auto",
        filters: Literal["auto"] | Iterable[CodecDict] | None = "auto",
        dimension_separator: Literal["auto", "/", "."] = "auto",
        compressor: Literal["auto"] | CodecDict | None = "auto",
        attributes: Literal["auto"] | TAttr = "auto",
    ) -> Self:
        """
        Create an `ArraySpec` from an array-like object. This is a convenience method for when Zarr array will be modelled from an existing array.
        This method takes nearly the same arguments as the `ArraySpec` constructor, minus `shape` and `dtype`, which will be inferred from the `array` argument.
        Additionally, this method accepts the string "auto" as a parameter for all other `ArraySpec` attributes, in which case these attributes will be
        inferred from the `array` argument, with a fallback value equal to the default `ArraySpec` parameters.

        Parameters
        ----------
        array : an array-like object.
            Must have `shape` and `dtype` attributes.
            The `shape` and `dtype` of this object will be used to construct an `ArraySpec`.
        chunks: "auto" | tuple[int, ...], default = "auto"
            The chunks for this `ArraySpec`. If `chunks` is "auto" (the default), then this method first checks if `array` has a `chunksize` attribute, using it if present.
            This supports copying chunk sizes from dask arrays. If `array` does not have `chunksize`, then a routine from `zarr-python` is used to guess the chunk size,
            given the `shape` and `dtype` of `array`. If `chunks` is not auto, then it should be a tuple of ints.
        order: "auto" | "C" | "F", default = "auto"
            The memory order of the `ArraySpec`. One of "auto", "C", or "F". The default is "auto", which means that, if present, `array.order`
            will be used, falling back to "C" if `array` does not have an `order` attribute.
        fill_value: "auto" | int | float | None, default = "auto"
            The fill value for this array. Either "auto" or FillValue. The default is "auto", which means that `array.fill_value` will be used if that attribute exists, with a fallback value of 0.
        compressor: "auto" | CodecDict | None, default = "auto"
            The compressor for this `ArraySpec`. One of "auto", a JSON-serializable representation of a compression codec, or `None`. The default is "auto", which means that `array.compressor` attribute will be used, with a fallback value of `None`.
        filters: "auto" | List[CodecDict] | None, default = "auto"
            The filters for this `ArraySpec`. One of "auto", a list of JSON-serializable representations of compression codec, or `None`. The default is "auto", which means that the `array.filters` attribute will be
            used, with a fallback value of `None`.
        dimension_separator: "auto" | "." | "/", default = "auto"
            Sets the character used for partitioning the different dimensions of a chunk key.
            Must be one of "auto", "/" or ".". The default is "auto", which means that `array.dimension_separator` is used, with a fallback value of "/".
        Returns
        -------
        ArraySpec
            An instance of `ArraySpec` with `shape` and `dtype` attributes derived from `array`.

        Examples
        --------
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> import numpy as np
        >>> x = ArrayMetadataSpec.from_array(np.arange(10))
        >>> x
        ArraySpec(zarr_format=2, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)


        """
        metadata: ArrayV2Config
        if isinstance(array, ArraySpec):
            metadata = array.model_dump()  # type: ignore[assignment]
        else:
            metadata = zarrify_array_v2(array)

        if chunks != "auto":
            metadata["chunks"] = chunks

        if fill_value != "auto":
            metadata["fill_value"] = fill_value

        if compressor != "auto":
            metadata["compressor"] = compressor

        if filters != "auto":
            metadata["filters"] = filters

        if order != "auto":
            metadata["order"] = order

        if dimension_separator != "auto":
            metadata["dimension_separator"] = dimension_separator

        return cls(**metadata)


class ArraySpec(ArrayMetadataSpec, Generic[TAttr]):
    """
    A model of a Zarr Version 2 array metadata document.
    The specification for the data structure being modeled by this class can be found in the
    [Zarr specification](https://zarr.readthedocs.io/en/stable/spec/v2.html#arrays).

    Attributes
    ----------
    zarr_format: Literal[2] = 2
        The Zarr format version of this metadata.
    shape: tuple[int, ...]
        The shape of this array.
    dtype: str
        The data type of this array.
    chunks: Tuple[int, ...]
        The chunk size for this array.
    order: "C" | "F", default = "C"
        The memory order of this array. Must be either "C", which designates "C order",
        AKA lexicographic ordering or "F", which designates "F order", AKA colexicographic ordering.
        The default is "C".
    fill_value: FillValue, default = 0
        The fill value for this array. The default is 0.
    compressor: CodecDict | None
        A JSON-serializable representation of a compression codec, or None. The default is None.
    filters: List[CodecDict] | None, default = None
        A list of JSON-serializable representations of compression codec, or None.
        The default is None.
    dimension_separator: "." | "/", default = "/"
        The character used for partitioning the different dimensions of a chunk key.
        Must be either "/" or ".", or absent, in which case it is interpreted as ".".
        The default is "/".
    """

    attributes: TAttr = {}

    @classmethod
    def from_array(
        cls,
        array: ArrayLike[TShape] | ArraySpec[TAttr],
        *,
        chunks: Literal["auto"] | tuple[int, ...] = "auto",
        fill_value: Literal["auto"] | float | None = "auto",
        order: Literal["auto", "C", "F"] = "auto",
        filters: Literal["auto"] | list[CodecDict] | None = "auto",
        dimension_separator: Literal["auto", "/", "."] = "auto",
        compressor: Literal["auto"] | CodecDict | None = "auto",
        attributes: Literal["auto"] | TAttr = "auto",
    ) -> Self:
        """
        Create an `ArraySpec` from an array-like object. This is a convenience method for when Zarr array will be modelled from an existing array.
        This method takes nearly the same arguments as the `ArraySpec` constructor, minus `shape` and `dtype`, which will be inferred from the `array` argument.
        Additionally, this method accepts the string "auto" as a parameter for all other `ArraySpec` attributes, in which case these attributes will be
        inferred from the `array` argument, with a fallback value equal to the default `ArraySpec` parameters.

        Parameters
        ----------
        array : an array-like object.
            Must have `shape` and `dtype` attributes.
            The `shape` and `dtype` of this object will be used to construct an `ArraySpec`.
        chunks: "auto" | tuple[int, ...], default = "auto"
            The chunks for this `ArraySpec`. If `chunks` is "auto" (the default), then this method first checks if `array` has a `chunksize` attribute, using it if present.
            This supports copying chunk sizes from dask arrays. If `array` does not have `chunksize`, then a routine from `zarr-python` is used to guess the chunk size,
            given the `shape` and `dtype` of `array`. If `chunks` is not auto, then it should be a tuple of ints.
        order: "auto" | "C" | "F", default = "auto"
            The memory order of the `ArraySpec`. One of "auto", "C", or "F". The default is "auto", which means that, if present, `array.order`
            will be used, falling back to "C" if `array` does not have an `order` attribute.
        fill_value: "auto" | int | float | None, default = "auto"
            The fill value for this array. Either "auto" or FillValue. The default is "auto", which means that `array.fill_value` will be used if that attribute exists, with a fallback value of 0.
        compressor: "auto" | CodecDict | None, default = "auto"
            The compressor for this `ArraySpec`. One of "auto", a JSON-serializable representation of a compression codec, or `None`. The default is "auto", which means that `array.compressor` attribute will be used, with a fallback value of `None`.
        filters: "auto" | List[CodecDict] | None, default = "auto"
            The filters for this `ArraySpec`. One of "auto", a list of JSON-serializable representations of compression codec, or `None`. The default is "auto", which means that the `array.filters` attribute will be
            used, with a fallback value of `None`.
        dimension_separator: "auto" | "." | "/", default = "auto"
            Sets the character used for partitioning the different dimensions of a chunk key.
            Must be one of "auto", "/" or ".". The default is "auto", which means that `array.dimension_separator` is used, with a fallback value of "/".
        Returns
        -------
        ArraySpec
            An instance of `ArraySpec` with `shape` and `dtype` attributes derived from `array`.

        Examples
        --------
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> import numpy as np
        >>> x = ArrayMetadataSpec.from_array(np.arange(10))
        >>> x
        ArraySpec(zarr_format=2, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)


        """
        metadata: ArrayV2Config
        if isinstance(array, ArraySpec):
            metadata = array.model_dump()  # type: ignore[assignment]
        else:
            metadata = zarrify_array_v2(array)

        if chunks != "auto":
            metadata["chunks"] = chunks

        if fill_value != "auto":
            metadata["fill_value"] = fill_value

        if compressor != "auto":
            metadata["compressor"] = compressor

        if filters != "auto":
            metadata["filters"] = filters

        if order != "auto":
            metadata["order"] = order

        if dimension_separator != "auto":
            metadata["dimension_separator"] = dimension_separator

        if attributes != "auto":
            metadata["attributes"] = attributes

        return cls(**metadata)

    def persist(
        self,
        store: StoreLike,
        *,
        path: str = '',
        overwrite: bool = False,
    ) -> Future[ArrayLike[tuple[int, ...]]]:
        """
        Serialize an `ArraySpec` to a Zarr array at a specific path in a Zarr store. This operation
        will create metadata documents in the store, but will not write any chunks.

        Parameters
        ----------
        store : StoreLike
            A specification of the storage backend that will manifest the array.
        path : str
            The location of the array inside the store.
        overwrite: bool, default = False
            Whether to overwrite existing objects in storage to create the Zarr array.
        **kwargs : Any
            Additional keyword arguments are passed to `zarr.create`.
        Returns
        -------
        ArrayProxy
            A wrapper around a zarr array that is structurally identical to `self`.
        """
        # get the appropriate storage backend
        resolved_store = get_storage_engine(store)
        return Future(resolved_store.write_array(
            path=path,
            metadata=self.model_dump()))

    def like(
        self,
        other: ArrayLike,
        *,
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare am `ArraySpec` to another `ArraySpec` or a `zarr.Array`, parameterized over the
        fields to exclude or include in the comparison. Models are first converted to `dict` via the
        `model_dump` method of `pydantic.BaseModel`, then compared with the `==` operator.

        Parameters
        ----------
        other: ArraySpec | zarr.Array
            The array (model or actual) to compare with. If other is a `zarr.Array`, it will be
            converted to `ArraySpec` first.
        include: IncEx, default = None
            A specification of fields to include in the comparison. The default value is `None`,
            which means that all fields will be included. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.
        exclude: IncEx, default = None
            A specification of fields to exclude from the comparison. The default value is `None`,
            which means that no fields will be excluded. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.

        Returns
        -------
        bool
            `True` if the two models have identical fields, `False` otherwise, given
            the set of fields specified by the `include` and `exclude` keyword arguments.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v2 import ArraySpec
        >>> x = zarr.create((10,10))
        >>> x.attrs.put({'foo': 10})
        >>> x_model = ArraySpec.from_zarr(x)
        >>> print(x_model.like(x_model)) # it is like itself.
        True
        >>> print(x_model.like(x))
        True
        >>> y = zarr.create((10,10))
        >>> y.attrs.put({'foo': 11}) # x and y are the same, other than their attrs
        >>> print(x_model.like(y))
        False
        >>> print(x_model.like(y, exclude={'attributes'}))
        True
        """
        other_parsed: ArraySpec[Any]
        if not isinstance(other, ArraySpec):
            other_parsed = ArraySpec.from_array(other)
        else:
            other_parsed = other

        return model_like(self, other_parsed, include=include, exclude=exclude)


