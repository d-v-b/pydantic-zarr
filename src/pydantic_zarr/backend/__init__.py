from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack

from pydantic_zarr.types import ArrayLike, ArrayV2Config, StoreSpec

if TYPE_CHECKING:
    from typing import Literal

    from pydantic_zarr._io import ZarrV2Writer

from ._tensorstore._import import TENSORSTORE_INSTALLED
from .zarr_python._import import ZARR_PYTHON_INSTALLED

zarr_writers: dict[Literal['zarr-python', 'tensorstore'], type[ZarrV2Writer]] = {}
if TENSORSTORE_INSTALLED:
    from pydantic_zarr.backend._tensorstore import TensorstoreZarrV2Writer
    zarr_writers["tensorstore"] = TensorstoreZarrV2Writer
if ZARR_PYTHON_INSTALLED:
    from pydantic_zarr.backend.zarr_python import ZarrPythonZarrV2Writer
    zarr_writers["zarr-python"] = ZarrPythonZarrV2Writer

def zarrify_array_v2(array: object) -> ArrayV2Config:
    if ZARR_PYTHON_INSTALLED:
        import zarr

        if isinstance(array, zarr.Array):
            from pydantic_zarr.backend.zarr_python import zarrify as pyz_zarrify

            return pyz_zarrify(array, zarr_format=2)
        else:
            pass

    if TENSORSTORE_INSTALLED:
        import tensorstore

        if isinstance(array, tensorstore.TensorStore):
            from pydantic_zarr.backend._tensorstore import zarrify as ts_zarrify

            return ts_zarrify(array, zarr_format=2)
        else:
            pass

    if isinstance(array, ArrayLike):
        from pydantic_zarr.backend.numpy_like import zarrify as np_zarrify

        return np_zarrify(array, zarr_format=2)
    else:
        raise ValueError(f"Unsupported array type: {type(array)}")

def get_storage_engine(store: StoreSpec | Any) -> ZarrV2Writer:
    """
    Get the storage engine from the store spec.
    """
    engine_name: Literal['tensorstore', 'zarr-python']
    if zarr_writers == {}:
        raise ValueError("No storage engine available. Please install either zarr-python or tensorstore.")
    if isinstance(store, dict):
        engine_param = store.get('engine')
        if engine_param is None:
            if 'zarr-python' in zarr_writers:
                engine_name = 'zarr-python'
            elif 'tensorstore' in zarr_writers:
                engine_name = 'tensorstore'
            else:
                raise ValueError("No storage engine specified and no default available.")
        else:
            if engine_param not in zarr_writers:
                raise ValueError(f"Unsupported storage engine: {engine_param}. Available engines: {list(zarr_writers.keys())}")
            else:
                engine_name = engine_param

        writer = zarr_writers[engine_name].from_url(store['url'])
    else:
        if ZARR_PYTHON_INSTALLED and isinstance(store, ZarrPythonZarrV2Writer):
    return writer