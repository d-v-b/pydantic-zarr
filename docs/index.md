# pydantic-zarr

[![PyPI](https://img.shields.io/pypi/v/pydantic-zarr)](https://pypi.python.org/pypi/pydantic-zarr)

Static typing and runtime validation for Zarr hiearchies.

## Overview

`pydantic-zarr` expresses data stored in the [zarr](https://zarr.readthedocs.io/en/stable/) format with [Pydantic](https://docs.pydantic.dev/1.10/). Specifically, `pydantic-zarr` encodes Zarr groups and arrays as [Pydantic models](https://docs.pydantic.dev/1.10/usage/models/). Programmers can use these models to formalize the structure of Zarr hierarchies, and apply type-checking and runtime validation to Zarr data.


```python
import zarr
from pydantic_zarr.v2 import GroupSpec

group = zarr.group(path='foo')
array = zarr.create(store = group.store, path='foo/bar', shape=10, dtype='uint8')
array.attrs.put({'metadata': 'hello'})

# this is a pydantic model
spec = GroupSpec.from_zarr(group)
print(spec.model_dump())
"""
{
    'zarr_version': 2,
    'attributes': {},
    'members': {
        'bar': {
            'zarr_version': 2,
            'attributes': {'metadata': 'hello'},
            'shape': (10,),
            'chunks': (10,),
            'dtype': '|u1',
            'fill_value': 0,
            'order': 'C',
            'filters': None,
            'dimension_separator': '.',
            'compressor': {
                'id': 'blosc',
                'cname': 'lz4',
                'clevel': 5,
                'shuffle': 1,
                'blocksize': 0,
            },
        }
    },
}
"""
```

Important note: this library only provides tools to represent the *layout* of Zarr groups and arrays, and the structure of their attributes. It performs no type checking or runtime validation of the multidimensional array data contained inside Zarr arrays.

## Installation

`pip install -U pydantic-zarr`

## Design

A Zarr group can be schematized as two elements:


- `attributes`: A dict-like object with string keys and JSON-serializable values.
- `members`: A dict-like object with string keys and values that are other Zarr groups, or Zarr arrays.

A Zarr array can be schematized similarly, but without the `members` property, and with a set of arrays-specific properties like `shape`, `dtype`, etc.

Note the use of the term "schematized": Zarr arrays also represent N-dimensional array data, but `pydantic-zarr` does not treat that data as part of the "schema" of a Zarr array.

Accordingly, in `pydantic-zarr`, Zarr groups are encoded by the `GroupSpec` class with two fields:


- `GroupSpec.attributes`: either a `Mapping` or a `pydantic.BaseModel`.
- `GroupSpec.members`: a mapping with string keys and values that must be `GroupSpec` or `ArraySpec` instances.

Zarr arrays are represented by the `ArraySpec` class, which has a similar `attributes` field, as well as fields for all the Zarr array properties (`dtype`, `shape`, `chunks`, etc).


`GroupSpec` and `ArraySpec` are both [generic models](https://docs.pydantic.dev/1.10/usage/models/#generic-models). `GroupSpec` takes two type parameters, the first specializing the type of `GroupSpec.attributes`, and the second specializing the type of the *values* of `GroupSpec.members` (the keys of `GroupSpec.members` are always strings). `ArraySpec` only takes one type parameter, which specializes the type of `ArraySpec.attributes`.

Examples using this generic typing functionality can be found in the [usage guide](usage.md#using-generic-types).

## Supported Zarr versions

This library supports [version 2](https://zarr.readthedocs.io/en/stable/spec/v2.html) of the Zarr format, with partial support for [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html). Progress towards complete support for Zarr v3 is tracked by [this issue](https://github.com/d-v-b/pydantic-zarr/issues/3).

## Supported Pydantic versions

This library is based on Pydantic version 2.
