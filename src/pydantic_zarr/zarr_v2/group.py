from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Generic, Literal, Self, TypeVar, Union

from pydantic import AfterValidator

from pydantic_zarr.base import StrictBase, ensure_key_no_path, model_like
from pydantic_zarr.types import GroupLike
from pydantic_zarr.zarr_v2.array import ArraySpec
from pydantic_zarr.zarr_v2.base import TAttr

BaseMember = Union["GroupSpec[TAttr, TMember]", "ArraySpec[TAttr]"]
TMember = TypeVar("TMember", bound=BaseMember)

class LeafGroupSpec(StrictBase, Generic[TAttr]):
    """
    A Zarr group with no members.
    """

    zarr_format: Literal[2] = 2
    attributes: TAttr = {}

class GroupSpec(LeafGroupSpec[TAttr], Generic[TAttr, TMember]):
    """
    A model of a Zarr Version 2 Group with members.
    The specification for the data structure being modeled by this
    class can be found in the
    [Zarr specification](https://zarr.readthedocs.io/en/stable/spec/v2.html#groups).

    Attributes
    ----------
    attributes: TAttr, default = {}
        The user-defined attributes of this group. Should be JSON-serializable.
    members: dict[str, TItem], default = {}
        The members of this group. `members` may be `None`, which models the condition
        where the members are unknown, e.g., because they have not been discovered yet.
        If `members` is not s`None`, then it must be a dict with string keys and values that
        are either `ArraySpec` or `GroupSpec`.
    """

    members: Annotated[Mapping[str, TMember], AfterValidator(ensure_key_no_path)] = {}

    @classmethod
    def from_grouplike(cls, group: GroupLike) -> Self:
        """
        Create a `GroupSpec` from a group-like object
        """
        members: dict[str, ArraySpec[Any] | GroupSpec[Any, Any]] = {}
        for name, member in group.members(max_depth=None):
            if hasattr(member, "shape"):
                members[name] = ArraySpec.from_array(member)  # type: ignore[arg-type]
            elif hasattr(member, "members"):
                members[name] = GroupSpec.from_grouplike(member)
        return cls(attributes=group.attrs, members=members)

    def like(
        self,
        other: GroupLike | GroupSpec[Any, Any],
        include: IncEx = None,
        exclude: IncEx = None,
    ) -> bool:
        """
        Compare a `GroupSpec` to another `GroupSpec` or a `zarr.Group`, parameterized over the
        fields to exclude or include in the comparison. Models are first converted to dict via the
        `model_dump` method of `pydantic.BaseModel`, then compared with the `==` operator.

        Parameters
        ----------
        other: GroupSpec | zarr.Group
            The group (model or actual) to compare with. If other is a `zarr.Group`, it will be
            converted to a `GroupSpec`.
        include: IncEx, default = None
            A specification of fields to include in the comparison. The default is `None`,
            which means that all fields will be included. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.
        exclude: IncEx, default = None
            A specification of fields to exclude from the comparison. The default is `None`,
            which means that no fields will be excluded. See the documentation of
            `pydantic.BaseModel.model_dump` for more details.

        Returns
        -------
        bool
            `True` if the two models have identical fields, `False` otherwise.

        Examples
        --------
        >>> import zarr
        >>> from pydantic_zarr.v2 import GroupSpec
        >>> import numpy as np
        >>> z1 = zarr.group(path='z1')
        >>> z1a = z1.array(name='foo', data=np.arange(10))
        >>> z1_model = GroupSpec.from_zarr(z1)
        >>> print(z1_model.like(z1_model)) # it is like itself
        True
        >>> print(z1_model.like(z1))
        True
        >>> z2 = zarr.group(path='z2')
        >>> z2a = z2.array(name='foo', data=np.arange(10))
        >>> print(z1_model.like(z2))
        True
        >>> z2.attrs.put({'foo' : 100}) # now they have different attributes
        >>> print(z1_model.like(z2))
        False
        >>> print(z1_model.like(z2, exclude={'attributes'}))
        True
        """

        other_parsed: GroupSpec[Any, Any]
        if isinstance(other, GroupSpec):
            other_parsed = other
        else:
            other_parsed = GroupSpec.from_grouplike(other)

        return model_like(self, other_parsed, include=include, exclude=exclude)

    def to_flat(self, root_path: str = "") -> dict[str, GroupSpec[Any, Any] | ArraySpec[Any]]:
        """
        Flatten this `GroupSpec`.
        This method returns a `dict` with string keys and values that are `GroupSpec` or
        `ArraySpec`.

        Then the resulting `dict` will contain a copy of the input with a null `members` attribute
        under the key `root_path`, as well as copies of the result of calling `node.to_flat` on each
        element of `node.members`, each under a key created by joining `root_path` with a '/`
        character to the name of each member, and so on recursively for each sub-member.

        Parameters
        ---------
        root_path: `str`, default = ''.
            The root path. The keys in `self.members` will be
            made relative to `root_path` when used as keys in the result dictionary.

        Returns
        -------
        Dict[str, ArraySpec | GroupSpec]
            A flattened representation of the hierarchy.

        Examples
        --------

        >>> from pydantic_zarr.v2 import to_flat, GroupSpec
        >>> g1 = GroupSpec(members=None, attributes={'foo': 'bar'})
        >>> to_flat(g1)
        {'': GroupSpec(zarr_version=2, attributes={'foo': 'bar'}, members=None)}
        >>> to_flat(g1 root_path='baz')
        {'baz': GroupSpec(zarr_version=2, attributes={'foo': 'bar'}, members=None)}
        >>> to_flat(GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}))
        {'/g1': GroupSpec(zarr_version=2, attributes={'foo': 'bar'}, members=None), '': GroupSpec(zarr_version=2, attributes={'foo': 'bar'}, members=None)}
        """
        return to_flat(self, root_path=root_path)

    @classmethod
    def from_flat(cls, data: dict[str, ArraySpec[Any] | GroupSpec[Any]]) -> Self:
        """
        Create a `GroupSpec` from a flat hierarchy representation. The flattened hierarchy is a
        `dict` with the following constraints: keys must be valid paths; values must
        be `ArraySpec` or `GroupSpec` instances.

        Parameters
        ----------
        data: Dict[str, ArraySpec | GroupSpec]
            A flattened representation of a Zarr hierarchy.

        Returns
        -------
        GroupSpec
            A `GroupSpec` representation of the hierarchy.

        Examples
        --------
        >>> from pydantic_zarr.v2 import GroupSpec, ArraySpec
        >>> import numpy as np
        >>> flat = {'': GroupSpec(attributes={'foo': 10}, members=None)}
        >>> GroupSpec.from_flat(flat)
        GroupSpec(zarr_version=2, attributes={'foo': 10}, members={})
        >>> flat = {
            '': GroupSpec(attributes={'foo': 10}, members=None),
            '/a': ArraySpec.from_array(np.arange(10))}
        >>> GroupSpec.from_flat(flat)
        GroupSpec(zarr_version=2, attributes={'foo': 10}, members={'a': ArraySpec(zarr_version=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
        """
        from_flated = from_flat_group(data)
        return cls(**from_flated.model_dump())


def to_flat(
    node: ArraySpec[TAttr] | GroupSpec[TAttr, TMember], root_path: str = ""
) -> dict[str, ArraySpec[TAttr] | GroupSpec[TAttr, TMember]]:
    """
    Flatten a `GroupSpec` or `ArraySpec`.
    Converts a `GroupSpec` or `ArraySpec` and a string, into a `dict` with string keys and
    values that are `GroupSpec` or `ArraySpec`.

    If the input is an `ArraySpec`, then this function just returns the dict `{root_path: node}`.

    If the input is a `GroupSpec`, then the resulting `dict` will contain a copy of the input
    with a null `members` attribute under the key `root_path`, as well as copies of the result of
    calling `flatten_node` on each element of `node.members`, each under a key created by joining
    `root_path` with a '/` character to the name of each member, and so on recursively for each
    sub-member.

    Parameters
    ---------
    node: `GroupSpec` | `ArraySpec`
        The node to flatten.
    root_path: `str`, default = ''.
        The root path. If `node` is a `GroupSpec`, then the keys in `node.members` will be
        made relative to `root_path` when used as keys in the result dictionary.

    Returns
    -------
    Dict[str, ArraySpec | GroupSpec]
        A flattened representation of the hierarchy.

    Examples
    --------

    >>> from pydantic_zarr.v2 import flatten, GroupSpec
    >>> g1 = GroupSpec(members=None, attributes={'foo': 'bar'})
    >>> to_flat(g1)
    {'': GroupSpec(zarr_version=2, attributes={'foo': 'bar'}, members=None)}
    >>> to_flat(g1 root_path='baz')
    {'baz': GroupSpec(zarr_version=2, attributes={'foo': 'bar'}, members=None)}
    >>> to_flat(GroupSpec(members={'g1': g1}, attributes={'foo': 'bar'}))
    {'/g1': GroupSpec(zarr_version=2, attributes={'foo': 'bar'}, members=None), '': GroupSpec(zarr_version=2, attributes={'foo': 'bar'}, members=None)}
    """
    result = {}
    model_copy: ArraySpec[TAttr] | GroupSpec[TAttr, TMember]
    if isinstance(node, ArraySpec):
        model_copy = node.model_copy(deep=True)
    else:
        model_copy = node.model_copy(deep=True, update={"members": None})
        if node.members is not None:
            for name, value in node.members.items():
                result.update(to_flat(value, "/".join([root_path, name])))

    result[root_path] = model_copy
    # sort by increasing key length
    result_sorted_keys = dict(sorted(result.items(), key=lambda v: len(v[0])))
    return result_sorted_keys


def from_flat_group(data: dict[str, ArraySpec | GroupSpec]) -> GroupSpec:
    """
    Generate a `GroupSpec` from a flat representation of a hierarchy, i.e. a `dict` with
    string keys (paths) and `ArraySpec` / `GroupSpec` values (nodes).

    Parameters
    ----------
    data: Dict[str, ArraySpec | GroupSpec]
        A flat representation of a Zarr hierarchy rooted at a Zarr group.

    Returns
    -------
    GroupSpec
        A `GroupSpec` that represents the hierarchy described by `data`.

    Examples
    --------
    >>> from pydantic_zarr.v2 import from_flat_group, GroupSpec, ArraySpec
    >>> import numpy as np
    >>> tree = {'/foo': ArraySpec.from_array(np.arange(10))}
    >>> from_flat_group(tree) # note that an implicit Group is created
    GroupSpec(zarr_version=2, attributes={}, members={'foo': ArraySpec(zarr_version=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
    """
    root_name = ""
    sep = "/"
    # arrays that will be members of the returned GroupSpec
    member_arrays: dict[str, ArraySpec] = {}
    # groups, and their members, that will be members of the returned GroupSpec.
    # this dict is populated by recursively applying `from_flat_group` function.
    member_groups: dict[str, GroupSpec] = {}
    # this dict collects the arrayspecs and groupspecs that belong to one of the members of the
    # groupspecs we are constructing. They will later be aggregated in a recursive step that
    # populates member_groups
    submember_by_parent_name: dict[str, dict[str, ArraySpec | GroupSpec]] = {}
    # copy the input to ensure that mutations are contained inside this function
    data_copy = data.copy()
    # Get the root node
    try:
        # The root node is a GroupSpec with the key ""
        root_node = data_copy.pop(root_name)
        if isinstance(root_node, ArraySpec):
            raise ValueError("Got an ArraySpec as the root node. This is invalid.")
    except KeyError:
        # If a root node was not found, create a default one
        root_node = GroupSpec(attributes={}, members=None)

    # partition the tree (sans root node) into 2 categories: (arrays, groups + their members).
    for key, value in data_copy.items():
        key_parts = key.split(sep)
        if key_parts[0] != root_name:
            raise ValueError(f'Invalid path: {key} does not start with "{root_name}{sep}".')

        subparent_name = key_parts[1]
        if len(key_parts) == 2:
            # this is an array or group that belongs to the group we are ultimately returning
            if isinstance(value, ArraySpec):
                member_arrays[subparent_name] = value
            else:
                if subparent_name not in submember_by_parent_name:
                    submember_by_parent_name[subparent_name] = {}
                submember_by_parent_name[subparent_name][root_name] = value
        else:
            # these are groups or arrays that belong to one of the member groups
            # not great that we repeat this conditional dict initialization
            if subparent_name not in submember_by_parent_name:
                submember_by_parent_name[subparent_name] = {}
            submember_by_parent_name[subparent_name][sep.join([root_name, *key_parts[2:]])] = value

    # recurse
    for subparent_name, submemb in submember_by_parent_name.items():
        member_groups[subparent_name] = from_flat_group(submemb)

    return GroupSpec(members={**member_groups, **member_arrays}, attributes=root_node.attributes)





def from_flat(data: dict[str, ArraySpec | GroupSpec]) -> ArraySpec | GroupSpec:
    """
    Wraps `from_flat_group`, handling the special case where a Zarr array is defined at the root of
    a hierarchy and thus is not contained by a Zarr group.

    Parameters
    ----------

    data: Dict[str, ArraySpec | GroupSpec]
        A flat representation of a Zarr hierarchy. This is a `dict` with keys that are strings,
        and values that are either `GroupSpec` or `ArraySpec` instances.

    Returns
    -------
    ArraySpec | GroupSpec
        The `ArraySpec` or `GroupSpec` representation of the input data.

    Examples
    --------
    >>> from pydantic_zarr.v2 import from_flat, GroupSpec, ArraySpec
    >>> import numpy as np
    >>> tree = {'': ArraySpec.from_array(np.arange(10))}
    >>> from_flat(tree) # special case of a Zarr array at the root of the hierarchy
    ArraySpec(zarr_version=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)
    >>> tree = {'/foo': ArraySpec.from_array(np.arange(10))}
    >>> from_flat(tree) # note that an implicit Group is created
    GroupSpec(zarr_version=2, attributes={}, members={'foo': ArraySpec(zarr_version=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
    """

    # minimal check that the keys are valid
    invalid_keys = []
    for key in data:
        if key.endswith("/"):
            invalid_keys.append(key)
    if len(invalid_keys) > 0:
        msg = f'Invalid keys {invalid_keys} found in data. Keys may not end with the "/"" character'
        raise ValueError(msg)

    if tuple(data.keys()) == ("",) and isinstance(tuple(data.values())[0], ArraySpec):
        return tuple(data.values())[0]
    else:
        return from_flat_group(data)

