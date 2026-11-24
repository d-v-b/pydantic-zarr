"""
Tests for fields that a subclass adds to a node spec.

Such a field is written as a top-level member of the node's metadata
document, so it is an extra field of the document. It must follow the rules
for extra fields, and it is kept when a hierarchy is flattened, written and
read back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired

import numpy as np
import pytest
from pydantic import ConfigDict, Field, ValidationError, computed_field
from typing_extensions import TypedDict
from zarr_metadata import JSONValue  # noqa: TC002 (pydantic resolves field annotations at runtime)
from zarr_metadata.v3.definition import (
    ArrayExtraFieldDefinition,
    GroupExtraFieldDefinition,
    ZarrV3Context,
)

from pydantic_zarr import v2, v3
from pydantic_zarr._metadata import CORE_CONTEXT

if TYPE_CHECKING:
    from collections.abc import Mapping


class Label(TypedDict):
    """A Zarr V3 extra field that a reader may ignore."""

    must_understand: Literal[False]
    text: str


DEFAULT_LABEL: Label = {"must_understand": False, "text": "none"}


def _label(text: str) -> Label:
    return {"must_understand": False, "text": text}


class TaggedArray(v3.ArraySpec):
    tag: Label = DEFAULT_LABEL

    @computed_field
    @property
    def shape_label(self) -> Label:
        return _label(f"{len(self.shape)}d")


class RequiredLeaf(v3.GroupSpec):
    label: Label


class MidMembers(TypedDict, closed=True):
    leaf: RequiredLeaf
    tagged: TaggedArray
    untagged: TaggedArray


class Mid(v3.GroupSpec[dict[str, object], MidMembers]):
    model_config = ConfigDict(serialize_by_alias=True, validate_by_name=True)
    label: Label = DEFAULT_LABEL
    aliased: Label = Field(default=DEFAULT_LABEL, alias="the-alias")


class RootMembers(TypedDict, closed=True):
    changed: Mid
    unchanged: Mid


class Root(v3.GroupSpec[dict[str, object], RootMembers]):
    label: Label = DEFAULT_LABEL


class BaseLabeled(v3.BaseGroupSpec):
    label: Label = DEFAULT_LABEL


class Acme(TypedDict, closed=True):
    level: int
    must_understand: NotRequired[bool]


ACME_SCOPE = CORE_CONTEXT.union(
    ZarrV3Context(
        group_extra_fields=[GroupExtraFieldDefinition(name="acme", configuration=Acme)],
        array_extra_fields=[ArrayExtraFieldDefinition(name="acme", configuration=Acme)],
    )
)
"""A scope that claims an extra field `acme`, which a reader must understand."""


class AcmeArray(v3.ArraySpec):
    zarr_scope = ACME_SCOPE
    acme: Acme = Field(default_factory=lambda: {"level": 1, "must_understand": True})


class AcmeMembers(TypedDict, closed=True):
    array: AcmeArray


class AcmeGroup(v3.GroupSpec[dict[str, object], AcmeMembers]):
    zarr_scope = ACME_SCOPE
    acme: Acme


class AcmeRootMembers(TypedDict, closed=True):
    group: AcmeGroup


class AcmeRoot(v3.GroupSpec[dict[str, object], AcmeRootMembers]):
    zarr_scope = ACME_SCOPE


class TaggedArrayV2(v2.ArraySpec):
    tag: str = "none"
    count: int


class SubMembersV2(TypedDict, closed=True):
    tagged: TaggedArrayV2
    untagged: TaggedArrayV2


class SubV2(v2.GroupSpec[dict[str, object], SubMembersV2]):
    pass


class RootMembersV2(TypedDict, closed=True):
    sub: SubV2
    top: TaggedArrayV2


class RootV2(v2.GroupSpec[dict[str, object], RootMembersV2]):
    pass


def _v3_hierarchy() -> Root:
    array = v3.ArraySpec.from_array(np.arange(4)).model_dump()

    def mid(label: Label) -> Mid:
        return Mid(
            label=label,
            aliased=label,
            members={
                "leaf": RequiredLeaf(label=_label("required")),
                "tagged": TaggedArray(**array, tag=_label("tagged")),
                "untagged": TaggedArray(**array),
            },
        )

    return Root(members={"changed": mid(_label("changed")), "unchanged": mid(DEFAULT_LABEL)})


def _acme_hierarchy() -> AcmeRoot:
    array = v3.ArraySpec.from_array(np.arange(4)).model_dump()
    return AcmeRoot(
        members={
            "group": AcmeGroup(
                acme={"level": 2, "must_understand": True},
                members={"array": AcmeArray(**array, acme={"level": 3})},
            )
        }
    )


def _v2_hierarchy() -> RootV2:
    array = v2.ArraySpec.from_array(np.arange(4)).model_dump()
    return RootV2(
        members={
            "sub": SubV2(
                members={
                    "tagged": TaggedArrayV2(**array, tag="tagged", count=1),
                    "untagged": TaggedArrayV2(**array, count=2),
                }
            ),
            "top": TaggedArrayV2(**array, tag="top", count=3),
        }
    )


@pytest.mark.parametrize(
    ("hierarchy", "expected_documents"),
    [
        (
            _v3_hierarchy(),
            {
                "zarr.json": {"label": DEFAULT_LABEL},
                "changed/zarr.json": {"label": _label("changed"), "the-alias": _label("changed")},
                "changed/leaf/zarr.json": {"label": _label("required")},
                "changed/tagged/zarr.json": {"tag": _label("tagged"), "shape_label": _label("1d")},
                "changed/untagged/zarr.json": {"tag": DEFAULT_LABEL},
                "unchanged/zarr.json": {"label": DEFAULT_LABEL, "the-alias": DEFAULT_LABEL},
                "unchanged/leaf/zarr.json": {"label": _label("required")},
            },
        ),
        (
            _acme_hierarchy(),
            {
                "group/zarr.json": {"acme": {"level": 2, "must_understand": True}},
                "group/array/zarr.json": {"acme": {"level": 3}},
            },
        ),
        (
            _v2_hierarchy(),
            {
                "sub/tagged/.zarray": {"tag": "tagged", "count": 1},
                "sub/untagged/.zarray": {"tag": "none", "count": 2},
                "top/.zarray": {"tag": "top", "count": 3},
            },
        ),
    ],
    ids=["v3", "v3-scoped", "v2"],
)
def test_added_fields_round_trip(
    hierarchy: Root | AcmeRoot | RootV2, expected_documents: dict[str, dict[str, JSONValue]]
) -> None:
    """
    Every stored document holds the fields that its node's class adds, and
    the hierarchy reads back equal, through its stored metadata, through
    `to_flat` and `from_flat`, and through `replace`. The fields are at their
    defaults or not, required or not, aliased or computed, and claimed by the
    class's scope or not, at every level.
    """
    stored = hierarchy.to_stored_metadata_json()
    for key, fields in expected_documents.items():
        document: Mapping[str, object] = stored[key]
        assert {name: document[name] for name in fields} == fields, key
    assert type(hierarchy).from_stored_metadata_json(stored) == hierarchy
    if isinstance(hierarchy, Root):
        assert Root.from_flat(hierarchy.to_flat()) == hierarchy
    elif isinstance(hierarchy, AcmeRoot):
        assert AcmeRoot.from_flat(hierarchy.to_flat()) == hierarchy
    else:
        assert RootV2.from_flat(hierarchy.to_flat()) == hierarchy
    replaced = hierarchy.replace(attributes={"changed": True}).to_stored_metadata_json()
    members = {key: document for key, document in stored.items() if "/" in key}
    assert {key: document for key, document in replaced.items() if "/" in key} == members


def test_member_typed_as_a_base_class_keeps_added_fields() -> None:
    """
    A plain `GroupSpec` whose member is a subclass instance dumps the member
    by its own class. So `replace` keeps the fields the member's class adds,
    and `from_flat` and `from_stored_metadata_json` read them back as extra
    fields of a plain member.
    """
    array = v3.ArraySpec.from_array(np.arange(4)).model_dump()
    member = TaggedArray(**array, tag=_label("kept"))
    group = v3.GroupSpec(members={"x": member})
    assert group.model_dump()["members"]["x"]["tag"] == _label("kept")
    replaced = group.replace(attributes={"a": 1})
    stored = group.to_stored_metadata_json()
    assert replaced.to_stored_metadata_json()["x/zarr.json"] == stored["x/zarr.json"]
    for read in (
        v3.GroupSpec.from_stored_metadata_json(stored),
        v3.GroupSpec.from_flat(group.to_flat()),
    ):
        assert read.members["x"].extra_fields == {
            "tag": _label("kept"),
            "shape_label": _label("1d"),
        }


def test_flat_groups_hold_added_fields_as_extra_fields() -> None:
    """
    A plain `BaseGroupSpec` holds a field that a subclass adds in its
    `extra_fields`, both in `to_flat` and in `own_metadata`. Validating its
    dump as the subclass reads the field back.
    """
    group = _v3_hierarchy()
    flat = group.to_flat()
    assert flat["/changed"] == v3.BaseGroupSpec(
        extra_fields={"label": _label("changed"), "the-alias": _label("changed")}
    )
    assert flat["/changed/leaf"] == v3.BaseGroupSpec(extra_fields={"label": _label("required")})
    assert type(flat["/changed/tagged"]) is TaggedArray
    root = group.own_metadata()
    assert root == v3.BaseGroupSpec(extra_fields={"label": DEFAULT_LABEL})
    assert BaseLabeled.model_validate(root.model_dump()) == BaseLabeled()
    labeled = BaseLabeled(label=_label("own"))
    assert v3.BaseGroupSpec.from_stored_metadata_json(labeled.to_stored_metadata_json()) == (
        v3.BaseGroupSpec(extra_fields={"label": _label("own")})
    )
    assert BaseLabeled.from_stored_metadata_json(labeled.to_stored_metadata_json()) == labeled


@pytest.mark.parametrize(
    ("spec_class", "fields"),
    [
        (v3.GroupSpec, {}),
        (v3.ArraySpec, v3.ArraySpec.from_array(np.arange(2)).model_dump()),
        (v2.GroupSpec, {}),
        (v2.ArraySpec, v2.ArraySpec.from_array(np.arange(2)).model_dump()),
    ],
    ids=["v3-group", "v3-array", "v2-group", "v2-array"],
)
def test_excluded_field_is_not_part_of_the_document(
    spec_class: type, fields: dict[str, object]
) -> None:
    """
    A field declared with `exclude=True` is not dumped, so it is not part of
    the document. It is not checked as an extra field, and it is not written.
    """

    class Excluded(spec_class):
        note: str = Field(default="none", exclude=True)

    spec = Excluded(**fields, note="kept here")
    assert spec.note == "kept here"
    assert "note" not in spec.to_metadata_json()
    assert Excluded.from_stored_metadata_json(spec.to_stored_metadata_json()).note == "none"


@pytest.mark.parametrize("spec_class", [v3.GroupSpec, v3.BaseGroupSpec])
def test_v3_group_added_field_not_an_object(spec_class: type) -> None:
    """A V3 group field that a subclass adds must be an object, as an extra field must."""

    class Labeled(spec_class):
        label: str = "none"

    with pytest.raises(ValidationError, match="label: a field this class adds"):
        Labeled()


def test_v3_array_added_field_not_an_object() -> None:
    """A V3 array field that a subclass adds must be an object, as an extra field must."""

    class Tagged(v3.ArraySpec):
        tag: str = "none"

    with pytest.raises(ValidationError, match="tag: a field this class adds"):
        Tagged(**v3.ArraySpec.from_array(np.arange(2)).model_dump())


def test_v3_added_field_without_must_understand_false() -> None:
    """
    A V3 field that a subclass adds, and that nothing in the scope claims,
    must say `"must_understand": false`.
    """

    class Labeled(v3.GroupSpec):
        label: dict[str, JSONValue] = Field(default_factory=lambda: {"text": "none"})

    with pytest.raises(ValidationError, match='label: .* does not set "must_understand": false'):
        Labeled()


def test_v3_strict_class_added_field_unclaimed() -> None:
    """
    A strict V3 class rejects a field that it adds when nothing in the scope
    claims the field's name, as it rejects such an extra field.
    """

    class Strict(v3.GroupSpec):
        zarr_scope_strict = True
        label: Label = DEFAULT_LABEL

    with pytest.raises(ValidationError, match="label: no .* in the scope claims 'label'"):
        Strict()


@pytest.mark.parametrize("spec_class", [v2.GroupSpec, v2.BaseGroupSpec])
def test_v2_group_added_field(spec_class: type) -> None:
    """A V2 group's `.zgroup` holds only `zarr_format`, so a subclass cannot add a field."""

    class Labeled(spec_class):
        label: str = "none"

    with pytest.raises(ValidationError, match=r"Fields that this class adds: \['label'\]"):
        Labeled()
