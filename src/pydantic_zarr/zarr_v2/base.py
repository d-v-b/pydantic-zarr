from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

BaseAttr = Mapping[str, object]
TAttr = TypeVar("TAttr", bound=BaseAttr)
