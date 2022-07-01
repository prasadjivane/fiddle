# coding=utf-8
# Copyright 2022 The Fiddle-Config Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for manipulating selections of a Buildable DAG.

A common need for configuration libraries is to override settings in some kind
of base configuration, and these APIs allow such overrides to take place
imperatively.
"""

from typing import Any, Callable, List, Iterator, Optional, Set, Type, Union

from fiddle import config
from fiddle import tagging
from fiddle.experimental import daglish
import tree

# Maybe DRY up with type declaration in autobuilders.py?
FnOrClass = Union[Callable[..., Any], Type[Any]]


class Selection:
  """Represents a selection of nodes.

  This selection is declarative, so if subtrees / subgraphs of `cfg` change and
  later match or don't match, a different set of nodes will be returned.

  Generally this class is intended for modifying attributes of a buildable DAG
  in a way that doesn't alter its structure. We do not pay particular attention
  to structure-altering modifications right now; please do not depend on such
  behavior.
  """
  cfg: config.Buildable
  fn_or_cls: Optional[FnOrClass]
  match_subclasses: bool
  buildable_type: Type[config.Buildable]

  def __init__(
      self,
      cfg: config.Buildable,
      fn_or_cls: Optional[FnOrClass] = None,
      *,
      tag: Optional[tagging.TagType] = None,
      match_subclasses: bool = True,
      buildable_type: Optional[Type[config.Buildable]] = None,
  ):
    super().__setattr__("cfg", cfg)
    super().__setattr__("fn_or_cls", fn_or_cls)
    super().__setattr__("match_subclasses", match_subclasses)

    if buildable_type is None:
      # Set `buildable_type` for implementation in `_matches` below. In general
      # buildable_type should only be None when `tag` is set and `fn_or_cls` is
      # not.
      if tag is None or fn_or_cls is not None:
        buildable_type = config.Buildable

    super().__setattr__("buildable_type", buildable_type)

  def _matches(self, node: config.Buildable) -> bool:
    """Helper for __iter__ function, determining if a node matches."""

    # Implementation note: To allow for future expansion of this class, checks
    # here should be expressed as `if not my_matcher.match(x): return False`.

    if self.buildable_type is not None:
      if not isinstance(node, self.buildable_type):
        return False

    if self.fn_or_cls is not None:
      if self.fn_or_cls != node.__fn_or_cls__:
        # Determines if subclass matching is allowed, and if the node is a
        # subclass of `self.fn_or_cls`. We check whether both are instances
        # of `type` to avoid `issubclass` errors when either side is actually a
        # function.
        is_subclass = (
            self.match_subclasses  #
            and isinstance(self.fn_or_cls, type)  #
            and isinstance(node.__fn_or_cls__, type)  #
            and issubclass(node.__fn_or_cls__, self.fn_or_cls))
        if not is_subclass:
          return False

    return True

  def _iter_helper(self, node: config.Buildable, seen: Set[int]):
    """Helper for __iter__ function below, keeping track of seen nodes."""
    if id(node) in seen:
      return
    seen.add(id(node))

    if self._matches(node):
      yield node

    for leaf in tree.flatten(node.__arguments__):
      if isinstance(leaf, config.Buildable):
        yield from self._iter_helper(leaf, seen)

  def __iter__(self) -> Iterator[config.Buildable]:
    """Iterates over nodes in the tree.

    Returns:
      config.Buildable nodes matching this selection.
    """
    return self._iter_helper(self.cfg, set())

  def __setattr__(self, name: str, value: Any) -> None:
    """Shorthand to set a single value.

    Args:
      name: Name of the attribute to set.
      value: Value to set on all matching nodes.
    """
    for matching in self:
      setattr(matching, name, value)

  def set(self, **kwargs) -> None:
    """Sets multiple attributes on nodes matching this selection.

    Args:
      **kwargs: Properties to set on matching nodes.
    """
    for matching in self:
      for name, value in kwargs.items():
        setattr(matching, name, value)

  def get(self, name: str) -> Iterator[Any]:
    """Gets all values for a particular attribute.

    Args:
      name: Name of the attribute on matching nodes.

    Yields:
      Values configured for the attribute with name `name` on matching nodes.
    """
    for matching in self:
      yield getattr(matching, name)


class TagSelection:
  """Represents a selection of fields tagged by a given tag."""

  cfg: config.Buildable
  tag: tagging.TagType

  def __init__(self, cfg: config.Buildable, tag: tagging.TagType):
    super().__setattr__("cfg", cfg)
    super().__setattr__("tag", tag)

  def set(self, value: Any) -> None:
    """Sets the value for the tag.

    Args:
      value: Value to set.
    """

    def traverse_fn(unused_path, old_value):
      if isinstance(old_value, config.Buildable):
        for name, tags in old_value.__argument_tags__.items():
          if any(issubclass(tag, self.tag) for tag in tags):
            setattr(old_value, name, value)
      return (yield)

    daglish.traverse_with_path(traverse_fn, self.cfg)

  def get(self) -> List[Any]:
    """Yields all values for the selected tag."""
    all_values = []

    def traverse_fn(unused_all_paths, old_value):
      if isinstance(old_value, config.Buildable):
        for name, tags in old_value.__argument_tags__.items():
          if any(issubclass(tag, self.tag) for tag in tags):
            all_values.append(getattr(old_value, name))
      return (yield)

    daglish.memoized_traverse(traverse_fn, self.cfg)
    return all_values


def select(
    cfg: config.Buildable,
    fn_or_cls: Optional[FnOrClass] = None,
    *,
    tag: Optional[tagging.TagType] = None,
    match_subclasses: bool = True,
    buildable_type: Optional[Type[config.Buildable]] = None,
) -> Union[Selection, TagSelection]:
  """Selects sub-buildables or fields within a configuration DAG.

  Example configuring attention classes:

  select(my_config, MyDenseAttention).set(num_heads=12, head_dim=512)

  Example configuring all activation dtypes:

  select(my_config, tag=DType).set(value=jnp.float32)

  Args:
    cfg: Configuraiton to traverse.
    fn_or_cls: Select by a given function or class that is being configured.
    tag: If set, selects all attributes tagged by `tag`. This will return a
      TagSelection instead of a Selection, which has a slightly different API.
    match_subclasses: If fn_or_cls is provided and a class, then also match
      subclasses of `fn_or_cls`.
    buildable_type: Restrict the selection to a particular buildable type. Not
      valid for tag selections.

  Returns:
    Either a Selection or TagSelection object.
  """
  if tag is not None:
    return TagSelection(cfg, tag)
  else:
    return Selection(
        cfg,
        fn_or_cls,
        match_subclasses=match_subclasses,
        buildable_type=buildable_type)
