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

"""Library for expressing placeholders.

When defining shared parameters across a project that later could be changed,
for example dtype or activation function, we encourage the following coding
pattern: a tiny shared library file should declare placeholder keys for an
entire project, like

  activation_dtype = fdl.PlaceholderKey("activation_dtype")
  kernel_init_fn = fdl.PlaceholderKey("kernel_init_fn")

Then, in library code which configures shared Fiddle fixtures, these keys are
used,

  def layer_norm_fixture() -> fdl.Config[LayerNorm]:
    cfg = fdl.Config(LayerNorm)
    cfg.dtype = fdl.Placeholder(activation_dtype, jnp.float32)

And in experiment code stitching everything together, all of these placeholders
can be set at once,

  def encoder_decoder_fixture() -> fdl.Config[EncoderDecoder]:
    cfg = fdl.Config(EncoderDecoder)
    ...
    cfg.encoder.encoder_norm = layer_norm_fixture()

    # Set all activation dtypes.
    fdl.set_placeholder(cfg, activation_dtype, jnp.bfloat16)
    return cfg

  model = fdl.build(encoder_decoder_fixture())
"""

from __future__ import annotations

import copy
import dataclasses
import pathlib
import traceback
from typing import Any, FrozenSet, Generic, Optional, Set, TypeVar, Union

from fiddle import config
import tree


class _NoValue:
  """Sentinel class (used in place of object for more precise errors)."""

  def __deepcopy__(self, memo):
    """Override for deepcopy that does not copy this sentinel object."""
    del memo
    return self


NO_VALUE = _NoValue()


def _infer_module_filename():

  stack = traceback.extract_stack()
  if len(stack) < 3:
    return None
  return stack[-3].filename


@dataclasses.dataclass(frozen=True)
class PlaceholderKey:
  """Represents a key identifying a type of placeholder.

  Instances of this class are typically compared on object identity, so please
  share them via a common library file (see module docstring).
  """
  name: str
  description: Optional[str] = None
  module_filename: Optional[str] = dataclasses.field(
      default_factory=_infer_module_filename)

  def __hash__(self):
    return id(self)

  def __repr__(self):
    return f"PlaceholderKey(name={self.name!r})"

  def __deepcopy__(self, memo):
    """Override for deepcopy that does not copy this immutable object.

    This is important because in `set_placeholder` we use object identity on
    keys, so we should not unnecessarily copy them.

    Args:
      memo: Unused memoization object from deepcopy API.

    Returns:
      `self`.
    """
    del memo
    return self


def placeholder_fn(keys: Set[PlaceholderKey], value: Any = NO_VALUE) -> Any:
  if value is NO_VALUE:
    raise config.PlaceholderNotFilledError(
        "Expected all placeholders to be replaced via fdl.set_placeholder() "
        f"calls, but one with keys {keys} was not set.")
  else:
    return value


T = TypeVar("T")


class Placeholder(Generic[T], config.Config[T]):
  """Declares a placeholder in a configuration."""

  def __init__(
      self,
      keys: Union[Set[PlaceholderKey], Placeholder[T]] = None,
      default: Union[_NoValue, T] = NO_VALUE,
  ):
    """Initializes the placeholder.

    Args:
      keys: Set of keys for this placeholder. Identifies types of this
        placeholder, so that all placeholders matching a given key can be set at
        once. Alternately an existing Placeholder can be specified, and it will
        be copied.
      default: Default value of the placeholder. This is normally a sentinel
        which will cause the configuration to fail to build when the
        placeholders are not set.
    """
    if isinstance(keys, Placeholder):
      assert default is NO_VALUE
      super().__init__(keys)
    else:
      if not keys:
        raise ValueError(
            "Placeholder(): Please provide a nonempty set for `keys`")
      super().__init__(placeholder_fn, keys=keys, value=default)

  def __deepcopy__(self, memo) -> config.Buildable[T]:
    """Implements the deepcopy API."""
    return Placeholder(
        keys=copy.deepcopy(self.keys, memo),
        default=copy.deepcopy(self.value, memo))


def set_placeholder(cfg: config.Buildable, key: PlaceholderKey,
                    value: Any) -> None:
  """Replaces placeholders of a given key with a given value.

  Specifically, placeholders are config.Config sub-nodes with keys and values.
  Calling this method will preserve the placeholder nodes, in case experiment
  configs would like to override placeholder values from another config/fixture.

  This implementation does not currently track nodes with shared parents, so for
  some pathological Buildable DAGs, e.g. lattices, it could take exponential
  time. We think these are edge cases, and could be easily fixed later.

  Args:
    cfg: A tree of buildable elements. This tree is mutated.
    key: Key identifying which placeholders' values should be set.
    value: Value to set for these placeholders.
  """
  if isinstance(cfg, Placeholder):
    if key in cfg.keys:
      cfg.value = value

  def map_fn(leaf):
    if isinstance(leaf, config.Buildable):
      set_placeholder(leaf, key, value)
    return leaf

  tree.map_structure(map_fn, cfg.__arguments__)


def list_placeholder_keys(cfg: config.Buildable) -> FrozenSet[PlaceholderKey]:
  """Lists all placeholder keys in a buildable.

  Args:
    cfg: A tree of buildable elements.

  Returns:
    Set of placeholder keys used in this buildable.
  """
  keys = set()

  def _inner(node: config.Buildable):
    if isinstance(node, Placeholder):
      keys.update(node.keys)

    def map_fn(leaf):
      if isinstance(leaf, config.Buildable):
        _inner(leaf)

    tree.map_structure(map_fn, node.__arguments__)

  _inner(cfg)
  return frozenset(keys)


def print_keys(
    keys: Union[config.Buildable, FrozenSet[PlaceholderKey]],
    relative_filenames=True,
) -> None:
  """Prints placeholder keys for a buildable or set of keys.

  Args:
    keys: Either a buildable or set of placeholder keys. If this is a buildable
      then list_placeholder_keys() will be called on it.
    relative_filenames: Whether to print relative filenames that keys are
      defined in.
  """

  if isinstance(keys, config.Buildable):
    keys = list_placeholder_keys(keys)

  for key in keys:
    desc_str = "" if not key.description else f": {key.description}"
    if key.module_filename:
      filename = pathlib.Path(key.module_filename)
      if len(filename.parents) > 3 and relative_filenames:
        filename = ".../" + str(filename.relative_to(filename.parents[3]))
      module_str = f" (defined in {filename})"
    else:
      module_str = ""
    print(f" - {key.name}{desc_str}{module_str}")
