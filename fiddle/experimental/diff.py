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

"""Library for finding differences between Fiddle configurations."""

import dataclasses

from typing import Any, Dict, Sequence, List, Tuple, Union
from fiddle import config
from fiddle.experimental import daglish


class AlignmentError(ValueError):
  """Indicates that two values cannot be aligned."""


@dataclasses.dataclass(frozen=True)
class Reference(object):
  """Symbolic reference to an object in a `Buildable`."""
  root: str
  target: daglish.Path

  def __repr__(self):
    return f'<Reference: {self.root}{daglish.path_str(self.target)}>'


class DiffOperation:
  """Base class for diff operations.

  Each `DiffOperation` describes a single change to a target `daglish.Path`.
  """


@dataclasses.dataclass(frozen=True)
class Diff:
  """Describes a set of changes to a `Buildable`.

  Attributes:
    changes: A dictionary whose keys are `Path`s used to identify objects in
      `old`; and whose values are `DiffOperation`s describing how those objects
      should be mutated.
    new_shared_values: A list of new shared values that can be pointed to using
      `Reference` objects in `changes`.
  """
  changes: Dict[daglish.Path, DiffOperation]
  new_shared_values: Tuple[Any, ...] = ()

  def __str__(self):
    return ('Diff(changes=[\n' +
            '\n'.join(f'          {daglish.path_str(path)}: {change!r}'
                      for (path, change) in self.changes.items()) +
            '\n      ],\n      new_shared_values=[\n' +
            '\n'.join(f'          {v!r}' for v in self.new_shared_values) +
            '\n      ])')


@dataclasses.dataclass(frozen=True)
class SetValue(DiffOperation):
  """Changes the target to new_value; fails if the target already has a value.

  The target's parent may not be a sequence (list or tuple).
  """
  new_value: Union[Reference, Any]


@dataclasses.dataclass(frozen=True)
class ModifyValue(DiffOperation):
  """Changes the target to new_value; fails if the target has no prior value.

  The target's parent may not be a tuple.
  """
  new_value: Union[Reference, Any]


@dataclasses.dataclass(frozen=True)
class DeleteValue(DiffOperation):
  """Removes the target from its parent; fails if the target has no prior value.

  The target's parent may not be a sequence (list or tuple).
  """


@dataclasses.dataclass(frozen=True)
class AlignedValues:
  """A pair of aligned values."""
  old_value: Any
  new_value: Any


@dataclasses.dataclass(frozen=True)
class AlignedValueIds:
  """A pair of `id`s for aligned values."""
  old_value_id: int
  new_value_id: int


class DiffAlignment:
  """Alignment between two structures, named `old` and `new`.

  `DiffAlignment` is a partial bidirectional mapping between nested objects in
  two structures (`old` and `new`).  When building diffs, this alignment is
  used to decide which objects in `old` should be mutated to become objects in
  `new`.

  This class places the following restrictions on when objects in `old` and
  `new` may be aligned:

  * Each `old_value` may be aligned with at most one `new_value`
    (and vice versa).
  * `type(new_value)` must be equal to `type(old_value)`.
  * If `isinstance(old_value, Sequence)`, then `len(new_value)` must be equal
    to `len(old_value)`.
  * If `old_value` and `new_value` are aligned, then it must be possible to
    mutate `old_value` to become `new_value`.
  * Alignments may not create cycles.  E.g., `old_value` may not be aligned
    with `new_value` if some value contained in `old_value` is aligned with a
    value that contains `new_value`.
  * Aligned values must be "memoizable" (as defined by `daglish.is_memoizable`).

  These restrictions help ensure that a `Diff` can be built from the alignment.
  Note: `DiffAlignment` is not guaranteed to catch all violations of these
  restrictions, since some are difficult or expensive to detect.
  """

  def __init__(self,
               old: Any,
               new: Any,
               old_name: str = 'old',
               new_name: str = 'new'):
    """Creates an empty alignment between objects in `old` and `new`.

    In the new alignment, no objects in `old` are aligned with any objects
    in `new` (including the root objects `old` and `new` themselves).  Call
    `DiffAlignment.align` to add alignments.

    Args:
      old: The root object of the `old` structure.
      new: The root object of the `new` structure.
      old_name: A name for the `old` structure (used to print alignments).
      new_name: A name for the `new` structure (used to print alignments).
    """
    self._old: Any = old
    self._new: Any = new
    self._old_name: str = old_name
    self._new_name: str = new_name
    self._new_by_old_id: Dict[int, Any] = {}  # id(old_value) -> new_value
    self._old_by_new_id: Dict[int, Any] = {}  # id(new_value) -> old_value

  @property
  def old(self) -> Any:
    """The root object of the `old` structure for this alignment."""
    return self._old

  @property
  def new(self) -> Any:
    """The root object of the `new` structure for this alignment."""
    return self._new

  @property
  def old_name(self) -> str:
    """The name of the `old` structure for this alignment."""
    return self._old_name

  @property
  def new_name(self) -> str:
    """The name of the `new` structure for this alignment."""
    return self._new_name

  def is_old_value_aligned(self, old_value):
    """Returns true if `old_value` is aligned with any value."""
    return id(old_value) in self._new_by_old_id

  def is_new_value_aligned(self, new_value):
    """Returns true if `new_value` is aligned with any value."""
    return id(new_value) in self._old_by_new_id

  def new_from_old(self, old_value):
    """Returns the object in `new` that is aligned with `old_value`."""
    return self._new_by_old_id[id(old_value)]

  def old_from_new(self, new_value):
    """Returns the object in `old` that is aligned with `new_value`."""
    return self._old_by_new_id[id(new_value)]

  def aligned_values(self) -> List[AlignedValues]:
    """Returns a list of `(old_value, new_value)` for all aligned values."""
    return [
        AlignedValues(self._old_by_new_id[id(new_value)], new_value)
        for (old_id, new_value) in self._new_by_old_id.items()
    ]

  def aligned_value_ids(self) -> List[AlignedValueIds]:
    """Returns a list of `(id(old_val), id(new_val))` for all aligned values."""
    return [
        AlignedValueIds(old_id, id(new_value))
        for (old_id, new_value) in self._new_by_old_id.items()
    ]

  def align(self, old_value: Any, new_value: Any):
    """Aligns `old_value` with `new_value`.

    Assumes that `old_value` is contained in `self.old`, and that `new_value`
    is contained in `self.new`.

    Args:
      old_value: The value in `old` that should be aligned with `new_value.`
      new_value: The value in `new` that should be aligned with `old_value.`

    Raises:
      AlignmentError: If `old_value` and `new_value` can not be aligned.  For
        example, this can happen if either value is already aligned, or if
        the values are incompatible.  See the docstring for `DiffAlignment`
        for a full list of restrictions.  Note: the `align` method is not
        guaranteed to catch all violations of these restrictions, since some
        are difficult or expensive to detect.
    """
    self._validate_alignment(old_value, new_value)
    self._new_by_old_id[id(old_value)] = new_value
    self._old_by_new_id[id(new_value)] = old_value

  def can_align(self, old_value, new_value):
    """Returns true if `old_value` could be aligned with `new_value`."""
    if not daglish.is_memoizable(old_value):
      return False
    if not daglish.is_memoizable(new_value):
      return False
    if self.is_old_value_aligned(old_value):
      return False
    if self.is_new_value_aligned(new_value):
      return False
    if type(old_value) is not type(new_value):
      return False
    if isinstance(old_value, Sequence) and len(old_value) != len(new_value):
      return False
    return True

  def _validate_alignment(self, old_value, new_value):
    """Raises AlignmentError if old_value can not be aligned with new_value."""
    if not daglish.is_memoizable(old_value):
      raise AlignmentError(f'old_value={old_value!r} may not be aligned because'
                           ' it is not memoizable.')
    if not daglish.is_memoizable(new_value):
      raise AlignmentError(f'new_value={new_value!r} may not be aligned because'
                           ' it is not memoizable.')
    if self.is_old_value_aligned(old_value):
      raise AlignmentError('An alignment has already been added for ' +
                           f'old value {old_value!r}')
    if self.is_new_value_aligned(new_value):
      raise AlignmentError('An alignment has already been added for ' +
                           f'new value {new_value!r}')
    if type(old_value) is not type(new_value):
      raise AlignmentError(
          f'Aligning objects of different types is not currently '
          f'supported.  ({type(old_value)} vs {type(new_value)})')
    if isinstance(old_value, Sequence):
      if len(old_value) != len(new_value):
        raise AlignmentError(
            f'Aligning sequences with different lengths is not '
            f'currently supported.  ({len(old_value)} vs {len(new_value)})')

  def __repr__(self):
    return (
        f'<DiffAlignment from {self._old_name!r} to ' +
        f'{self._new_name!r}: {len(self._new_by_old_id)} object(s) aligned>')

  def __str__(self):
    id_to_old_path = daglish.collect_paths_by_id(self.old, memoizable_only=True)
    id_to_new_path = daglish.collect_paths_by_id(self.new, memoizable_only=True)
    old_to_new_paths = [(id_to_old_path[aligned_ids.old_value_id][0],
                         id_to_new_path[aligned_ids.new_value_id][0])
                        for aligned_ids in self.aligned_value_ids()]
    lines = [
        f'    {self.old_name}{daglish.path_str(old_path)}'
        f' -> {self.new_name}{daglish.path_str(new_path)}'
        for (old_path, new_path) in old_to_new_paths
    ]
    if not lines:
      lines.append('    (no objects aligned)')
    return 'DiffAlignment:\n' + '\n'.join(lines)


def align_by_id(old: Any, new: Any, old_name='old', new_name='new'):
  """Aligns any memoizable object that is contained in both `old` and `new`.

  Returns a `DiffAlignment` that aligns any memoizable object that can be
  reached by traversing both `old` and `new`.  (It must be the same object,
  as defined by `is`; not just an equal object.)

  I.e., if `old_values` is the list of all memoizable objects reachable from
  `old`, and `new_values` is the list of all memoizable objects reachable from
  `new`, then this will call `alignment.align(v, v)` for any `v` that is in
  both `old_values` and `new_values`.

  Args:
    old: The root object of the `old` structure.
    new: The root object of the `new` structure.
    old_name: A name for the `old` structure.
    new_name: A name for the `new` structure.

  Returns:
    A `DiffAlignment`.
  """
  alignment = DiffAlignment(old, new, old_name, new_name)
  old_by_id = daglish.collect_value_by_id(old, memoizable_only=True)
  new_by_id = daglish.collect_value_by_id(new, memoizable_only=True)
  for (value_id, value) in old_by_id.items():
    if value_id in new_by_id:
      alignment.align(value, value)
  return alignment


def align_heuristically(old: Any, new: Any, old_name='old', new_name='new'):
  """Returns an alignment between `old` and `new`, based on heuristics.

  These heuristics may be changed or improved over time, and are not guaranteed
  to stay the same for different versions of Fiddle.

  The current implementation makes three passes over the structures:

  * The first pass aligns any memoizable object that can be reached by
    traversing both `old` and `new`.  (It must be the same object, as defined
    by `is`; not just an equal object.)

  * The second pass aligns any memoizable objects in `old` and `new` that can
    be reached using the same path.

  * The third pass aligns any memoizable objects in `old` and `new` that have
    equal values.  Note: this takes `O(size(old) * size(new))` time.

  Args:
    old: The root object of the `old` structure.
    new: The root object of the `new` structure.
    old_name: A name for the `old` structure.
    new_name: A name for the `new` structure.

  Returns:
    A `DiffAlignment`.
  """
  # First pass: align by id.
  alignment = DiffAlignment(old, new, old_name, new_name)
  old_by_id = daglish.collect_value_by_id(old, memoizable_only=True)
  new_by_id = daglish.collect_value_by_id(new, memoizable_only=True)
  for (value_id, value) in old_by_id.items():
    if value_id in new_by_id:
      alignment.align(value, value)

  # Second pass: align any objects that are reachable by the same path.
  path_to_old = daglish.collect_value_by_path(old, memoizable_only=True)
  path_to_new = daglish.collect_value_by_path(new, memoizable_only=True)
  for (path, old_value) in path_to_old.items():
    if path in path_to_new:
      if alignment.can_align(old_value, path_to_new[path]):
        alignment.align(old_value, path_to_new[path])

  # Third pass: align any objects that are equal (__eq__).
  for old_value in old_by_id.values():
    for new_value in new_by_id.values():
      if type(old_value) is type(new_value) and old_value == new_value:
        if alignment.can_align(old_value, new_value):
          alignment.align(old_value, new_value)

  return alignment


class _DiffFromAlignmentBuilder:
  """Class used to build a `Diff` from a `DiffAlignment`.

  This private class is used to implement `build_diff_from_alignment`.
  """
  alignment: DiffAlignment
  changes: Dict[daglish.Path, DiffOperation]
  new_shared_values: List[Any]
  paths_by_old_id: Dict[int, List[daglish.Path]]

  def __init__(self, alignment: DiffAlignment):
    self.changes: Dict[daglish.Path, DiffOperation] = {}
    self.new_shared_values: List[Any] = []
    self.alignment: DiffAlignment = alignment
    self.paths_by_old_id = daglish.collect_paths_by_id(
        alignment.old, memoizable_only=True)

  def build_diff(self) -> Diff:
    """Returns a `Diff` between `alignment.old` and `alignment.new`."""
    if self.changes or self.new_shared_values:
      raise ValueError('build_diff should be called at most once.')
    daglish.memoized_traverse(self.record_diffs, self.alignment.new)
    return Diff(self.changes, tuple(self.new_shared_values))

  def record_diffs(self, new_paths: daglish.Paths, new_value: Any):
    """Daglish traversal function that records diffs to generate `new_value`.

    If `new_value` is not aligned with any `old_value`, and `new_value` can
    be reached by a single path, returns `new_value` as-is.

    If `new_value` is not aligned with any `old_value`, and `new_value` can
    be reached by multiple paths, then adds `new_value` to
    `self.new_shared_values`, and returns a reference to the new shared value.

    If `new_value` is aligned with any `old_value`, then updates
    `self.changes` with any changes necessary to mutate `old_value` into
    `new_value`.

    Returns a copy of `new_value`, or a `Reference` pointing at a shared copy of
    `new_value` or an `old_value` that will be transformed into `new_value`.

    Args:
      new_paths: The paths to a value reachable from `alignment.new`.
      new_value: The value reachable from `alignment.new`.

    Yields:
      None
    """
    # `diff_value` is a copy of `new_value` with shared objects replaced by
    # `Reference`s where appropriate.
    diff_value = yield

    if not self.alignment.is_new_value_aligned(new_value):  # New object.
      if len(new_paths) == 1 or not daglish.is_memoizable(new_value):
        return diff_value
      else:
        index = len(self.new_shared_values)
        self.new_shared_values.append(diff_value)
        return Reference(
            root='new_shared_value', target=(daglish.Index(index),))
    else:
      # Old object: check for modifications.  (Note: only memoizable values
      # may be aligned, so old_value must be memoizable here.)
      old_value = self.alignment.old_from_new(new_value)
      old_path = self.paths_by_old_id[id(old_value)][0]
      if isinstance(new_value, config.Buildable):
        self.record_buildable_diffs(old_path, old_value, new_value, diff_value)
      elif isinstance(new_value, Dict):
        self.record_dict_diffs(old_path, old_value, new_value, diff_value)
      elif isinstance(new_value, Sequence):
        self.record_sequence_diffs(old_path, old_value, new_value, diff_value)

      return Reference(root='old', target=old_path)

  def record_buildable_diffs(self, old_path: daglish.Path,
                             old_value: config.Buildable,
                             new_value: config.Buildable,
                             diff_value: config.Buildable):
    """Records changes needed to turn Buildable `old_value` into `new_value."""
    if old_value.__fn_or_cls__ != new_value.__fn_or_cls__:
      old_callable_path = old_path + (daglish.BuildableFnOrCls(),)
      self.changes[old_callable_path] = ModifyValue(new_value.__fn_or_cls__)

    for name, old_child in old_value.__arguments__.items():
      old_child_path = old_path + (daglish.Attr(name),)
      if name in new_value.__arguments__:
        new_child = getattr(new_value, name)
        if not self.aligned_or_equal(old_child, new_child):
          self.changes[old_child_path] = ModifyValue(getattr(diff_value, name))
      else:
        self.changes[old_child_path] = DeleteValue()

    for name in new_value.__arguments__:
      if name not in old_value.__arguments__:
        old_child_path = old_path + (daglish.Attr(name),)
        self.changes[old_child_path] = SetValue(getattr(diff_value, name))

  def record_dict_diffs(self, old_path: daglish.Path, old_value: Dict[Any, Any],
                        new_value: Dict[Any, Any], diff_value: Dict[Any, Any]):
    """Records changes needed to turn dict `old_value` into `new_value."""
    for key, old_child in old_value.items():
      old_child_path = old_path + (daglish.Key(key),)
      if key in new_value:
        if not self.aligned_or_equal(old_child, new_value[key]):
          self.changes[old_child_path] = ModifyValue(diff_value[key])
      else:
        self.changes[old_child_path] = DeleteValue()

    for key in new_value:
      if key not in old_value:
        old_child_path = old_path + (daglish.Key(key),)
        self.changes[old_child_path] = SetValue(diff_value[key])

  def record_sequence_diffs(self, old_path: daglish.Path,
                            old_value: Sequence[Any], new_value: Sequence[Any],
                            diff_value: Sequence[Any]):
    """Records changes needed to turn sequence `old_value` into `new_value."""
    for index, old_child in enumerate(old_value):
      old_child_path = old_path + (daglish.Index(index),)
      if not self.aligned_or_equal(old_child, new_value[index]):
        self.changes[old_child_path] = ModifyValue(diff_value[index])

  def aligned_or_equal(self, old_value: Any, new_value: Any) -> bool:
    """Returns true if `old_value` and `new_value` are aligned or equal.

    * If either `old_value` or `new_value` is memoizable, then returns True
      if they are aligned.
    * Otherwise, then returns True if they are equal.

    Args:
      old_value: A value reachable from `self.alignment.old`.
      new_value: A value reachable from `self.alignment.new`.
    """
    if daglish.is_memoizable(new_value) or daglish.is_memoizable(old_value):
      return (self.alignment.is_old_value_aligned(old_value) and
              self.alignment.new_from_old(old_value) is new_value)
    elif old_value is new_value:
      return True
    elif type(new_value) is not type(old_value):
      return False
    else:
      return old_value == new_value


def build_diff_from_alignment(alignment: DiffAlignment) -> Diff:
  """Returns a `Diff` with the changes from `alignment.old` to `alignment.new`.

  Args:
    alignment: The `DiffAlignment` between two structures (`old` and `new`).

  Returns:
    A `Diff` describing the changes needed to transform `old` into `new`.
    For values in `new` that are aligned with values in `old`, the diff
    describes how to modify `old_value` in place to become `new_value`. Values
    in `new` that are not aligned are added by the diff as new values.
  """
  return _DiffFromAlignmentBuilder(alignment).build_diff()