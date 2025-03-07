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

"""Library for converting generating fiddlers from diffs."""

import ast
import collections
import functools
import re
import types
from typing import List, Tuple, Any, Set, Dict, Callable

from fiddle import config
from fiddle.codegen import codegen
from fiddle.codegen import py_val_to_ast_converter
from fiddle.experimental import daglish
from fiddle.experimental import diff as fdl_diff


def fiddler_from_diff(diff: fdl_diff.Diff,
                      old: Any = None,
                      func_name: str = 'fiddler',
                      param_name: str = 'cfg'):
  """Returns the AST for a fiddler function that applies the changes in `diff`.

  The returned `ast.Module` consists of a set of `import` statements for any
  necessary imports, followed by a function definition for a function whose
  name is `func_name`, which takes a single parameter named `param_name`
  containing a `fdl.Config` (or other `Buildable` or structure), and mutates
  that `Config` in-place as described by `diff`.

  The body of the returned function has three sections:

  * The first section creates variables for any new shared values that are
    added by the diff (i.e., values in `diff.new_shared_values`).
  * The second section creates variables to act as aliases for values in the
    in the input `Config`.  This ensures that we can still reference those
    values even after we've made mutations to the `Config` that might make
    them unreachable from their original location.
  * The final section modifies the `Config` in-place, as described by
    `diff.changes`.  Changes are grouped by the parent object that they modify.
    This section contains one statement for each change.

  Args:
    diff: A `fdl.Diff` describing the change that should be made by the fiddler
      function.
    old: The original config that is transformed by `diff`.  If specified, then
      this is used when creating aliases for values in the input `Config` to
      determine which paths need to have aliases created.  (In particular, it is
      used to determine which paths are aliases for one another.)  If not
      specified, then pessimistically assume that aliases must be created for
      all referenced paths.
    func_name: The name for the fiddler function.
    param_name: The name for the parameter to the fiddler function.

  Returns:
    An `ast.Module` object.  You can convert this to a string using
    `ast.unparse(result)`.
  """
  # Create a namespace to keep track of variables that we add.  Reserve the
  # names of the param & func.
  namespace = codegen.Namespace()
  namespace.add(param_name)
  namespace.add(func_name)

  import_manager = codegen.ImportManager(namespace)

  # Get a list of paths that are referenced by the diff.
  used_paths = _find_used_paths(diff)

  # Add variables for any used paths where the value (or any of the value's
  # ancestors) will be replaced by a change in the diff.  If we don't have an
  # `old` structure, then we pessimistically assume that we need to create
  # variables for all used paths.
  moved_value_names = {}
  if old is not None:
    modified_paths = set(diff.changes)
    _add_path_aliases(modified_paths, old)
    for path in sorted(used_paths, key=daglish.path_str):
      if any(path[:i] in modified_paths for i in range(len(path) + 1)):
        moved_value_names[path] = namespace.get_new_name(
            _path_to_name(path), f'moved_{param_name}_')
  else:
    for path in sorted(used_paths, key=daglish.path_str):
      moved_value_names[path] = namespace.get_new_name(
          _path_to_name(path), f'original_{param_name}_')

  # Add variables for new shared values added by the diff.
  new_shared_value_names = [
      namespace.get_new_name(_name_for_value(value))
      for value in diff.new_shared_values
  ]

  # Construct a PyValToAstConverter to convert constants to AST.
  value_converters = [
      py_val_to_ast_converter.ValueConverter(
          matcher=types.ModuleType,
          priority=200,
          converter=functools.partial(
              _convert_module, import_manager=import_manager)),
      py_val_to_ast_converter.ValueConverter(
          matcher=fdl_diff.Reference,
          priority=200,
          converter=functools.partial(
              _convert_reference,
              param_name=param_name,
              moved_value_names=moved_value_names,
              new_shared_value_names=new_shared_value_names)),
  ]

  pyval_to_ast = functools.partial(
      py_val_to_ast_converter.convert_py_val_to_ast,
      additional_converters=value_converters)

  body = []
  body += _ast_for_new_shared_value_variables(diff.new_shared_values,
                                              new_shared_value_names,
                                              pyval_to_ast)
  body += _ast_for_moved_value_variables(param_name, moved_value_names)
  body += _ast_for_changes(diff, param_name, moved_value_names, pyval_to_ast)

  imports = _ast_for_imports(import_manager)
  fiddler = _ast_for_fiddler(func_name, param_name, body)

  result = ast.Module(body=imports + [fiddler], type_ignores=[])

  # Add .lineno to AST nodes (required by `ast.unparse`).
  ast.fix_missing_locations(result)

  return result


def _ast_for_imports(import_manager: codegen.ImportManager) -> List[ast.AST]:
  """Returns a list of `ast.AST` for import satements in `import_manager`."""
  imp_lines = []
  for imp in import_manager.sorted_imports():
    imp_lines.extend(imp.lines())
  import_str = '\n'.join(imp_lines)
  module = ast.parse(import_str)
  assert isinstance(module, ast.Module)
  return module.body


def _ast_for_fiddler(func_name: str, param_name: str,
                     body: List[ast.AST]) -> ast.FunctionDef:
  """Returns an `ast.FunctionDef` for the fiddler function.

  Args:
    func_name: The name of the fiddler function.
    param_name: The name of the fiddler function's parameter.
    body: The body of the fiddler function.
  """
  return ast.FunctionDef(
      name=func_name,
      args=ast.arguments(
          args=[ast.arg(arg=param_name)],
          posonlyargs=[],
          kwonlyargs=[],
          kw_defaults=[],
          defaults=[]),
      body=body,
      decorator_list=[])


# A function that takes any python value, and returns an ast node.
PyValToAstFunc = Callable[[Any], ast.AST]


def _ast_for_new_shared_value_variables(
    values: Tuple[Any], names: List[str],
    pyval_to_ast: PyValToAstFunc) -> List[ast.AST]:
  """Returns a list of `ast.AST` for creating new shared value variables."""
  statements = []
  for value, name in sorted(zip(values, names), key=lambda item: item[1]):
    statements.append(
        ast.Assign(
            targets=[ast.Name(name, ctx=ast.Store())],
            value=pyval_to_ast(value)))
  return statements


def _ast_for_moved_value_variables(
    param_name: str, moved_value_names: Dict[daglish.Path,
                                             str]) -> List[ast.AST]:
  """Returns a list of `ast.AST` for creating moved value alias variables."""
  statements = []
  sorted_moved_value_names = sorted(
      moved_value_names.items(), key=lambda item: daglish.path_str(item[0]))
  for path, name in sorted_moved_value_names:
    statements.append(
        ast.Assign(
            targets=[ast.Name(name, ctx=ast.Store())],
            value=_ast_for_path(param_name, path)))
  return statements


def _find_used_paths(diff: fdl_diff.Diff) -> Set[daglish.Path]:
  """Returns a list of paths referenced in `diff`.

  This list includes paths for any values we might need to create aliases
  for, if that value moved.  In particular, it includes the parent path
  for each change in `diff.changes`, plus the target path for any
  `diff.Reference` in `diff` whose root is `'old'`.

  Args:
    diff: The `fdl.Diff` that should be scanned for used paths.
  """
  # For each change, we need the path to its *parent* object.
  used_paths = set(path[:-1] for path in diff.changes)

  # For each Reference to `old`, we need the target path.
  def collect_ref_targets(path, node):
    del path  # Unused.
    yield
    if isinstance(node, fdl_diff.Reference) and node.root == 'old':
      used_paths.add(node.target)

  for change in diff.changes.values():
    if isinstance(change, (fdl_diff.SetValue, fdl_diff.ModifyValue)):
      daglish.traverse_with_path(collect_ref_targets, change.new_value)
  daglish.traverse_with_path(collect_ref_targets, diff.new_shared_values)

  return used_paths


def _add_path_aliases(paths: Set[daglish.Path], structure: Any):
  """Update `paths` to include any other paths that reach the same objects.

  If any value `v` reachable by a path `p` in `paths` is also reachable by one
  or more other paths, then add those paths to `paths`.  E.g., if a shared
  object is reachable by paths `.x.y` and `.x.z', and `paths` includes
  only `.x.y`, then this will add `.x.z` to `paths`.

  Args:
    paths: A set of paths to values in `structure`.
    structure: The structure used to determine the paths for shared values.
  """
  path_to_value = daglish.collect_value_by_path(structure, memoizable_only=True)
  id_to_paths = daglish.collect_paths_by_id(structure, memoizable_only=True)

  for path in list(paths):
    value = path_to_value.get(path, None)  # None if not memoizable.
    if value is not None:
      paths.update(id_to_paths[id(value)])


ChangeToChild = Tuple[daglish.PathElement, fdl_diff.DiffOperation]
ChangesByParent = List[Tuple[daglish.Path, List[ChangeToChild]]]


def _group_changes_by_parent(diff: fdl_diff.Diff) -> ChangesByParent:
  """Returns a sorted list of changes in `diff`, grouped by their parent."""
  # Group changes by parent path.
  changes_by_parent = collections.defaultdict(list)
  for (path, change) in diff.changes.items():
    if not path:
      raise ValueError('Changing the root object is not supported')
    changes_by_parent[path[:-1]].append((path[-1], change))

  # Sort by path (converted to path_str).
  return sorted(
      changes_by_parent.items(), key=lambda item: daglish.path_str(item[0]))


def _ast_for_changes(diff: fdl_diff.Diff, param_name: str,
                     moved_value_names: Dict[daglish.Path, str],
                     pyval_to_ast: PyValToAstFunc) -> List[ast.AST]:
  """Returns a list of AST nodes that apply the changes described in `diff`.

  Args:
    diff: The `fdl.Diff` whose changes should be applied.
    param_name: The name of the parameter to the fiddler function.
    moved_value_names: Dictionary mapping any paths that might become
      unreachable once the config is mutated to alias variables that can be used
      to reach those values.
    pyval_to_ast: A function used to convert Python values to AST.
  """
  body = []

  # Apply changes to a single parent at a time.
  for parent_path, changes in _group_changes_by_parent(diff):

    # Get an AST expression that can be used to refer to the parent.
    if parent_path in moved_value_names:
      parent_ast = ast.Name(moved_value_names[parent_path], ctx=ast.Load())
    else:
      parent_ast = _ast_for_path(param_name, parent_path)

    # Add AST statements that apply the changes to the parent.  Ensure that
    # all DeleteValues occur before Buildable.__fn_or_cls__ is changed, and
    # that all SetValues occur after Buildable.__fn_or_cls__ is changed
    # (because changing __fn_or_cls__ can change the set of parameters that
    # a Buildable is allowed to have).
    deletes = []
    update_callable = None
    assigns = []
    for child_path_elt, change in changes:
      child_ast = _ast_for_child(parent_ast, child_path_elt, pyval_to_ast)

      if isinstance(child_path_elt, daglish.BuildableFnOrCls):
        assert isinstance(change, fdl_diff.ModifyValue)
        assert update_callable is None
        new_value_ast = pyval_to_ast(change.new_value)
        update_callable = ast.Expr(
            value=ast.Call(
                func=pyval_to_ast(config.update_callable),
                args=[parent_ast, new_value_ast],
                keywords=[]))

      elif isinstance(change, fdl_diff.DeleteValue):
        child_ast.ctx = ast.Del()
        deletes.append(ast.Delete(targets=[child_ast]))

      elif isinstance(change, (fdl_diff.SetValue, fdl_diff.ModifyValue)):
        child_ast.ctx = ast.Store()
        new_value_ast = pyval_to_ast(change.new_value)
        assigns.append(ast.Assign(targets=[child_ast], value=new_value_ast))

      else:
        raise ValueError(f'Unsupported DiffOperation {type(change)}')

    body.extend(deletes)
    if update_callable is not None:
      body.append(update_callable)
    body.extend(assigns)

  return body


def _ast_for_child(parent_ast: ast.AST, child_path_elt: daglish.PathElement,
                   pyval_to_ast: PyValToAstFunc) -> ast.AST:
  """Returns an AST expression that can be used to access a child of a parent.

  Args:
    parent_ast: AST expression for the parent object.
    child_path_elt: A PathElement specifying a child of the parent.
    pyval_to_ast: A function used to convert Python values to AST.
  """
  if isinstance(child_path_elt, daglish.Attr):
    return ast.Attribute(
        value=parent_ast, attr=child_path_elt.name, ctx=ast.Load())
  elif isinstance(child_path_elt, daglish.Index):
    index_ast = pyval_to_ast(child_path_elt.index)
    return ast.Subscript(value=parent_ast, slice=index_ast, ctx=ast.Load())
  elif isinstance(child_path_elt, daglish.Key):
    key_ast = pyval_to_ast(child_path_elt.key)
    return ast.Subscript(value=parent_ast, slice=key_ast, ctx=ast.Load())
  else:
    raise ValueError(f'Unsupported PathElement {type(child_path_elt)}')


def _ast_for_path(name: str, path: daglish.Path):
  """Converts a `daglish.Path` to an `ast.AST` expression."""
  node = ast.Name(id=name, ctx=ast.Load())
  for path_elt in path:
    if isinstance(path_elt, daglish.Index):
      node = ast.Subscript(
          value=node, slice=ast.Constant(path_elt.index), ctx=ast.Load())
    elif isinstance(path_elt, daglish.Key):
      assert isinstance(path_elt.key, (int, str, bool))
      node = ast.Subscript(
          value=node, slice=ast.Constant(path_elt.key), ctx=ast.Load())
    elif isinstance(path_elt, daglish.Attr):
      node = ast.Attribute(value=node, attr=path_elt.name, ctx=ast.Load())
    else:
      raise ValueError(f'Unsupported PathElement {path_elt}')
  return node


def _camel_to_snake(name: str) -> str:
  """Converts a camel or studly-caps name to a snake_case name."""
  return re.sub(r'(?<=.)([A-Z])', lambda m: '_' + m.group(0).lower(),
                name).lower()


def _name_for_value(value: Any) -> str:
  """Returns a name for a value, based on its type."""
  if isinstance(value, config.Buildable):
    return _camel_to_snake(value.__fn_or_cls__.__name__)
  else:
    return _camel_to_snake(type(value).__name__)


def _path_to_name(path: daglish.Path) -> str:
  """Converts a path to a variable name."""
  name = daglish.path_str(path)
  name = re.sub('[^a-zA-Z_0-9]+', '_', name)
  return name.strip('_').lower()


def _convert_reference(value, convert_child, param_name, moved_value_names,
                       new_shared_value_names) -> ast.AST:
  """Converts a `Reference` to an AST expression."""
  del convert_child  # Unused.
  if value.root == 'old':
    if value.target in moved_value_names:
      return ast.Name(moved_value_names[value.target], ctx=ast.Load())
    else:
      return _ast_for_path(param_name, value.target)
  else:
    assert isinstance(value.target[0], daglish.Index)
    var_name = new_shared_value_names[value.target[0].index]
    return _ast_for_path(var_name, value.target[1:])


def _convert_module(value, convert_child, import_manager) -> ast.AST:
  """Converts a Module to AST, using an ImportManager."""
  del convert_child  # Unused.
  name = import_manager.add_by_name(value.__name__)
  return py_val_to_ast_converter.dotted_name_to_ast(name)
