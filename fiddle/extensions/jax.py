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

"""Fiddle extensions to handle JAX code more elegantly.

Currently this just affects codegen, graphviz, and other debugging functions.
"""

import ast
from fiddle.codegen import codegen
from fiddle.codegen import mini_ast
from fiddle.codegen import py_val_to_ast_converter
from fiddle.codegen import special_value_codegen
import jax
from jax import numpy as jnp


def _make_jnp_importable(name: str) -> special_value_codegen.Importable:
  return special_value_codegen.SingleImportable(
      "jax.numpy", lambda jnp_name: f"{jnp_name}.{name}")


_jnp_type_importables = (
    (jnp.bool_, _make_jnp_importable("bool_")),
    (jnp.uint8, _make_jnp_importable("uint8")),
    (jnp.uint16, _make_jnp_importable("uint16")),
    (jnp.uint32, _make_jnp_importable("uint32")),
    (jnp.uint64, _make_jnp_importable("uint64")),
    (jnp.int8, _make_jnp_importable("int8")),
    (jnp.int16, _make_jnp_importable("int16")),
    (jnp.int32, _make_jnp_importable("int32")),
    (jnp.int64, _make_jnp_importable("int64")),
    (jnp.bfloat16, _make_jnp_importable("bfloat16")),
    (jnp.float16, _make_jnp_importable("float16")),
    (jnp.float32, _make_jnp_importable("float32")),
    (jnp.float64, _make_jnp_importable("float64")),
    (jnp.complex64, _make_jnp_importable("complex64")),
    (jnp.complex128, _make_jnp_importable("complex128")),
)

_import_aliases = (
    # Rewrite internal import for JAX initializers.
    ("jax._src.nn.initializers",
     mini_ast.FromImport(name="initializers", parent="jax.nn")),
    (
        "jax.numpy",
        mini_ast.FromImportAs(name="jnp", parent="jax", module="numpy"),
    ),
)


def _make_jax_nn_importable(name: str) -> special_value_codegen.Importable:
  return special_value_codegen.SingleImportable(
      "jax", lambda jax_mod_name: f"{jax_mod_name}.nn.{name}")


_nn_type_importables = (
    (jax.nn.relu, _make_jax_nn_importable("relu")),
    (jax.nn.gelu, _make_jax_nn_importable("gelu")),
    (jax.nn.relu6, _make_jax_nn_importable("relu6")),
    (jax.nn.silu, _make_jax_nn_importable("silu")),
    (jax.nn.soft_sign, _make_jax_nn_importable("soft_sign")),
    (jax.nn.sigmoid, _make_jax_nn_importable("sigmoid")),
    (jax.nn.selu, _make_jax_nn_importable("selu")),
    (jax.nn.log_sigmoid, _make_jax_nn_importable("log_sigmoid")),
    (jax.nn.hard_tanh, _make_jax_nn_importable("hard_tanh")),
    (jax.nn.hard_swish, _make_jax_nn_importable("hard_swish")),
    (jax.nn.hard_silu, _make_jax_nn_importable("hard_silu")),
    (jax.nn.tanh, _make_jax_nn_importable("tanh")),
    (jax.nn.swish, _make_jax_nn_importable("swish")),
)


def is_jnp_device_array(value):
  """Returns true if `value` is a JAX numpy `DeviceArray`."""
  return isinstance(value, jnp.DeviceArray)


def convert_jnp_device_array_to_ast(value, convert_child):
  return ast.Call(
      func=ast.Attribute(value=convert_child(jnp), attr="array"),
      args=[convert_child(value.tolist())],
      keywords=[
          # Explicitly specify dtype (can't always be inferred from elts).
          ast.keyword(arg="dtype", value=convert_child(value.dtype.name)),
          # Explicitly specify rank (in case value.size==0).
          ast.keyword(arg="ndmin", value=convert_child(value.ndim)),
      ])


def enable():
  """Registers JAX fiddle extensions.

  This allows for things like nicer handling of jax.numpy dtypes.
  """
  for value, importable in _jnp_type_importables:
    special_value_codegen.register_exact_value(value, importable)

  for value, importable in _nn_type_importables:
    special_value_codegen.register_exact_value(value, importable)

  for module_str, import_stmt in _import_aliases:
    codegen.register_import_alias(module_str, import_stmt)

  # The odd calling syntax here ("register(type)(handler)") comes from the fact
  # that register_converter is usually a decorator, but we call it directly.
  py_val_to_ast_converter.register_py_val_to_ast_converter(is_jnp_device_array)(
      convert_jnp_device_array_to_ast)
