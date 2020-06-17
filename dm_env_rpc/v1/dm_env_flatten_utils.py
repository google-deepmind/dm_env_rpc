# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Python utilities for flattening and unflattening key-value mappings."""

import collections
from typing import Any, Dict, Mapping


def flatten_dict(input_dict: Mapping[str, Any],
                 separator: str) -> Dict[str, Any]:
  """Flattens mappings by joining sub-keys using the provided separator.

  Only non-empty, mapping types will be flattened. All other types are deemed
  leaf values.

  Args:
    input_dict: Mapping of key-value pairs to flatten.
    separator: Delimiter used to concatenate keys.

  Returns:
    Flattened dictionary of key-value pairs.

  Raises:
    ValueError: If the `input_dict` has a key that contains the separator
      string.
  """
  result = collections.OrderedDict()
  for key, value in input_dict.items():
    if separator in key:
      raise ValueError(f"Key '{key}' already contains separator!")
    if isinstance(value, Mapping) and len(value):
      result.update({
          f'{key}{separator}{sub_key}': sub_value
          for sub_key, sub_value in flatten_dict(value, separator).items()
      })
    else:
      result[key] = value
  return result


def unflatten_dict(input_dict: Mapping[str, Any],
                   separator: str) -> Dict[str, Any]:
  """Unflatten dictionary using split keys to determine the structure.

  For each key, split based on the provided separator and create nested
  dictionary entry for each sub-key.

  Args:
    input_dict: Mapping of key-value pairs to un-flatten.
    separator: Delimiter used to split keys.

  Returns:
    Unflattened dictionary.

  Raises:
    ValueError: If a key, or it's constituent sub-keys already has a value. For
      instance, unflattening `{"foo": True, "foo.bar": "baz"}` will result in
      "foo" being set to both a dict and a Bool.
  """
  result = collections.OrderedDict()
  for key, value in input_dict.items():
    sub_keys = key.split(separator)
    sub_tree = result
    for sub_key in sub_keys[:-1]:
      sub_tree = sub_tree.setdefault(sub_key, {})
      if not isinstance(sub_tree, Mapping):
        raise ValueError(f"Sub-tree '{sub_key}' has already been assigned a "
                         "leaf value {sub_tree}")

    if sub_keys[-1] in sub_tree:
      raise ValueError(f'Duplicate key {key}')
    sub_tree[sub_keys[-1]] = value
  return result
