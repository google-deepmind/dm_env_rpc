# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Utilities for interfacing dm_env and dm_env_rpc."""

from dm_env import specs
import numpy as np

from dm_env_rpc.v1 import tensor_utils


def _np_range_info(np_type):
  """Returns type info for `np_type`, which includes min and max attributes."""
  if issubclass(np_type, np.floating):
    return np.finfo(np_type)
  elif issubclass(np_type, np.integer):
    return np.iinfo(np_type)
  else:
    raise ValueError('{} does not have range info.'.format(np_type))


def _find_extreme(tensor_spec, tensor_type, extremal_name):
  """Finds the min or max value the tensor spec sets."""
  explicit_extreme = None
  if tensor_spec.HasField(extremal_name):
    explicit_extreme = getattr(tensor_spec, extremal_name)
  if explicit_extreme is not None:
    value_field = explicit_extreme.WhichOneof('value')
    if value_field is None:
      raise ValueError(
          'Tensor spec had {} present but no value was given.'.format(
              extremal_name))
    if getattr(np, value_field) != tensor_type:
      raise ValueError(
          'Tensor spec had {}.{} set, but tensor has type {}.'.format(
              extremal_name, value_field, tensor_type))
    return getattr(explicit_extreme, value_field)
  else:
    return getattr(_np_range_info(tensor_type), extremal_name)


def tensor_spec_to_dm_env_spec(tensor_spec):
  """Returns the dm_env Array or BoundedArray given a dm_env_rpc TensorSpec."""
  tensor_type = tensor_utils.data_type_to_np_type(tensor_spec.dtype)
  if tensor_spec.HasField('min') or tensor_spec.HasField('max'):
    return specs.BoundedArray(
        shape=tensor_spec.shape,
        dtype=tensor_type,
        name=tensor_spec.name,
        minimum=_find_extreme(tensor_spec, tensor_type, 'min'),
        maximum=_find_extreme(tensor_spec, tensor_type, 'max'))
  else:
    return specs.Array(
        shape=tensor_spec.shape, dtype=tensor_type, name=tensor_spec.name)


def dm_env_spec(spec_manager):
  """Returns a dm_env spec for the given `spec_manager`.

  Args:
    spec_manager: An instance of SpecManager.

  Returns:
    A dict mapping names to either dm_env ArraySpecs or BoundedArraySpecs for
    each named TensorSpec in `spec_manager`.
  """
  return {
      name: tensor_spec_to_dm_env_spec(spec_manager.name_to_spec(name))
      for name in spec_manager.names()
  }
