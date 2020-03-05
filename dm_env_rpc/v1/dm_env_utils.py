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

from dm_env_rpc.v1 import tensor_spec_utils
from dm_env_rpc.v1 import tensor_utils


def tensor_spec_to_dm_env_spec(tensor_spec):
  """Returns a dm_env spec given a dm_env_rpc TensorSpec.

  Args:
    tensor_spec: A dm_env_rpc TensorSpec protobuf.

  Returns:
    Either a DiscreteArray, BoundedArray or Array, depending on the content of
    the TensorSpec.
  """
  np_type = tensor_utils.data_type_to_np_type(tensor_spec.dtype)
  if tensor_spec.HasField('min') or tensor_spec.HasField('max'):
    bounds = tensor_spec_utils.bounds(tensor_spec)

    if (not tensor_spec.shape
        and np.issubdtype(np_type, np.integer)
        and bounds.min == 0
        and tensor_spec.HasField('max')):
      return specs.DiscreteArray(
          num_values=bounds.max + 1, dtype=np_type, name=tensor_spec.name)
    else:
      return specs.BoundedArray(
          shape=tensor_spec.shape,
          dtype=np_type,
          name=tensor_spec.name,
          minimum=bounds.min,
          maximum=bounds.max)
  else:
    return specs.Array(
        shape=tensor_spec.shape, dtype=np_type, name=tensor_spec.name)


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
