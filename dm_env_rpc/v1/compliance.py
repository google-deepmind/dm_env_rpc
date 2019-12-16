# Lint as: python3
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
"""A compliance mixin for testing that a server complies with dm_env_rpc."""

import abc
import operator

from absl.testing import absltest
import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import tensor_spec_utils
from dm_env_rpc.v1 import tensor_utils


def _find_uid_not_in_list(uid_list):
  """Finds an example UID not in `uid_list`."""
  uids = set(uid_list)
  uid = 0
  while uid in uids:
    uid = uid + 1
  return uid


def _is_numeric_type(dtype):
  return (dtype != dm_env_rpc_pb2.DataType.PROTO and
          np.issubdtype(tensor_utils.data_type_to_np_type(dtype), np.number))


def _assert_less_equal(x, y, err_msg='', verbose=True):
  np.testing.assert_array_compare(
      operator.__le__, x, y, err_msg=err_msg, verbose=verbose,
      header='Arrays are not less or equal ordered', equal_inf=False)


def _assert_greater_equal(x, y, err_msg='', verbose=True):
  np.testing.assert_array_compare(
      operator.__ge__, x, y, err_msg=err_msg, verbose=verbose,
      header='Arrays are not greater or equal ordered', equal_inf=False)


class StepComplianceTestCase(absltest.TestCase, metaclass=abc.ABCMeta):
  """A base class for compliance tests."""

  @property
  @abc.abstractmethod
  def connection(self):
    """An instance of dm_env_rpc's Connection already joined to a world."""
    pass

  @property
  @abc.abstractmethod
  def specs(self):
    """The specs from a JoinWorldResponse."""
    pass

  @property
  def observation_uids(self):
    return list(self.specs.observations.keys())

  def step(self, **kwargs):
    """Sends a StepRequest and returns the observations."""
    return self.connection.send(dm_env_rpc_pb2.StepRequest(**kwargs))

  # pylint: disable=missing-docstring
  def test_no_observations_returned_if_not_requested(self):
    observations = self.step().observations
    self.assertEqual({}, observations)

  def test_requested_observations_are_returned(self):
    response = self.step(requested_observations=self.observation_uids)
    observations = response.observations
    self.assertEqual(self.observation_uids, list(observations.keys()))

  def test_cannot_request_invalid_observation_uid(self):
    bad_uid = _find_uid_not_in_list(self.observation_uids)
    with self.assertRaisesRegex(ValueError, str(bad_uid)):
      self.step(requested_observations=[bad_uid])

  def test_all_observation_dtypes_match_spec_dtypes(self):
    response = self.step(requested_observations=self.observation_uids)
    for uid, observation in response.observations.items():
      spec = self.specs.observations[uid]
      spec_type = tensor_utils.data_type_to_np_type(spec.dtype)
      tensor_type = tensor_utils.get_tensor_type(observation)
      self.assertEqual(spec_type, tensor_type,
                       '"{}" has spec type {} but actual tensor has type {}.'
                       .format(spec.name, spec_type, tensor_type))

  def test_all_numerical_observations_in_range(self):
    numeric_uids = (uid for uid, spec in self.specs.observations.items()
                    if _is_numeric_type(spec.dtype))
    response = self.step(requested_observations=numeric_uids)
    for uid, observation in response.observations.items():
      spec = self.specs.observations[uid]
      unpacked = tensor_utils.unpack_tensor(observation)
      bounds = tensor_spec_utils.bounds(spec)
      message = '"{}" has value {} which is outside bounds [{}, {}].'.format(
          spec.name, unpacked, bounds.min, bounds.max)
      _assert_less_equal(unpacked, bounds.max, message)
      _assert_greater_equal(unpacked, bounds.min, message)

  def test_duplicated_requested_observations_are_redundant(self):
    response = self.step(requested_observations=self.observation_uids * 2)
    self.assertEqual(set(self.observation_uids),
                     set(response.observations.keys()))

  def test_can_request_each_observation_individually(self):
    for uid in self.observation_uids:
      response = self.step(requested_observations=[uid])
      self.assertEqual([uid], list(response.observations.keys()))
  # pylint: enable=missing-docstring

