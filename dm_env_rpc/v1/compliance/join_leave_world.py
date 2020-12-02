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
"""A base class for JoinWorld and LeaveWord tests for a server."""
import abc

from absl.testing import absltest
import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import tensor_spec_utils


def _find_duplicates(iterable):
  """Returns a list of duplicate entries found in `iterable`."""
  duplicates = []
  seen = set()
  for item in iterable:
    if item in seen:
      duplicates.append(item)
    else:
      seen.add(item)
  return duplicates


def _check_tensor_spec(tensor_spec):
  """Raises an error if the given `tensor_spec` is internally inconsistent."""
  if np.sum(np.asarray(tensor_spec.shape) < 0) > 1:
    raise ValueError(
        f'"{tensor_spec.name}" has shape {tensor_spec.shape} which has more '
        'than one negative element.')
  min_type = tensor_spec.min and tensor_spec.min.WhichOneof('payload')
  max_type = tensor_spec.max and tensor_spec.max.WhichOneof('payload')
  if min_type or max_type:
    _ = tensor_spec_utils.bounds(tensor_spec)


class JoinLeaveWorld(absltest.TestCase, metaclass=abc.ABCMeta):
  """A base class for `JoinWorld` and `LeaveWorld` compliance tests."""

  @property
  def required_join_settings(self):
    """A dict of required settings for a Join World call."""
    return {}

  @property
  def invalid_join_settings(self):
    """A list of dicts of Join World settings which are invalid in some way."""
    return {}

  @abc.abstractproperty
  def world_name(self):
    """A string of the world name of an already created world."""
    pass

  @property
  def invalid_world_name(self):
    """A string which doesn't correspond to any valid world_name."""
    return 'invalid_world_name'

  @property
  @abc.abstractmethod
  def connection(self):
    """An instance of dm_env_rpc's Connection."""
    pass

  def tearDown(self):
    super().tearDown()
    try:
      self.leave_world()
    finally:
      pass

  def join_world(self, **kwargs):
    """Joins the world and returns the spec."""
    response = self.connection.send(dm_env_rpc_pb2.JoinWorldRequest(**kwargs))
    return response.specs

  def leave_world(self):
    """Leaves currently joined world, if any."""
    self.connection.send(dm_env_rpc_pb2.LeaveWorldRequest())

  # pylint: disable=missing-docstring
  def test_can_join(self):
    self.join_world(
        world_name=self.world_name, settings=self.required_join_settings)
    # Success if there's no error raised.

  def test_cannot_join_with_wrong_world_name(self):
    with self.assertRaises(error.DmEnvRpcError):
      self.join_world(world_name=self.invalid_world_name)

  def test_cannot_join_world_with_invalid_settings(self):
    settings = self.required_join_settings
    for name, tensor in self.invalid_join_settings.items():
      with self.assertRaises(error.DmEnvRpcError):
        self.join_world(
            world_name=self.world_name, settings={
                name: tensor,
                **settings
            })

  def test_cannot_join_world_twice(self):
    self.join_world(
        world_name=self.world_name, settings=self.required_join_settings)
    with self.assertRaises(error.DmEnvRpcError):
      self.join_world(
          world_name=self.world_name, settings=self.required_join_settings)

  def test_action_specs_have_unique_names(self):
    specs = self.join_world(
        world_name=self.world_name, settings=self.required_join_settings)
    self.assertEmpty(_find_duplicates(
        spec.name for spec in specs.actions.values()))

  def test_action_specs_for_consistency(self):
    specs = self.join_world(
        world_name=self.world_name, settings=self.required_join_settings)
    for action_spec in specs.actions.values():
      _check_tensor_spec(action_spec)

  def test_observation_specs_have_unique_names(self):
    specs = self.join_world(
        world_name=self.world_name, settings=self.required_join_settings)
    self.assertEmpty(_find_duplicates(
        spec.name for spec in specs.observations.values()))

  def test_observation_specs_for_consistency(self):
    specs = self.join_world(
        world_name=self.world_name, settings=self.required_join_settings)
    for observation_spec in specs.observations.values():
      _check_tensor_spec(observation_spec)

  def test_can_leave_world_if_not_joined(self):
    self.leave_world()
    # Success if there's no error raised.

  def test_can_leave_world_after_joining(self):
    self.join_world(
        world_name=self.world_name, settings=self.required_join_settings)
    self.leave_world()
    # Success if there's no error raised.

  def test_can_rejoin_world_after_leaving(self):
    self.join_world(
        world_name=self.world_name, settings=self.required_join_settings)
    self.leave_world()
    self.join_world(
        world_name=self.world_name, settings=self.required_join_settings)
    # Success if there's no error raised.

  # pylint: enable=missing-docstring
