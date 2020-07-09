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
"""A base class for ResetWorld tests for a server."""

import abc

from absl.testing import absltest

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error


class ResetWorld(absltest.TestCase, metaclass=abc.ABCMeta):
  """A base class for dm_env_rpc `ResetWorld` compliance tests."""

  @property
  @abc.abstractmethod
  def connection(self):
    """An instance of dm_env_rpc's Connection already joined to a world."""
    pass

  @property
  def required_reset_world_settings(self):
    """Settings necessary to pass to ResetWorld."""
    return {}

  @property
  def required_join_world_settings(self):
    """Settings necessary to pass to JoinWorld."""
    return {}

  @property
  def invalid_world_name(self):
    """The name of a world which doesn't exist."""
    return 'invalid_world_name'

  @property
  @abc.abstractmethod
  def world_name(self):
    """The name of the world to attempt to call ResetWorld on."""
    return ''

  def join_world(self):
    """Joins the world to call ResetWorld on."""
    self.connection.send(dm_env_rpc_pb2.JoinWorldRequest(
        world_name=self.world_name, settings=self.required_join_world_settings))

  def reset_world(self, world_name):
    """Resets the world."""
    self.connection.send(dm_env_rpc_pb2.ResetWorldRequest(
        world_name=world_name, settings=self.required_reset_world_settings))

  def leave_world(self):
    """Leaves the world."""
    self.connection.send(dm_env_rpc_pb2.LeaveWorldRequest())

  # pylint: disable=missing-docstring
  def test_cannot_reset_invalid_world(self):
    with self.assertRaises(error.DmEnvRpcError):
      self.reset_world(self.invalid_world_name)

  def test_can_reset_world_not_joined_to(self):
    self.reset_world(self.world_name)
    # If there are no errors the test passes.

  def test_can_reset_world_when_joined_to_it(self):
    try:
      self.join_world()
      self.reset_world(self.world_name)
      # If there are no errors the test passes.
    finally:
      self.leave_world()
  # pylint: enable=missing-docstring
