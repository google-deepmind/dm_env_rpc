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
"""A base class for Reset tests for a server."""

import abc

from absl.testing import absltest

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error


class Reset(absltest.TestCase, metaclass=abc.ABCMeta):
  """A base class for dm_env_rpc `Reset` compliance tests."""

  @property
  @abc.abstractmethod
  def connection(self):
    """An instance of dm_env_rpc's Connection already joined to a world."""
    pass

  @property
  def required_reset_settings(self):
    return {}

  @abc.abstractmethod
  def join_world(self):
    """Joins a world, returning the specs."""
    pass

  def reset(self):
    """Resets the environment, returning the specs."""
    return self.connection.send(dm_env_rpc_pb2.ResetRequest(
        settings=self.required_reset_settings)).specs

  # pylint: disable=missing-docstring
  def test_reset_resends_the_specs(self):
    join_specs = self.join_world()
    specs = self.reset()
    self.assertEqual(join_specs, specs)

  def test_cannot_reset_if_not_joined_to_world(self):
    with self.assertRaises(error.DmEnvRpcError):
      self.reset()

  def test_can_reset_multiple_times(self):
    join_specs = self.join_world()
    self.reset()
    specs = self.reset()
    self.assertEqual(join_specs, specs)
  # pylint: enable=missing-docstring
