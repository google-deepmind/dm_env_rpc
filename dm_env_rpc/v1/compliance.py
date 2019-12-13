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

from absl.testing import absltest

from dm_env_rpc.v1 import dm_env_rpc_pb2


class StepComplianceTestCase(absltest.TestCase, metaclass=abc.ABCMeta):
  """A base class for compliance tests."""

  @property
  @abc.abstractmethod
  def connection(self):
    """An instance of dm_env_rpc's Connection already joined to a world."""
    pass

  def step(self, **kwargs):
    """Sends a StepRequest and returns the observations."""
    return self.connection.send(dm_env_rpc_pb2.StepRequest(**kwargs))

  def test_no_observations_returned_if_not_requested(self):
    observations = self.step().observations
    self.assertEqual({}, observations)
