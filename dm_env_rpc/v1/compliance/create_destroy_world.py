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
"""A base class for CreateWorld and DestroyWorld tests for a server."""
import abc

from absl.testing import absltest

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error


class CreateDestroyWorld(absltest.TestCase, metaclass=abc.ABCMeta):
  """A base class for `CreateWorld` and `DestroyWorld` compliance tests."""

  @abc.abstractproperty
  def required_world_settings(self):
    """A string to Tensor mapping of the minimum set of required settings."""
    pass

  @abc.abstractproperty
  def invalid_world_settings(self):
    """World creation settings which are invalid in some way."""
    pass

  @abc.abstractproperty
  def has_multiple_world_support(self):
    """Does the server support creating more than one world?"""
    pass

  @abc.abstractproperty
  def connection(self):
    """An instance of dm_env_rpc's Connection."""
    pass

  def create_world(self, settings):
    """Returns the world name of the world created with the given settings."""
    response = self.connection.send(
        dm_env_rpc_pb2.CreateWorldRequest(settings=settings))
    return response.world_name

  def destroy_world(self, world_name):
    """Destroys the world named `world_name`."""
    if world_name is not None:
      self.connection.send(
          dm_env_rpc_pb2.DestroyWorldRequest(world_name=world_name))

  # pylint: disable=missing-docstring
  def test_can_create_and_destroy_world(self):
    # If this doesn't raise an exception the test passes.
    world_name = self.create_world(self.required_world_settings)
    self.destroy_world(world_name)

  def test_cannot_create_world_with_less_than_required_settings(self):
    settings = self.required_world_settings

    for name, _ in settings.items():
      sans_setting = dict(settings)
      del sans_setting[name]
      with self.assertRaises(error.DmEnvRpcError):
        self.create_world(sans_setting)

  def test_cannot_create_world_with_invalid_settings(self):
    settings = self.required_world_settings
    invalid_settings = self.invalid_world_settings
    for name, tensor in invalid_settings.items():
      with self.assertRaises(error.DmEnvRpcError):
        self.create_world({name: tensor, **settings})

  def test_world_name_is_unique(self):
    if not self.has_multiple_world_support:
      return
    world1_name = None
    world2_name = None
    try:
      world1_name = self.create_world(self.required_world_settings)
      world2_name = self.create_world(self.required_world_settings)
      self.assertNotNone(world1_name)
      self.assertNotNone(world2_name)
      self.assertNotEqual(world1_name, world2_name)
    finally:
      self.destroy_world(world1_name)
      self.destroy_world(world2_name)

  def test_cannot_destroy_uncreated_world(self):
    with self.assertRaises(error.DmEnvRpcError):
      self.destroy_world('foo')
  # pylint: enable=missing-docstring
