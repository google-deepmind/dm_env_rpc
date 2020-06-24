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
"""Tests for CatchEnvironment."""

from concurrent import futures

from absl.testing import absltest
from dm_env import test_utils
import grpc
import numpy as np

import catch_environment
from dm_env_rpc.v1 import compliance
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc
from dm_env_rpc.v1 import error


class ServerConnection:

  def __init__(self):
    self._server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1))
    servicer = catch_environment.CatchEnvironmentService()
    dm_env_rpc_pb2_grpc.add_EnvironmentServicer_to_server(
        servicer, self._server)
    port = self._server.add_secure_port('[::]:0',
                                        grpc.local_server_credentials())
    self._server.start()

    self._channel = grpc.secure_channel(f'[::]:{port}',
                                        grpc.local_channel_credentials())
    grpc.channel_ready_future(self._channel).result()

    self.connection = dm_env_rpc_connection.Connection(self._channel)

  def close(self):
    self.connection.close()
    self._channel.close()
    self._server.stop(grace=None)


class JoinedServerConnection(ServerConnection):

  def __init__(self):
    super().__init__()
    response = self.connection.send(dm_env_rpc_pb2.CreateWorldRequest())
    self.world_name = response.world_name

    response = self.connection.send(dm_env_rpc_pb2.JoinWorldRequest(
        world_name=self.world_name))
    self.specs = response.specs

  def close(self):
    try:
      self.connection.send(dm_env_rpc_pb2.LeaveWorldRequest())
      self.connection.send(
          dm_env_rpc_pb2.DestroyWorldRequest(world_name=self.world_name))
    finally:
      super().close()


class CatchDmEnvRpcStepTest(compliance.Step):

  @property
  def connection(self):
    return self._server_connection.connection

  @property
  def specs(self):
    return self._server_connection.specs

  def setUp(self):
    super().setUp()
    self._server_connection = JoinedServerConnection()

  def tearDown(self):
    self._server_connection.close()
    super().tearDown()


class CatchDmEnvRpcCreateAndDestoryWorldTest(compliance.CreateDestroyWorld):

  @property
  def connection(self):
    return self._server_connection.connection

  @property
  def required_world_settings(self):
    """A string to Tensor mapping of the minimum set of required settings."""
    return {}

  @property
  def invalid_world_settings(self):
    """World creation settings which are invalid in some way."""
    return {}

  @property
  def has_multiple_world_support(self):
    """Does the server support creating more than one world?"""
    return False

  def setUp(self):
    self._server_connection = ServerConnection()
    super().setUp()

  def tearDown(self):
    super().tearDown()
    self._server_connection.close()


class CatchDmEnvRpcJoinAndLeaveWorldTest(compliance.JoinLeaveWorld):

  @property
  def connection(self):
    return self._server_connection.connection

  @property
  def world_name(self):
    return self._world_name

  def setUp(self):
    self._server_connection = ServerConnection()
    response = self.connection.send(dm_env_rpc_pb2.CreateWorldRequest())
    self._world_name = response.world_name
    super().setUp()

  def tearDown(self):
    super().tearDown()
    try:
      self.connection.send(
          dm_env_rpc_pb2.DestroyWorldRequest(world_name=self.world_name))
    finally:
      self._server_connection.close()


class CatchDmEnvRpcResetTest(compliance.Reset):

  @property
  def connection(self):
    return self._server_connection.connection

  def join_world(self):
    """Joins a world, returning the specs."""
    response = self.connection.send(dm_env_rpc_pb2.JoinWorldRequest(
        world_name=self.world_name))
    return response.specs

  @property
  def world_name(self):
    return self._world_name

  def setUp(self):
    self._server_connection = ServerConnection()
    response = self.connection.send(dm_env_rpc_pb2.CreateWorldRequest())
    self._world_name = response.world_name
    super().setUp()

  def tearDown(self):
    super().tearDown()
    try:
      self.connection.send(dm_env_rpc_pb2.LeaveWorldRequest())
      self.connection.send(
          dm_env_rpc_pb2.DestroyWorldRequest(world_name=self.world_name))
    finally:
      self._server_connection.close()


class CatchDmEnvTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def setUp(self):
    self._server_connection = JoinedServerConnection()
    self._connection = self._server_connection.connection
    self.world_name = self._server_connection.world_name

    self._dm_env = dm_env_adaptor.DmEnvAdaptor(
        self._connection, self._server_connection.specs)
    super().setUp()

  def tearDown(self):
    super().tearDown()
    self._server_connection.close()

  def make_object_under_test(self):
    return self._dm_env


class CatchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self._server_connection = ServerConnection()
    self._connection = self._server_connection.connection

    response = self._connection.send(dm_env_rpc_pb2.CreateWorldRequest())
    self._world_name = response.world_name

  def tearDown(self):
    try:
      self._connection.send(dm_env_rpc_pb2.LeaveWorldRequest())
      self._connection.send(
          dm_env_rpc_pb2.DestroyWorldRequest(world_name=self._world_name))
    finally:
      self._server_connection.close()
    super().tearDown()

  def test_can_reset_world_when_joined(self):
    self._connection.send(
        dm_env_rpc_pb2.JoinWorldRequest(world_name=self._world_name))
    self._connection.send(dm_env_rpc_pb2.ResetWorldRequest())

  def test_cannot_reset_world_when_not_joined(self):
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(dm_env_rpc_pb2.ResetWorldRequest())

  def test_cannot_step_when_not_joined(self):
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(dm_env_rpc_pb2.StepRequest())

  def test_cannot_reset_when_not_joined(self):
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(dm_env_rpc_pb2.ResetRequest())

  def test_cannot_join_world_with_wrong_name(self):
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(
          dm_env_rpc_pb2.JoinWorldRequest(world_name='wrong_name'))

  def test_cannot_create_world_when_world_exists(self):
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(dm_env_rpc_pb2.CreateWorldRequest())

  def test_cannot_join_when_no_world_exists(self):
    self._connection.send(
        dm_env_rpc_pb2.DestroyWorldRequest(world_name=self._world_name))
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(
          dm_env_rpc_pb2.JoinWorldRequest(world_name=self._world_name))
    self._connection.send(dm_env_rpc_pb2.CreateWorldRequest())

  def test_cannot_destroy_world_when_still_joined(self):
    self._connection.send(
        dm_env_rpc_pb2.JoinWorldRequest(world_name=self._world_name))
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(
          dm_env_rpc_pb2.DestroyWorldRequest(world_name=self._world_name))

  def test_cannot_destroy_world_with_wrong_name(self):
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(
          dm_env_rpc_pb2.DestroyWorldRequest(world_name='wrong_name'))

  def test_read_property_request_is_not_supported(self):
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(dm_env_rpc_pb2.ReadPropertyRequest())

  def test_write_property_request_is_not_supported(self):
    with self.assertRaises(error.DmEnvRpcError):
      self._connection.send(dm_env_rpc_pb2.WritePropertyRequest())


class CatchGameTest(absltest.TestCase):

  def setUp(self):
    super(CatchGameTest, self).setUp()
    self._rows = 3
    self._cols = 3
    self._game = catch_environment.CatchGame(self._rows, self._cols, 1)

  def test_draw_board_correct_initial_state(self):
    board = self._game.draw_board()
    self.assertEqual(board.shape, (3, 3))

  def test_draw_board_ball_in_top_row(self):
    board = self._game.draw_board()
    self.assertIn(1, board[0])

  def test_draw_board_bat_in_center_bottom_row(self):
    board = self._game.draw_board()
    self.assertTrue(np.array_equal([0, 1, 0], board[2]))

  def test_update_drops_ball(self):
    self._game.update(action=0)
    board = self._game.draw_board()
    self.assertNotIn(1, board[0])
    self.assertIn(1, board[1])

  def test_has_terminated_when_ball_hits_bottom(self):
    self.assertFalse(self._game.has_terminated())
    self._game.update(action=0)
    self.assertFalse(self._game.has_terminated())
    self._game.update(action=0)
    self.assertTrue(self._game.has_terminated())

  def test_update_moves_paddle(self):
    self._game.update(action=1)
    board = self._game.draw_board()
    self.assertTrue(np.array_equal([0, 0, 1], board[2]))

  def test_cannot_update_game_when_has_terminated(self):
    self._game.update(action=0)
    self._game.update(action=0)
    with self.assertRaises(RuntimeError):
      self._game.update(action=0)

  def test_no_reward_when_not_terminated(self):
    self.assertEqual(0, self._game.reward())
    self._game.update(action=0)
    self.assertEqual(0, self._game.reward())
    self._game.update(action=0)

  def test_has_reward_when_terminated(self):
    self._game.update(action=0)
    self._game.update(action=0)
    self.assertNotEqual(0, self._game.reward())

if __name__ == '__main__':
  absltest.main()
