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
"""Catch example implemented as a gRPC EnvironmentServicer."""

import numpy as np

from google.rpc import status_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc
from dm_env_rpc.v1 import spec_manager

_ACTION_PADDLE = 'paddle'
_DEFAULT_ACTION = 0
_INITIAL_SEED = 1
_NUM_ROWS = 10
_NUM_COLUMNS = 10
_OBSERVATION_REWARD = 'reward'
_OBSERVATION_BOARD = 'board'
_WORLD_NAME = 'catch'


class CatchGame(object):
  """Simple Catch game environment.

  The agent must move a paddle to intercept falling balls. Falling balls only
  move downwards on the column they are in.

  The observation is an array shape (rows, columns), with binary values:
  zero if a space is empty; 1 if it contains the paddle or a ball.

  The actions are discrete, and there are three available: stay, move left
  and move right.

  The rewards adjusted when the ball reaches the bottom of the screen.
  """

  def __init__(self, rows, columns, seed):
    """Initializes a new Catch environment.

    Args:
      rows: number of rows.
      columns: number of columns.
      seed: random seed for the RNG.
    """
    self._rows = rows
    self._columns = columns
    self._ball_x = np.random.RandomState(seed).randint(self._columns)
    self._ball_y = 0
    self._paddle_x = self._columns // 2
    self._paddle_y = self._rows - 1

  def draw_board(self):
    """Draw the board into a numpy array and return it."""
    board = np.zeros((self._rows, self._columns), dtype=np.float32)
    board[self._ball_y, self._ball_x] = 1.
    board[self._paddle_y, self._paddle_x] = 1.
    return board

  def update(self, action):
    """Updates the environment according to the action."""
    if self.has_terminated():
      raise RuntimeError('Trying to update terminated environment')

    # Move the paddle.
    self._paddle_x = np.clip(self._paddle_x + action, 0, self._columns - 1)

    # Drop the ball.
    self._ball_y += 1

  def has_terminated(self):
    return self._ball_y == self._paddle_y

  def reward(self):
    """Provides the incremental reward for the current frame."""
    if self.has_terminated():
      return 1. if self._paddle_x == self._ball_x else -1.
    else:
      return 0


def _check_message_type(env, is_joined, message_type):
  """Checks the message type is valid given the environment's world state."""

  if not env:
    if message_type not in ['create_world', 'leave_world']:
      raise RuntimeError('Cannot {} when no world exists.'.format(message_type))
  else:
    if message_type == 'create_world':
      raise RuntimeError(
          'This example does not support creating multiple worlds.')
    if is_joined:
      if message_type == 'destroy_world':
        raise RuntimeError('Cannot destroy world when still joined.')
    else:
      if message_type == 'reset_world':
        raise RuntimeError(
            'This example does not support reset_world when not joined.')
      elif message_type in ['step', 'reset']:
        raise RuntimeError(
            'Cannot {} when world not joined.'.format(message_type))


def _observation_spec():
  """Returns the observation spec."""
  return {
      1:
          dm_env_rpc_pb2.TensorSpec(
              name=_OBSERVATION_BOARD,
              shape=[_NUM_ROWS, _NUM_COLUMNS],
              dtype=dm_env_rpc_pb2.FLOAT),
      2:
          dm_env_rpc_pb2.TensorSpec(
              name=_OBSERVATION_REWARD, dtype=dm_env_rpc_pb2.FLOAT)
  }


def _action_spec():
  """Returns the action spec."""
  return {
      1:
          dm_env_rpc_pb2.TensorSpec(
              dtype=dm_env_rpc_pb2.INT8, name=_ACTION_PADDLE)
  }


class CatchGameFactory(object):
  """Factory for creating new CatchGame instances."""

  def __init__(self, initial_seed):
    self._seed = initial_seed

  def new_game(self):
    env = CatchGame(rows=_NUM_ROWS, columns=_NUM_COLUMNS, seed=self._seed)
    self._seed += 1
    return env


class CatchEnvironmentService(dm_env_rpc_pb2_grpc.EnvironmentServicer):
  """Runs the Catch game as a gRPC EnvironmentServicer."""

  def Process(self, request_iterator, context):
    """Processes incoming EnvironmentRequests.

    For each EnvironmentRequest the internal message is extracted and handled.
    The response for that message is then placed in a EnvironmentResponse which
    is returned to the client.

    An error status will be returned if an unknown message type is received or
    if the message is invalid for the current world state.


    Args:
      request_iterator: Message iterator provided by gRPC.
      context: Context provided by gRPC.

    Yields:
      EnvironmentResponse: Response for each incoming EnvironmentRequest.
    """

    env_factory = CatchGameFactory(_INITIAL_SEED)
    env = None
    is_joined = False
    skip_next_frame = False
    action_manager = spec_manager.SpecManager(_action_spec())
    observation_manager = spec_manager.SpecManager(_observation_spec())

    for request in request_iterator:
      environment_response = dm_env_rpc_pb2.EnvironmentResponse()
      try:
        message_type = request.WhichOneof('payload')
        internal_request = getattr(request, message_type)
        _check_message_type(env, is_joined, message_type)

        if message_type == 'create_world':
          env = env_factory.new_game()
          skip_next_frame = True
          response = dm_env_rpc_pb2.CreateWorldResponse(world_name=_WORLD_NAME)
        elif message_type == 'join_world':
          if internal_request.world_name != _WORLD_NAME:
            raise RuntimeError(
                'Tried to join world "{}" but only support world "{}"'.format(
                    internal_request.world_name, _WORLD_NAME))
          response = dm_env_rpc_pb2.JoinWorldResponse()
          for uid, action in _action_spec().items():
            response.specs.actions[uid].CopyFrom(action)
          for uid, observation in _observation_spec().items():
            response.specs.observations[uid].CopyFrom(observation)
          is_joined = True
        elif message_type == 'step':
          # We need to skip all actions after creating or resetting the
          # environment.
          if skip_next_frame:
            skip_next_frame = False
          else:
            unpacked_actions = action_manager.unpack(internal_request.actions)
            paddle_action = unpacked_actions.get(_ACTION_PADDLE,
                                                 _DEFAULT_ACTION)
            env.update(paddle_action)

          response = dm_env_rpc_pb2.StepResponse()
          packed_observations = observation_manager.pack({
              _OBSERVATION_BOARD: env.draw_board(),
              _OBSERVATION_REWARD: env.reward()
          })

          for requested_observation in internal_request.requested_observations:
            response.observations[requested_observation].CopyFrom(
                packed_observations[requested_observation])
          if env.has_terminated():
            response.state = dm_env_rpc_pb2.EnvironmentStateType.TERMINATED
          else:
            response.state = dm_env_rpc_pb2.EnvironmentStateType.RUNNING

          if env.has_terminated():
            env = env_factory.new_game()
            skip_next_frame = True
        elif message_type == 'reset':
          env = env_factory.new_game()
          skip_next_frame = True
          response = dm_env_rpc_pb2.ResetResponse()
          for uid, action in _action_spec().items():
            response.specs.actions[uid].CopyFrom(action)
          for uid, observation in _observation_spec().items():
            response.specs.observations[uid].CopyFrom(observation)
        elif message_type == 'reset_world':
          env = env_factory.new_game()
          skip_next_frame = True
          response = dm_env_rpc_pb2.ResetWorldResponse()
        elif message_type == 'leave_world':
          is_joined = False
          response = dm_env_rpc_pb2.LeaveWorldResponse()
        elif message_type == 'destroy_world':
          if internal_request.world_name != _WORLD_NAME:
            raise RuntimeError(
                'Tried to destroy world "{}" but we only support world "{}"'
                .format(internal_request.world_name, _WORLD_NAME))
          env = None
          response = dm_env_rpc_pb2.DestroyWorldResponse()
        else:
          raise RuntimeError('Unhandled message: {}'.format(message_type))
        getattr(environment_response, message_type).CopyFrom(response)
      except RuntimeError as e:
        environment_response.error.CopyFrom(status_pb2.Status(message=str(e)))

      yield environment_response
