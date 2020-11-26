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
"""Example Catch human agent."""

from concurrent import futures

from absl import app
import grpc
import pygame

import catch_environment
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc

_FRAMES_PER_SEC = 3
_FRAME_DELAY_MS = int(1000.0 // _FRAMES_PER_SEC)

_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)

_ACTION_LEFT = -1
_ACTION_NOTHING = 0
_ACTION_RIGHT = 1

_ACTION_PADDLE = 'paddle'
_OBSERVATION_REWARD = 'reward'
_OBSERVATION_BOARD = 'board'


def _draw_row(row_str, row_index, standard_font, window_surface):
  text = standard_font.render(row_str, True, _WHITE)
  text_rect = text.get_rect()
  text_rect.left = 50
  text_rect.top = 30 + (row_index * 30)
  window_surface.blit(text, text_rect)


def _render_window(board, window_surface, reward):
  """Render the game onto the window surface."""

  standard_font = pygame.font.SysFont('Courier', 24)
  instructions_font = pygame.font.SysFont('Courier', 16)

  num_rows = board.shape[0]
  num_cols = board.shape[1]

  window_surface.fill(_BLACK)

  # Draw board.
  header = '* ' * (num_cols + 2)
  _draw_row(header, 0, standard_font, window_surface)
  for board_index in range(num_rows):
    row = board[board_index]
    row_str = '* '
    for c in row:
      row_str += 'x ' if c == 1. else '  '
    row_str += '* '
    _draw_row(row_str, board_index + 1, standard_font, window_surface)
  _draw_row(header, num_rows + 1, standard_font, window_surface)

  # Draw footer.
  reward_str = 'Reward: {}'.format(reward)
  _draw_row(reward_str, num_rows + 3, standard_font, window_surface)
  instructions = ('Instructions: Left/Right arrow keys to move paddle, Escape '
                  'to exit.')
  _draw_row(instructions, num_rows + 5, instructions_font, window_surface)


def _start_server():
  """Starts the Catch gRPC server."""
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  servicer = catch_environment.CatchEnvironmentService()
  dm_env_rpc_pb2_grpc.add_EnvironmentServicer_to_server(servicer, server)

  port = server.add_secure_port('localhost:0', grpc.local_server_credentials())
  server.start()
  return server, port


def main(_):
  pygame.init()

  server, port = _start_server()

  with dm_env_rpc_connection.create_secure_channel_and_connect(
      f'localhost:{port}') as connection:
    env, world_name = dm_env_adaptor.create_and_join_world(
        connection, create_world_settings={}, join_world_settings={})
    with env:
      window_surface = pygame.display.set_mode((800, 600), 0, 32)
      pygame.display.set_caption('Catch Human Agent')

      keep_running = True
      while keep_running:
        requested_action = _ACTION_NOTHING

        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            keep_running = False
            break
          elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
              requested_action = _ACTION_LEFT
            elif event.key == pygame.K_RIGHT:
              requested_action = _ACTION_RIGHT
            elif event.key == pygame.K_ESCAPE:
              keep_running = False
              break

        actions = {_ACTION_PADDLE: requested_action}
        step_result = env.step(actions)

        board = step_result.observation[_OBSERVATION_BOARD]
        reward = step_result.observation[_OBSERVATION_REWARD]

        _render_window(board, window_surface, reward)

        pygame.display.update()

        pygame.time.wait(_FRAME_DELAY_MS)

    connection.send(dm_env_rpc_pb2.DestroyWorldRequest(world_name=world_name))

  server.stop(None)


if __name__ == '__main__':
  app.run(main)
