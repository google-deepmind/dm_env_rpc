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
"""An implementation of a dm_env environment using dm_env_rpc."""

import dm_env

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_utils
from dm_env_rpc.v1 import spec_manager

# Default observation names for common RL concepts.  By default the dm_env
# wrapper will use these for reward and discount if available, but this behavior
# can be overridden.
DEFAULT_REWARD_KEY = 'reward'
DEFAULT_DISCOUNT_KEY = 'discount'


class DmEnvAdaptor(dm_env.Environment):
  """An implementation of dm_env using dm_env_rpc as the data protocol."""

  def __init__(self, connection, specs, requested_observations=None):
    """Initializes the environment with the provided dm_env_rpc connection.

    Args:
      connection: An instance of Connection already connected to a dm_env_rpc
        server and after a successful JoinWorldRequest has been sent.
      specs: A dm_env_rpc ActionObservationSpecs message for the environment.
      requested_observations: The observation names to be requested from the
        environment when step is called. If None is specified then all
        observations will be requested.
    """
    self._dm_env_rpc_specs = specs
    self._action_specs = spec_manager.SpecManager(specs.actions)
    self._observation_specs = spec_manager.SpecManager(specs.observations)
    self._connection = connection
    self._last_state = dm_env_rpc_pb2.EnvironmentStateType.TERMINATED

    if requested_observations is None:
      requested_observations = self._observation_specs.names()
    requested_observations = set(requested_observations)

    self._is_reward_requested = DEFAULT_REWARD_KEY in requested_observations
    self._is_discount_requested = DEFAULT_DISCOUNT_KEY in requested_observations

    self._default_reward_spec = None
    self._default_discount_spec = None
    if DEFAULT_REWARD_KEY in self._observation_specs.names():
      self._default_reward_spec = dm_env_utils.tensor_spec_to_dm_env_spec(
          self._observation_specs.name_to_spec(DEFAULT_REWARD_KEY))
      requested_observations.add(DEFAULT_REWARD_KEY)
    if DEFAULT_DISCOUNT_KEY in self._observation_specs.names():
      self._default_discount_spec = (
          dm_env_utils.tensor_spec_to_dm_env_spec(
              self._observation_specs.name_to_spec(DEFAULT_DISCOUNT_KEY)))
      requested_observations.add(DEFAULT_DISCOUNT_KEY)

    unsupported_observations = requested_observations.difference(
        self._observation_specs.names())
    if unsupported_observations:
      raise ValueError('Unsupported observations requested: {}'.format(
          unsupported_observations))
    self._requested_observation_uids = [
        self._observation_specs.name_to_uid(name)
        for name in requested_observations
    ]

    # Not strictly necessary but it makes the unit tests deterministic.
    self._requested_observation_uids.sort()

  def reset(self):
    """Implements dm_env.Environment.reset."""
    response = self._connection.send(dm_env_rpc_pb2.ResetRequest())
    if self._dm_env_rpc_specs != response.specs:
      raise RuntimeError('Environment changed spec after reset')
    self._last_state = dm_env_rpc_pb2.EnvironmentStateType.INTERRUPTED
    return self.step({})

  def step(self, actions):
    """Implements dm_env.Environment.step."""
    step_response = self._connection.send(
        dm_env_rpc_pb2.StepRequest(
            requested_observations=self._requested_observation_uids,
            actions=self._action_specs.pack(actions)))

    observations = self._observation_specs.unpack(step_response.observations)

    if (step_response.state == dm_env_rpc_pb2.EnvironmentStateType.RUNNING and
        self._last_state == dm_env_rpc_pb2.EnvironmentStateType.RUNNING):
      step_type = dm_env.StepType.MID
    elif step_response.state == dm_env_rpc_pb2.EnvironmentStateType.RUNNING:
      step_type = dm_env.StepType.FIRST
    elif self._last_state == dm_env_rpc_pb2.EnvironmentStateType.RUNNING:
      step_type = dm_env.StepType.LAST
    else:
      raise RuntimeError('Environment transitioned from {} to {}'.format(
          self._last_state, step_response.state))

    self._last_state = step_response.state

    reward = self.reward(
        state=step_response.state,
        step_type=step_type,
        observations=observations)
    discount = self.discount(
        state=step_response.state,
        step_type=step_type,
        observations=observations)
    if not self._is_reward_requested:
      observations.pop(DEFAULT_REWARD_KEY, None)
    if not self._is_discount_requested:
      observations.pop(DEFAULT_DISCOUNT_KEY, None)
    return dm_env.TimeStep(step_type, reward, discount, observations)

  def reward(self, state, step_type, observations):
    """Returns the reward for the given observation state.

    Override in inherited classes to give different reward functions.

    Args:
      state: A dm_env_rpc EnvironmentStateType enum describing the state of the
        environment.
      step_type: The dm_env StepType describing the state of the environment.
      observations: The unpacked observations dictionary mapping string keys to
        scalars and NumPy arrays.

    Returns:
      A reward for the given step.  The shape and type matches that returned by
      `self.reward_spec()`.
    """
    if step_type == dm_env.StepType.FIRST:
      return None
    elif self._default_reward_spec:
      return observations[DEFAULT_REWARD_KEY]
    else:
      return 0.0

  def discount(self, state, step_type, observations):
    """Returns the discount for the given observation state.

    Override in inherited classes to give different discount functions.

    Args:
      state: A dm_env_rpc EnvironmentStateType enum describing the state of the
        environment.
      step_type: The dm_env StepType describing the state of the environment.
      observations: The unpacked observations dictionary mapping string keys to
        scalars and NumPy arrays.

    Returns:
      The discount for the given step.  The shape and type matches that returned
      by `self.discount_spec()`.
    """
    if self._default_discount_spec:
      return observations[DEFAULT_DISCOUNT_KEY]
    if step_type == dm_env.StepType.FIRST:
      return None
    elif (state == dm_env_rpc_pb2.EnvironmentStateType.RUNNING or
          state == dm_env_rpc_pb2.EnvironmentStateType.INTERRUPTED):
      return 1.0
    else:
      return 0.0

  def observation_spec(self):
    """Implements dm_env.Environment.observation_spec."""
    specs = {}
    for uid in self._requested_observation_uids:
      name = self._observation_specs.uid_to_name(uid)
      specs[name] = dm_env_utils.tensor_spec_to_dm_env_spec(
          self._observation_specs.uid_to_spec(uid))
    if not self._is_reward_requested:
      specs.pop(DEFAULT_REWARD_KEY, None)
    if not self._is_discount_requested:
      specs.pop(DEFAULT_DISCOUNT_KEY, None)
    return specs

  def action_spec(self):
    """Implements dm_env.Environment.action_spec."""
    return dm_env_utils.dm_env_spec(self._action_specs)

  def reward_spec(self):
    """Implements dm_env.Environment.reward_spec."""
    return (self._default_reward_spec or
            super(DmEnvAdaptor, self).reward_spec())

  def discount_spec(self):
    """Implements dm_env.Environment.discount_spec."""
    return (self._default_discount_spec or
            super(DmEnvAdaptor, self).discount_spec())

  def close(self):
    """Implements dm_env.Environment.close."""
    # Leaves the world if we were joined.  If not, this will be a no-op anyway.
    self._connection.send(dm_env_rpc_pb2.LeaveWorldRequest())
    self._connection = None
