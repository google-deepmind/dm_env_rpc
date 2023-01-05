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

from typing import Any, Mapping, NamedTuple, Optional, Sequence

import dm_env
import immutabledict

from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_flatten_utils
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_utils
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import spec_manager
from dm_env_rpc.v1 import tensor_utils


# Default observation names for common RL concepts.  By default the dm_env
# wrapper will use these for reward and discount if available, but this behavior
# can be overridden.
DEFAULT_REWARD_KEY = 'reward'
DEFAULT_DISCOUNT_KEY = 'discount'

# Default key separator, used in flattening/unflattening nested structures.
DEFAULT_KEY_SEPARATOR = '.'

# Format string for error message used in DmEnvAdaptor.Reset
_RESET_ENVIRONMENT_ERROR = r'''Environment changed spec after reset.
before: "{specs}"
after: "{new_specs}"'''


class DmEnvAdaptor(dm_env.Environment):
  """An implementation of dm_env using dm_env_rpc as the data protocol.

  Users can also optionally provide a mapping of objects to DmEnvAdaptor
  attributes. This is to accommodate user-created protocol extensions that
  compliment the core protocol.
  """

  # Disable pytype attribute checking for dynamically created extension attrs.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(
      self,
      connection: dm_env_rpc_connection.ConnectionType,
      specs: dm_env_rpc_pb2.ActionObservationSpecs,
      requested_observations: Optional[Sequence[str]] = None,
      nested_tensors: bool = True,
      extensions: Mapping[str, Any] = immutabledict.immutabledict()):
    """Initializes the environment with the provided dm_env_rpc connection.

    Args:
      connection: An instance of ConnectionType already connected to a
        dm_env_rpc server and after a successful JoinWorldRequest has been sent.
      specs: A dm_env_rpc ActionObservationSpecs message for the environment.
      requested_observations: List of observation names to be requested from the
        environment when step is called. If None is specified then all
        observations will be requested.
      nested_tensors: Boolean to determine whether to flatten/unflatten tensors.
      extensions: Mapping of extension instances to DmEnvAdaptor attributes.
        Raises ValueError if attribute already exists.
    """
    self._dm_env_rpc_specs = specs
    self._action_specs = spec_manager.SpecManager(specs.actions)
    self._observation_specs = spec_manager.SpecManager(specs.observations)
    self._connection = connection
    self._last_state = dm_env_rpc_pb2.EnvironmentStateType.TERMINATED
    self._nested_tensors = nested_tensors

    if requested_observations is None:
      requested_observations = self._observation_specs.names()
      self._is_reward_requested = False
      self._is_discount_requested = False
    else:
      self._is_reward_requested = DEFAULT_REWARD_KEY in requested_observations
      self._is_discount_requested = (
          DEFAULT_DISCOUNT_KEY in requested_observations)

    requested_observations = set(requested_observations)

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

    self._extension_names = extensions.keys()
    for extension_name, extension in extensions.items():
      if hasattr(self, extension_name):
        raise ValueError(
            f'DmEnvAdaptor already has attribute "{extension_name}"!')
      setattr(self, extension_name, extension)

  def reset(self):
    """Implements dm_env.Environment.reset."""
    reset_response = self._connection.send(dm_env_rpc_pb2.ResetRequest())
    if self._dm_env_rpc_specs != reset_response.specs:
      raise RuntimeError(_RESET_ENVIRONMENT_ERROR.format(
          specs=self._dm_env_rpc_specs, new_specs=reset_response.specs))
    self._last_state = dm_env_rpc_pb2.EnvironmentStateType.INTERRUPTED
    return self.step({})

  def step(self, actions):
    """Implements dm_env.Environment.step."""
    actions = dm_env_flatten_utils.flatten_dict(
        actions, DEFAULT_KEY_SEPARATOR) if self._nested_tensors else actions
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
      # Neither response.state nor _last_state is RUNNING.
      # See common causes for state transition errors:
      # https://github.com/deepmind/dm_env_rpc/blob/master/docs/v1/appendix.md#common-state-transition-errors
      raise RuntimeError('Environment transitioned from {} to {}'.format(
          dm_env_rpc_pb2.EnvironmentStateType.Name(self._last_state),
          dm_env_rpc_pb2.EnvironmentStateType.Name(step_response.state)))

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
    observations = dm_env_flatten_utils.unflatten_dict(
        observations,
        DEFAULT_KEY_SEPARATOR) if self._nested_tensors else observations
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

    if self._nested_tensors:
      return dm_env_flatten_utils.unflatten_dict(specs, DEFAULT_KEY_SEPARATOR)
    else:
      return specs

  def action_spec(self):
    """Implements dm_env.Environment.action_spec."""
    action_spec = dm_env_utils.dm_env_spec(self._action_specs)
    if self._nested_tensors:
      return dm_env_flatten_utils.unflatten_dict(action_spec,
                                                 DEFAULT_KEY_SEPARATOR)
    else:
      return action_spec

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
    # Release any extensions associated with this EnvAdaptor:
    for extension_name in self._extension_names:
      setattr(self, extension_name, None)
    # Leaves the world if we were joined.  If not, this will be a no-op anyway.
    if self._connection is not None:
      self._connection.send(dm_env_rpc_pb2.LeaveWorldRequest())
      self._connection = None


def create_world(connection: dm_env_rpc_connection.ConnectionType,
                 create_world_settings: Mapping[str, Any]) -> str:
  """Helper function to create a world with the provided settings.

  Args:
    connection: An instance of Connection already connected to a dm_env_rpc
      server.
    create_world_settings: Settings used to create the world. Nested settings
      will be automatically flattened before sending to the server. Values must
      be packable into a Tensor proto or already packed.

  Returns:
    Created world name.
  """
  create_world_settings = dm_env_flatten_utils.flatten_dict(
      create_world_settings, DEFAULT_KEY_SEPARATOR, strict=False
  )

  create_world_settings = {
      key: (value if isinstance(value, dm_env_rpc_pb2.Tensor) else
            tensor_utils.pack_tensor(value))
      for key, value in create_world_settings.items()
  }
  return connection.send(
      dm_env_rpc_pb2.CreateWorldRequest(
          settings=create_world_settings)).world_name


def join_world(
    connection: dm_env_rpc_connection.ConnectionType,
    world_name: str,
    join_world_settings: Mapping[str, Any],
    **adaptor_kwargs,
) -> DmEnvAdaptor:
  """Helper function to join a world with the provided settings.

  Args:
    connection: An instance of Connection already connected to a dm_env_rpc
      server.
    world_name: Name of the world to join.
    join_world_settings: Settings used to join the world. Nested settings will
      be automatically flattened before sending to the server. Values must be
      packable into a Tensor message or already packed.
    **adaptor_kwargs: Additional keyword args used to create the DmEnvAdaptor
      instance.

  Returns:
    Instance of DmEnvAdaptor.
  """
  join_world_settings = dm_env_flatten_utils.flatten_dict(
      join_world_settings, DEFAULT_KEY_SEPARATOR, strict=False
  )

  join_world_settings = {
      key: (value if isinstance(value, dm_env_rpc_pb2.Tensor) else
            tensor_utils.pack_tensor(value))
      for key, value in join_world_settings.items()
  }
  specs = connection.send(
      dm_env_rpc_pb2.JoinWorldRequest(
          world_name=world_name, settings=join_world_settings)).specs

  try:
    return DmEnvAdaptor(connection, specs, **adaptor_kwargs)
  except ValueError:
    connection.send(dm_env_rpc_pb2.LeaveWorldRequest())
    raise


class DmEnvAndWorldName(NamedTuple):
  """Environment and world_name created when calling create_and_join_world."""
  env: DmEnvAdaptor
  world_name: str


def create_and_join_world(connection: dm_env_rpc_connection.ConnectionType,
                          create_world_settings: Mapping[str, Any],
                          join_world_settings: Mapping[str, Any],
                          **adaptor_kwargs) -> DmEnvAndWorldName:
  """Helper function to create and join a world with the provided settings.

  Args:
    connection: An instance of Connection already connected to a dm_env_rpc
      server.
    create_world_settings: Settings used to create the world. Values must be
      packable into a Tensor proto or already packed.
    join_world_settings: Settings used to join the world. Nested settings will
      be automatically flattened before sending to the server. Values must be
      packable into a Tensor message.
    **adaptor_kwargs: Additional keyword args used to create the DmEnvAdaptor
      instance.

  Returns:
    Tuple of DmEnvAdaptor and the created world name.
  """
  world_name = create_world(connection, create_world_settings)
  try:
    return DmEnvAndWorldName(
        join_world(connection, world_name, join_world_settings,
                   **adaptor_kwargs), world_name)
  except (error.DmEnvRpcError, ValueError):
    connection.send(dm_env_rpc_pb2.DestroyWorldRequest(world_name=world_name))
    raise
