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
"""A base class for Step tests for a server."""

import abc
import functools
import operator

from absl.testing import absltest
import numpy as np

from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1 import tensor_spec_utils
from dm_env_rpc.v1 import tensor_utils


def _find_uid_not_in_set(uid_set):
  """Finds an example UID not in `uid_set`."""
  uids = set(uid_set)
  uid = 0
  while uid in uids:
    uid = uid + 1
  return uid


def _is_numeric_type(dtype):
  return (dtype != dm_env_rpc_pb2.DataType.PROTO and
          np.issubdtype(tensor_utils.data_type_to_np_type(dtype), np.number))


def _assert_less_equal(x, y, err_msg='', verbose=True):
  np.testing.assert_array_compare(
      operator.__le__, x, y, err_msg=err_msg, verbose=verbose,
      header='Arrays are not less or equal ordered', equal_inf=False)


def _assert_greater_equal(x, y, err_msg='', verbose=True):
  np.testing.assert_array_compare(
      operator.__ge__, x, y, err_msg=err_msg, verbose=verbose,
      header='Arrays are not greater or equal ordered', equal_inf=False)


def _create_test_value(spec, dtype=None):
  """Creates a NumPy array test value consistent with the TensorSpec `spec`."""
  if _is_numeric_type(spec.dtype):
    value = tensor_spec_utils.bounds(spec).min
  else:
    value = tensor_utils.data_type_to_np_type(spec.dtype)()
  shape = np.asarray(spec.shape)
  shape[shape < 0] = 1
  return np.full(shape=shape, fill_value=value, dtype=dtype)


def _create_test_tensor(spec, dtype=None):
  """Creates an arbitrary tensor consistent with the TensorSpec `spec`."""
  value = _create_test_value(spec, dtype)
  return tensor_utils.pack_tensor(value)


def _np_range_info(np_type):
  """Returns type info for `np_type`, which includes min and max attributes."""
  if issubclass(np_type, np.floating):
    return np.finfo(np_type)
  elif issubclass(np_type, np.integer):
    return np.iinfo(np_type)
  else:
    raise ValueError('{} does not have range info.'.format(np_type))


def _below_min(spec):
  """Generates values below spec's min.

  Args:
    spec: An instance of `TensorSpec`.

  Yields:
    A sequence of tuples of `(value, index)` where `index` is an indexer into
    `value` where the element has been set below the spec's min.
  """
  if not spec.HasField('min'):
    return

  np_type = tensor_utils.data_type_to_np_type(spec.dtype)
  min_type_value = _np_range_info(np_type).min
  minimum = tensor_spec_utils.bounds(spec).min

  for index in np.ndindex(*spec.shape):
    if min_type_value < minimum[index]:
      value = _create_test_value(spec)
      value[index] = min_type_value
      yield value, index


def _above_max(spec):
  """Generates values above spec's max.

  Args:
    spec: An instance of `TensorSpec`.

  Yields:
    A sequence of tuples of `(value, index)` where `index` is an indexer into
    `value` where the element has been set above the spec's max.
  """
  if not spec.HasField('max'):
    return

  np_type = tensor_utils.data_type_to_np_type(spec.dtype)
  max_type_value = _np_range_info(np_type).max
  maximum = tensor_spec_utils.bounds(spec).max

  for index in np.ndindex(*spec.shape):
    if max_type_value > maximum[index]:
      value = _create_test_value(spec)
      value[index] = max_type_value
      yield value, index


def _find_scalar_within_bounds(spec):
  """Returns a scalar which can satisfy the spec's bounds for any element."""
  if _is_numeric_type(spec.dtype):
    bounds = tensor_spec_utils.bounds(spec)
    hard_min = np.amax(bounds.min)
    hard_max = np.amin(bounds.max)
    if hard_min < hard_max:
      return hard_min
    else:
      # There is no single scalar value that can satisfy all bounds.
      return None
  else:
    np_type = tensor_utils.data_type_to_np_type(spec.dtype)
    return np_type()


def _step_before_test(function):
  """Decorator which calls step before test function is run."""
  @functools.wraps(function)
  def wrapper(self, *args, **kwargs):
    # First step's actions are ignored, so step once to start the sequence.
    step_response = self.step()
    self.assertEqual(
        step_response.state, dm_env_rpc_pb2.EnvironmentStateType.RUNNING)
    return function(self, *args, **kwargs)
  return wrapper


class Step(absltest.TestCase, metaclass=abc.ABCMeta):
  """A base class for dm_env_rpc `Step` compliance tests."""

  @property
  @abc.abstractmethod
  def connection(self):
    """An instance of dm_env_rpc's Connection already joined to a world."""
    pass

  @property
  @abc.abstractmethod
  def specs(self):
    """The specs from a JoinWorldResponse."""
    pass

  @property
  def observation_uids(self):
    return set(self.specs.observations.keys())

  @property
  def action_uids(self):
    return set(self.specs.actions.keys())

  @property
  def required_actions(self):
    """A dict of required actions for a Step call."""
    return {}

  @property
  def numeric_actions(self):
    return {uid: spec for uid, spec in self.specs.actions.items()
            if _is_numeric_type(spec.dtype)}

  @property
  def nonnumeric_actions(self):
    return {uid: spec for uid, spec in self.specs.actions.items()
            if not _is_numeric_type(spec.dtype)}

  def step(self, actions=None, **kwargs):
    """Sends a StepRequest and returns the StepResponse."""
    actions = {**self.required_actions, **(actions or {})}
    return self.connection.send(
        dm_env_rpc_pb2.StepRequest(actions=actions, **kwargs))

  # pylint: disable=missing-docstring
  ##############################################################################
  # Observations
  ##############################################################################
  def test_no_observations_returned_if_not_requested(self):
    observations = self.step().observations
    self.assertEqual({}, observations)

  def test_requested_observations_are_returned(self):
    response = self.step(requested_observations=self.observation_uids)
    observations = response.observations
    self.assertEqual(self.observation_uids, set(observations.keys()))

  def test_cannot_request_invalid_observation_uid(self):
    bad_uid = _find_uid_not_in_set(self.observation_uids)
    with self.assertRaisesRegex(error.DmEnvRpcError, str(bad_uid)):
      self.step(requested_observations=[bad_uid])

  def test_all_observation_dtypes_match_spec_dtypes(self):
    response = self.step(requested_observations=self.observation_uids)
    for uid, observation in response.observations.items():
      spec = self.specs.observations[uid]
      with self.subTest(uid=uid, name=spec.name):
        spec_type = tensor_utils.data_type_to_np_type(spec.dtype)
        tensor_type = tensor_utils.get_tensor_type(observation)
        self.assertEqual(spec_type, tensor_type)

  def test_all_numerical_observations_in_range(self):
    numeric_uids = (uid for uid, spec in self.specs.observations.items()
                    if _is_numeric_type(spec.dtype))
    response = self.step(requested_observations=numeric_uids)
    for uid, observation in response.observations.items():
      spec = self.specs.observations[uid]
      with self.subTest(uid=uid, name=spec.name):
        unpacked = tensor_utils.unpack_tensor(observation)
        bounds = tensor_spec_utils.bounds(spec)
        if spec.max.WhichOneof('payload') is not None:
          _assert_less_equal(unpacked, bounds.max)
        if spec.min.WhichOneof('payload') is not None:
          _assert_greater_equal(unpacked, bounds.min)

  def test_duplicated_requested_observations_are_redundant(self):
    response = self.step(requested_observations=list(self.observation_uids) * 2)
    self.assertEqual(self.observation_uids, set(response.observations.keys()))

  def test_can_request_each_observation_individually(self):
    for uid in self.observation_uids:
      spec = self.specs.observations[uid]
      with self.subTest(uid=uid, name=spec.name):
        response = self.step(requested_observations=[uid])
        self.assertEqual([uid], list(response.observations.keys()))

  ##############################################################################
  # Actions
  ##############################################################################
  def test_first_step_actions_are_ignored(self):
    bad_uid = _find_uid_not_in_set(self.action_uids)
    self.step(actions={bad_uid: tensor_utils.pack_tensor(0)})

  @_step_before_test
  def test_can_send_each_action_individually(self):
    for uid, spec in self.specs.actions.items():
      with self.subTest(uid=uid, name=spec.name):
        tensor = _create_test_tensor(spec)
        self.step(actions={uid: tensor})

  @_step_before_test
  def test_cannot_send_wrong_numeric_type_action(self):
    for uid, spec in self.numeric_actions.items():
      with self.subTest(uid=uid, name=spec.name):
        np_type = tensor_utils.data_type_to_np_type(spec.dtype)
        wrong_dtype = (np.float64 if not np.issubdtype(np_type, np.float64)
                       else np.float32)
        tensor = _create_test_tensor(spec, dtype=wrong_dtype)
        with self.assertRaises(error.DmEnvRpcError):
          self.step(actions={uid: tensor})

  @_step_before_test
  def test_cannot_send_wrong_type_to_nonnumeric_actions(self):
    tensor = tensor_utils.pack_tensor(0, dtype=np.int32)
    for uid, spec in self.nonnumeric_actions.items():
      with self.subTest(uid=uid, name=spec.name):
        shape = np.asarray(spec.shape)
        shape[shape < 0] = 1
        tensor.shape[:] = shape
        with self.assertRaises(error.DmEnvRpcError):
          self.step(actions={uid: tensor})

  @_step_before_test
  def test_cannot_send_invalid_action_uid(self):
    bad_uid = _find_uid_not_in_set(self.action_uids)
    with self.assertRaises(error.DmEnvRpcError):
      self.step(actions={bad_uid: tensor_utils.pack_tensor(0)})

  @_step_before_test
  def test_cannot_send_action_below_min(self):
    for uid, spec in self.numeric_actions.items():
      with self.subTest(uid=uid, name=spec.name):
        shape = np.asarray(spec.shape)
        shape[shape < 0] = 1
        for value, index in _below_min(spec):
          with self.subTest(below_min_index=index):
            tensor = tensor_utils.pack_tensor(value, dtype=spec.dtype)
            tensor.shape[:] = shape
            with self.assertRaises(error.DmEnvRpcError):
              self.step(actions={uid: tensor})

  @_step_before_test
  def test_cannot_send_action_above_max(self):
    for uid, spec in self.numeric_actions.items():
      with self.subTest(uid=uid, name=spec.name):
        shape = np.asarray(spec.shape)
        shape[shape < 0] = 1
        for value, index in _above_max(spec):
          with self.subTest(above_max_index=index):
            tensor = tensor_utils.pack_tensor(value, dtype=spec.dtype)
            tensor.shape[:] = shape
            with self.assertRaises(error.DmEnvRpcError):
              self.step(actions={uid: tensor})

  @_step_before_test
  def test_cannot_send_action_with_wrong_shape(self):
    for uid, spec in self.specs.actions.items():
      with self.subTest(uid=uid, name=spec.name):
        tensor = _create_test_tensor(spec)
        # Add too many dimensions to shape.
        tensor.shape[:] = tensor.shape[:] + [1]
        with self.assertRaises(error.DmEnvRpcError):
          self.step(actions={uid: tensor})

  @_step_before_test
  def test_can_send_variable_dimension_tensor_action(self):
    actions_with_shape = {uid: spec for uid, spec in self.specs.actions.items()
                          if spec.shape}
    for uid, spec in actions_with_shape.items():
      with self.subTest(uid=uid, name=spec.name):
        tensor = _create_test_tensor(spec)
        # Set first dimension to be variable.
        tensor.shape[0] = -1
        self.step(actions={uid: tensor})

  @_step_before_test
  def test_cannot_send_tensor_with_too_many_variable_dimensions(self):
    actions_with_multidimensional_shape = {
        uid: spec for uid, spec in self.specs.actions.items()
        if len(spec.shape) >= 2}
    for uid, spec in actions_with_multidimensional_shape.items():
      with self.subTest(uid=uid, name=spec.name):
        tensor = _create_test_tensor(spec)
        # Set multiple variable dimensions.
        tensor.shape[0] = -1
        tensor.shape[1] = -1
        with self.assertRaises(error.DmEnvRpcError):
          self.step(actions={uid: tensor})

  @_step_before_test
  def test_can_send_broadcastable_actions(self):
    for uid, spec in self.specs.actions.items():
      with self.subTest(uid=uid, name=spec.name):
        scalar = _find_scalar_within_bounds(spec)
        if scalar is None:
          # The action has no scalars we could feasibly broadcast.
          continue
        tensor = tensor_utils.pack_tensor(scalar, dtype=spec.dtype)
        shape = np.asarray(spec.shape)
        shape[shape < 0] = 1
        tensor.shape[:] = shape
        self.step(actions={uid: tensor})
  # pylint: enable=missing-docstring
