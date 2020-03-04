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
"""Tests for dm_env_rpc/dm_env adaptor."""

from absl.testing import absltest
import dm_env
from dm_env import specs
import mock
import numpy as np

from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import tensor_utils

_SAMPLE_STEP_REQUEST = dm_env_rpc_pb2.StepRequest(
    requested_observations=[1, 2],
    actions={
        1: tensor_utils.pack_tensor(4, dtype=dm_env_rpc_pb2.UINT8),
        2: tensor_utils.pack_tensor('hello')
    })
_SAMPLE_STEP_RESPONSE = dm_env_rpc_pb2.StepResponse(
    state=dm_env_rpc_pb2.EnvironmentStateType.RUNNING,
    observations={
        1: tensor_utils.pack_tensor(5, dtype=dm_env_rpc_pb2.UINT8),
        2: tensor_utils.pack_tensor('goodbye')
    })
_TERMINATED_STEP_RESPONSE = dm_env_rpc_pb2.StepResponse(
    state=dm_env_rpc_pb2.EnvironmentStateType.TERMINATED,
    observations={
        1: tensor_utils.pack_tensor(5, dtype=dm_env_rpc_pb2.UINT8),
        2: tensor_utils.pack_tensor('goodbye')
    })
_SAMPLE_SPEC = dm_env_rpc_pb2.ActionObservationSpecs(
    actions={
        1: dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.UINT8, name='foo'),
        2: dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.STRING, name='bar')
    },
    observations={
        1: dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.UINT8, name='foo'),
        2: dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.STRING, name='bar')
    },
)
_SAMPLE_SPEC_REORDERED = dm_env_rpc_pb2.ActionObservationSpecs(
    observations={
        2: dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.STRING, name='bar'),
        1: dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.UINT8, name='foo')
    },
    actions={
        2: dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.STRING, name='bar'),
        1: dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.UINT8, name='foo')
    },
)
# Ensures the equality check in reset() works if the dictionary elements are
# created in a different order.
_SAMPLE_RESET_RESPONSE = dm_env_rpc_pb2.ResetResponse(
    specs=_SAMPLE_SPEC_REORDERED)
_RESET_CHANGES_SPEC_RESPONSE = dm_env_rpc_pb2.ResetResponse(
    specs=dm_env_rpc_pb2.ActionObservationSpecs())

_RESERVED_SPEC = dm_env_rpc_pb2.ActionObservationSpecs(
    actions={},
    observations={
        1:
            dm_env_rpc_pb2.TensorSpec(
                dtype=dm_env_rpc_pb2.UINT8,
                name=dm_env_adaptor.DEFAULT_REWARD_KEY),
        2:
            dm_env_rpc_pb2.TensorSpec(
                dtype=dm_env_rpc_pb2.STRING,
                name=dm_env_adaptor.DEFAULT_DISCOUNT_KEY)
    })
_RESERVED_STEP_RESPONSE = dm_env_rpc_pb2.StepResponse(
    state=dm_env_rpc_pb2.EnvironmentStateType.RUNNING,
    observations={
        1: tensor_utils.pack_tensor(5, dtype=dm_env_rpc_pb2.UINT8),
        2: tensor_utils.pack_tensor('goodbye')
    })


class DmEnvAdaptorTests(absltest.TestCase):

  def setUp(self):
    super(DmEnvAdaptorTests, self).setUp()
    self._connection = mock.MagicMock()
    self._env = dm_env_adaptor.DmEnvAdaptor(self._connection, _SAMPLE_SPEC)

  def test_requested_observations(self):
    requested_observations = ['foo']
    filtered_env = dm_env_adaptor.DmEnvAdaptor(self._connection, _SAMPLE_SPEC,
                                               requested_observations)

    expected_filtered_step_request = dm_env_rpc_pb2.StepRequest(
        requested_observations=[1],
        actions={
            1: tensor_utils.pack_tensor(4, dtype=dm_env_rpc_pb2.UINT8),
            2: tensor_utils.pack_tensor('hello')
        })

    self._connection.send = mock.MagicMock(return_value=_SAMPLE_STEP_RESPONSE)
    filtered_env.step({'foo': 4, 'bar': 'hello'})

    self._connection.send.assert_called_once_with(
        expected_filtered_step_request)

  def test_invalid_requested_observations(self):
    requested_observations = ['invalid']
    with self.assertRaisesRegex(ValueError,
                                'Unsupported observations requested'):
      dm_env_adaptor.DmEnvAdaptor(self._connection, _SAMPLE_SPEC,
                                  requested_observations)

  def test_requested_observation_spec(self):
    requested_observations = ['foo']
    filtered_env = dm_env_adaptor.DmEnvAdaptor(self._connection, _SAMPLE_SPEC,
                                               requested_observations)
    observation_names = [name for name in filtered_env.observation_spec()]
    self.assertEqual(requested_observations, observation_names)

  def test_first_running_step(self):
    self._connection.send = mock.MagicMock(return_value=_SAMPLE_STEP_RESPONSE)
    timestep = self._env.step({'foo': 4, 'bar': 'hello'})

    self._connection.send.assert_called_once_with(_SAMPLE_STEP_REQUEST)
    self.assertEqual(dm_env.StepType.FIRST, timestep.step_type)
    self.assertEqual(None, timestep.reward)
    self.assertEqual(None, timestep.discount)
    self.assertEqual({'foo': 5, 'bar': 'goodbye'}, timestep.observation)

  def test_mid_running_step(self):
    self._connection.send = mock.MagicMock(return_value=_SAMPLE_STEP_RESPONSE)
    self._env.step({'foo': 4, 'bar': 'hello'})
    self._connection.send.assert_called_once_with(_SAMPLE_STEP_REQUEST)
    timestep = self._env.step({'foo': 4, 'bar': 'hello'})

    self.assertEqual(dm_env.StepType.MID, timestep.step_type)
    self.assertEqual(0.0, timestep.reward)
    self.assertEqual(1.0, timestep.discount)
    self.assertEqual({'foo': 5, 'bar': 'goodbye'}, timestep.observation)

  def test_last_step(self):
    self._connection.send = mock.MagicMock(return_value=_SAMPLE_STEP_RESPONSE)
    self._env.step({'foo': 4, 'bar': 'hello'})
    self._connection.send.assert_called_once_with(_SAMPLE_STEP_REQUEST)
    self._connection.send = mock.MagicMock(
        return_value=_TERMINATED_STEP_RESPONSE)
    timestep = self._env.step({'foo': 4, 'bar': 'hello'})

    self.assertEqual(dm_env.StepType.LAST, timestep.step_type)
    self.assertEqual(0.0, timestep.reward)
    self.assertEqual(0.0, timestep.discount)
    self.assertEqual({'foo': 5, 'bar': 'goodbye'}, timestep.observation)

  def test_illegal_state_transition(self):
    self._connection.send = mock.MagicMock(
        return_value=_TERMINATED_STEP_RESPONSE)
    with self.assertRaisesRegex(RuntimeError, 'Environment transitioned'):
      self._env.step({})

  def test_reset(self):
    self._connection.send = mock.MagicMock(return_value=_SAMPLE_STEP_RESPONSE)
    self._env.step({'foo': 4, 'bar': 'hello'})
    self._connection.send.assert_called_once_with(_SAMPLE_STEP_REQUEST)
    self._connection.send = mock.MagicMock(
        side_effect=[_SAMPLE_RESET_RESPONSE, _SAMPLE_STEP_RESPONSE])
    timestep = self._env.reset()

    self.assertEqual(dm_env.StepType.FIRST, timestep.step_type)
    self.assertEqual(None, timestep.reward)
    self.assertEqual(None, timestep.discount)
    self.assertEqual({'foo': 5, 'bar': 'goodbye'}, timestep.observation)

  def test_reset_changes_spec_raises_error(self):
    self._connection.send = mock.MagicMock(return_value=_SAMPLE_STEP_RESPONSE)
    self._env.step({'foo': 4, 'bar': 'hello'})
    self._connection.send.assert_called_once_with(_SAMPLE_STEP_REQUEST)
    self._connection.send = mock.MagicMock(
        side_effect=[_RESET_CHANGES_SPEC_RESPONSE, _SAMPLE_STEP_RESPONSE])
    with self.assertRaisesRegex(RuntimeError, 'changed spec'):
      self._env.reset()

  def test_observation_spec(self):
    expected_spec = {
        'foo': specs.Array(shape=(), dtype=np.uint8, name='foo'),
        'bar': specs.Array(shape=(), dtype=np.str_, name='bar')
    }
    self.assertEqual(expected_spec, self._env.observation_spec())

  def test_action_spec(self):
    expected_spec = {
        'foo': specs.Array(shape=(), dtype=np.uint8, name='foo'),
        'bar': specs.Array(shape=(), dtype=np.str_, name='bar')
    }
    self.assertEqual(expected_spec, self._env.action_spec())

  def test_cant_step_after_close(self):
    self._connection.send = mock.MagicMock(
        return_value=dm_env_rpc_pb2.LeaveWorldResponse())
    self._env.close()
    with self.assertRaisesRegex(AttributeError, 'send'):
      self._env.step({})

  def test_reward_spec_default(self):
    self.assertEqual(
        specs.Array(shape=(), dtype=np.float64), self._env.reward_spec())

  def test_discount_spec_default(self):
    self.assertEqual(
        specs.BoundedArray(
            shape=(), dtype=np.float64, minimum=0.0, maximum=1.0),
        self._env.discount_spec())

  def test_close_leaves_world(self):
    self._connection.send = mock.MagicMock(
        return_value=dm_env_rpc_pb2.LeaveWorldResponse())
    self._env.close()
    self._connection.send.assert_called_once_with(
        dm_env_rpc_pb2.LeaveWorldRequest())

  def test_close_errors_when_cannot_leave_world(self):
    self._connection.send = mock.MagicMock(side_effect=ValueError('foo'))
    with self.assertRaisesRegex(ValueError, 'foo'):
      self._env.close()


class OverrideRewardDiscount(dm_env_adaptor.DmEnvAdaptor):

  def __init__(self):
    self.connection = mock.MagicMock()
    self.reward = mock.MagicMock()
    self.discount = mock.MagicMock()
    super(OverrideRewardDiscount, self).__init__(self.connection, _SAMPLE_SPEC)


class RewardDiscountOverrideTests(absltest.TestCase):

  def test_override_reward(self):
    env = OverrideRewardDiscount()
    env.reward.return_value = 0.5
    env.connection.send.return_value = _SAMPLE_STEP_RESPONSE
    timestep = env.step({})
    self.assertEqual(0.5, timestep.reward)
    env.reward.assert_called()
    self.assertEqual(dm_env_rpc_pb2.EnvironmentStateType.RUNNING,
                     env.reward.call_args[1]['state'])
    self.assertEqual(dm_env.StepType.FIRST,
                     env.reward.call_args[1]['step_type'])
    self.assertDictEqual({
        'foo': 5,
        'bar': 'goodbye'
    }, env.reward.call_args[1]['observations'])

  def test_override_discount(self):
    env = OverrideRewardDiscount()
    env.discount.return_value = 0.5
    env.connection.send.return_value = _SAMPLE_STEP_RESPONSE
    timestep = env.step({})
    self.assertEqual(0.5, timestep.discount)
    env.discount.assert_called()
    self.assertEqual(dm_env_rpc_pb2.EnvironmentStateType.RUNNING,
                     env.discount.call_args[1]['state'])
    self.assertEqual(dm_env.StepType.FIRST,
                     env.discount.call_args[1]['step_type'])
    self.assertDictEqual({
        'foo': 5,
        'bar': 'goodbye'
    }, env.discount.call_args[1]['observations'])


class ReservedKeywordTests(absltest.TestCase):

  def setUp(self):
    super(ReservedKeywordTests, self).setUp()
    self._connection = mock.MagicMock()
    self._env = dm_env_adaptor.DmEnvAdaptor(self._connection, _RESERVED_SPEC)

  def test_reward_spec(self):
    self.assertEqual(
        specs.Array(shape=(), dtype=np.uint8), self._env.reward_spec())

  def test_discount_spec(self):
    self.assertEqual(
        specs.Array(shape=(), dtype=np.str_), self._env.discount_spec())

  def test_reward_from_reserved_keyword(self):
    self._connection.send = mock.MagicMock(return_value=_RESERVED_STEP_RESPONSE)
    self._env.step({})  # Reward is None for first step.
    timestep = self._env.step({})

    self.assertEqual(5, timestep.reward)

  def test_discount(self):
    self._connection.send = mock.MagicMock(return_value=_RESERVED_STEP_RESPONSE)
    timestep = self._env.step({})

    self.assertEqual('goodbye', timestep.discount)


class EnvironmentAutomaticallyRequestsReservedKeywords(absltest.TestCase):

  def setUp(self):
    super(EnvironmentAutomaticallyRequestsReservedKeywords, self).setUp()
    self._connection = mock.MagicMock()
    self._env = dm_env_adaptor.DmEnvAdaptor(
        self._connection, _RESERVED_SPEC, requested_observations=[])
    self._connection.send = mock.MagicMock(return_value=_RESERVED_STEP_RESPONSE)

  def test_reward_spec_unrequested(self):
    self.assertEqual(
        specs.Array(shape=(), dtype=np.uint8), self._env.reward_spec())

  def test_discount_spec_unrequested(self):
    self.assertEqual(
        specs.Array(shape=(), dtype=np.str_), self._env.discount_spec())

  def test_does_not_give_back_unrequested_observations(self):
    timestep = self._env.step({})
    self.assertEqual({}, timestep.observation)

  def test_first_reward_none(self):
    timestep = self._env.step({})
    self.assertIsNone(timestep.reward)

  def test_reward_piped_correctly(self):
    self._env.step({})  # Reward is None for first step.
    timestep = self._env.step({})
    self.assertEqual(5, timestep.reward)

  def test_discount_piped_correctly(self):
    timestep = self._env.step({})
    self.assertEqual('goodbye', timestep.discount)


if __name__ == '__main__':
  absltest.main()
