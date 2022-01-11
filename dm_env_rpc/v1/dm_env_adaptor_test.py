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

from google.rpc import status_pb2
from google.protobuf import text_format
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
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
_SAMPLE_NESTED_SPECS = dm_env_rpc_pb2.ActionObservationSpecs(
    actions={
        1:
            dm_env_rpc_pb2.TensorSpec(
                dtype=dm_env_rpc_pb2.INT32, name='foo.bar'),
        2:
            dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.STRING, name='baz')
    },
    observations={
        1:
            dm_env_rpc_pb2.TensorSpec(
                dtype=dm_env_rpc_pb2.INT32, name='foo.bar'),
        2:
            dm_env_rpc_pb2.TensorSpec(dtype=dm_env_rpc_pb2.STRING, name='baz')
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
    self.assertIsNone(timestep.reward)
    self.assertIsNone(timestep.discount)
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
    self.assertIsNone(timestep.reward)
    self.assertIsNone(timestep.discount)
    self.assertEqual({'foo': 5, 'bar': 'goodbye'}, timestep.observation)

  def test_spec_generate_value_step(self):
    self._connection.send = mock.MagicMock(return_value=_SAMPLE_STEP_RESPONSE)
    action_spec = self._env.action_spec()
    actions = {
        name: spec.generate_value() for name, spec in action_spec.items()
    }
    self._env.step(actions)
    self._connection.send.assert_called_once_with(
        dm_env_rpc_pb2.StepRequest(
            requested_observations=[1, 2],
            actions={
                1: tensor_utils.pack_tensor(actions['foo']),
                2: tensor_utils.pack_tensor(actions['bar'], dtype=np.str_)
            }))

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
        'bar': specs.StringArray(shape=(), name='bar')
    }
    self.assertEqual(expected_spec, self._env.observation_spec())

  def test_action_spec(self):
    expected_spec = {
        'foo': specs.Array(shape=(), dtype=np.uint8, name='foo'),
        'bar': specs.StringArray(shape=(), name='bar')
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
    self.assertEqual(specs.StringArray(shape=()), self._env.discount_spec())

  def test_reward_from_reserved_keyword(self):
    self._connection.send = mock.MagicMock(return_value=_RESERVED_STEP_RESPONSE)
    self._env.step({})  # Reward is None for first step.
    timestep = self._env.step({})

    self.assertEqual(5, timestep.reward)

  def test_discount(self):
    self._connection.send = mock.MagicMock(return_value=_RESERVED_STEP_RESPONSE)
    timestep = self._env.step({})

    self.assertEqual('goodbye', timestep.discount)

  def test_observations_empty(self):
    self.assertEmpty(self._env.observation_spec())

  def test_explicitly_requesting_reward_and_discount(self):
    env = dm_env_adaptor.DmEnvAdaptor(
        self._connection,
        _RESERVED_SPEC,
        requested_observations=[
            dm_env_adaptor.DEFAULT_REWARD_KEY,
            dm_env_adaptor.DEFAULT_DISCOUNT_KEY
        ])
    expected_observation_spec = {
        dm_env_adaptor.DEFAULT_REWARD_KEY: env.reward_spec(),
        dm_env_adaptor.DEFAULT_DISCOUNT_KEY: env.discount_spec(),
    }
    self.assertEqual(env.observation_spec(), expected_observation_spec)


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
    self.assertEqual(specs.StringArray(shape=()), self._env.discount_spec())

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


class EnvironmentNestedActionsObservations(absltest.TestCase):

  def test_nested_specs(self):
    env = dm_env_adaptor.DmEnvAdaptor(
        connection=mock.MagicMock(), specs=_SAMPLE_NESTED_SPECS)
    expected_actions = {
        'foo': {
            'bar': specs.Array(shape=(), dtype=np.int32, name='foo.bar'),
        },
        'baz': specs.Array(shape=(), dtype=np.str_, name='baz'),
    }
    expected_observations = {
        'foo': {
            'bar': specs.Array(shape=(), dtype=np.int32, name='foo.bar'),
        },
        'baz': specs.Array(shape=(), dtype=np.str_, name='baz'),
    }

    self.assertSameElements(expected_actions, env.action_spec())
    self.assertSameElements(expected_observations, env.observation_spec())

  def test_no_nested_specs(self):
    env = dm_env_adaptor.DmEnvAdaptor(
        connection=mock.MagicMock(),
        specs=_SAMPLE_NESTED_SPECS,
        nested_tensors=False)
    expected_actions = {
        'foo.bar': specs.Array(shape=(), dtype=np.int32, name='foo.bar'),
        'baz': specs.Array(shape=(), dtype=np.str_, name='baz'),
    }
    expected_observations = {
        'foo.bar': specs.Array(shape=(), dtype=np.int32, name='foo.bar'),
        'baz': specs.Array(shape=(), dtype=np.str_, name='baz'),
    }

    self.assertSameElements(expected_actions, env.action_spec())
    self.assertSameElements(expected_observations, env.observation_spec())

  def test_nested_actions_step(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(
        return_value=text_format.Parse("""state: RUNNING""",
                                       dm_env_rpc_pb2.StepResponse()))
    env = dm_env_adaptor.DmEnvAdaptor(
        connection, specs=_SAMPLE_NESTED_SPECS, requested_observations=[])

    timestep = env.step({'foo': {'bar': 123}})

    self.assertEqual(dm_env.StepType.FIRST, timestep.step_type)

    connection.send.assert_called_once_with(
        text_format.Parse(
            """actions: { key: 1, value: { int32s: { array: 123 } } }""",
            dm_env_rpc_pb2.StepRequest()))

  def test_no_nested_actions_step(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(
        return_value=text_format.Parse("""state: RUNNING""",
                                       dm_env_rpc_pb2.StepResponse()))
    env = dm_env_adaptor.DmEnvAdaptor(
        connection,
        specs=_SAMPLE_NESTED_SPECS,
        requested_observations=[],
        nested_tensors=False)
    timestep = env.step({'foo.bar': 123})

    self.assertEqual(dm_env.StepType.FIRST, timestep.step_type)

    connection.send.assert_called_once_with(
        text_format.Parse(
            """actions: { key: 1, value: { int32s: { array: 123 } } }""",
            dm_env_rpc_pb2.StepRequest()))

  def test_nested_observations_step(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(
        return_value=text_format.Parse(
            """state: RUNNING
        observations: { key: 1, value: { int32s: { array: 42 } } }""",
            dm_env_rpc_pb2.StepResponse()))

    expected = {'foo': {'bar': 42}}

    env = dm_env_adaptor.DmEnvAdaptor(
        connection,
        specs=_SAMPLE_NESTED_SPECS,
        requested_observations=['foo.bar'])
    timestep = env.step({})
    self.assertEqual(dm_env.StepType.FIRST, timestep.step_type)
    self.assertSameElements(expected, timestep.observation)

    connection.send.assert_called_once_with(
        dm_env_rpc_pb2.StepRequest(requested_observations=[1]))

  def test_extensions(self):
    class _ExampleExtension:

      def foo(self):
        return 'bar'

    env = dm_env_adaptor.DmEnvAdaptor(
        connection=mock.MagicMock(),
        specs=_SAMPLE_SPEC,
        extensions={'extension': _ExampleExtension()})

    self.assertEqual('bar', env.extension.foo())

  def test_invalid_extension_attr(self):
    with self.assertRaisesRegex(ValueError,
                                'DmEnvAdaptor already has attribute'):
      dm_env_adaptor.DmEnvAdaptor(
          connection=mock.MagicMock(),
          specs=_SAMPLE_SPEC,
          extensions={'_connection': object()})


class CreateJoinHelpers(absltest.TestCase):

  def test_create_world(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(
        return_value=dm_env_rpc_pb2.CreateWorldResponse(
            world_name='Damogran_01'))

    world_name = dm_env_adaptor.create_world(connection, {'planet': 'Damogran'})
    self.assertEqual('Damogran_01', world_name)

    connection.send.assert_called_once_with(
        text_format.Parse(
            """settings: {
                key: 'planet', value: { strings: { array: 'Damogran' } }
            }""", dm_env_rpc_pb2.CreateWorldRequest()))

  def test_join_world(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(
        return_value=dm_env_rpc_pb2.JoinWorldResponse(specs=_SAMPLE_SPEC))

    env = dm_env_adaptor.join_world(connection, 'Damogran_01',
                                    {'player': 'zaphod'})
    self.assertIsNotNone(env)

    connection.send.assert_called_once_with(
        text_format.Parse(
            """world_name: 'Damogran_01'
                settings: {
                    key: 'player', value: { strings: { array: 'zaphod' } }
                }""", dm_env_rpc_pb2.JoinWorldRequest()))

  def test_create_join_world(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(side_effect=[
        dm_env_rpc_pb2.CreateWorldResponse(world_name='Damogran_01'),
        dm_env_rpc_pb2.JoinWorldResponse(specs=_SAMPLE_SPEC)
    ])
    env, world_name = dm_env_adaptor.create_and_join_world(
        connection,
        create_world_settings={'planet': 'Damogran'},
        join_world_settings={
            'ship_type': 1,
            'player': 'zaphod',
        })
    self.assertIsNotNone(env)
    self.assertEqual('Damogran_01', world_name)

    connection.send.assert_has_calls([
        mock.call(
            text_format.Parse(
                """settings: {
                key: 'planet', value: { strings: { array: 'Damogran' } }
            }""", dm_env_rpc_pb2.CreateWorldRequest())),
        mock.call(
            text_format.Parse(
                """world_name: 'Damogran_01'
                settings: { key: 'ship_type', value: { int64s: { array: 1 } } }
                settings: {
                    key: 'player', value: { strings: { array: 'zaphod' } }
                }""", dm_env_rpc_pb2.JoinWorldRequest())),
    ])

  def test_create_join_world_with_packed_settings(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(side_effect=[
        dm_env_rpc_pb2.CreateWorldResponse(world_name='Magrathea_02'),
        dm_env_rpc_pb2.JoinWorldResponse(specs=_SAMPLE_SPEC)
    ])
    env_and_world_name = dm_env_adaptor.create_and_join_world(
        connection,
        create_world_settings={'planet': tensor_utils.pack_tensor('Magrathea')},
        join_world_settings={
            'ship_type': tensor_utils.pack_tensor(2),
            'player': tensor_utils.pack_tensor('arthur'),
            'unpacked_setting': [1, 2, 3],
        })
    self.assertIsNotNone(env_and_world_name.env)
    self.assertEqual('Magrathea_02', env_and_world_name.world_name)

    connection.send.assert_has_calls([
        mock.call(
            text_format.Parse(
                """settings: {
                key: 'planet', value: { strings: { array: 'Magrathea' } }
            }""", dm_env_rpc_pb2.CreateWorldRequest())),
        mock.call(
            text_format.Parse(
                """world_name: 'Magrathea_02'
                settings: { key: 'ship_type', value: { int64s: { array: 2 } } }
                settings: {
                    key: 'player', value: { strings: { array: 'arthur' } }
                }
                settings: {
                    key: 'unpacked_setting', value: {
                      int64s: { array: 1 array: 2 array: 3 }
                      shape: 3
                    }
                }""", dm_env_rpc_pb2.JoinWorldRequest())),
    ])

  def test_create_join_world_with_extension(self):

    class _ExampleExtension:

      def foo(self):
        return 'bar'

    connection = mock.MagicMock()
    connection.send = mock.MagicMock(side_effect=[
        dm_env_rpc_pb2.CreateWorldResponse(world_name='foo'),
        dm_env_rpc_pb2.JoinWorldResponse(specs=_SAMPLE_SPEC)
    ])

    env, _ = dm_env_adaptor.create_and_join_world(
        connection,
        create_world_settings={},
        join_world_settings={},
        extensions={'extension': _ExampleExtension()})
    self.assertEqual('bar', env.extension.foo())

  def test_create_join_world_with_unnested_tensors(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(side_effect=[
        dm_env_rpc_pb2.CreateWorldResponse(world_name='Damogran_01'),
        dm_env_rpc_pb2.JoinWorldResponse(specs=_SAMPLE_NESTED_SPECS)
    ])
    env, _ = dm_env_adaptor.create_and_join_world(
        connection,
        create_world_settings={},
        join_world_settings={},
        nested_tensors=False)
    expected_actions = {
        'foo.bar': specs.Array(shape=(), dtype=np.int32, name='foo.bar'),
        'baz': specs.Array(shape=(), dtype=np.str_, name='baz'),
    }
    expected_observations = {
        'foo.bar': specs.Array(shape=(), dtype=np.int32, name='foo.bar'),
        'baz': specs.Array(shape=(), dtype=np.str_, name='baz'),
    }

    self.assertSameElements(expected_actions, env.action_spec())
    self.assertSameElements(expected_observations, env.observation_spec())

  def test_create_join_world_with_invalid_extension(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(side_effect=[
        dm_env_rpc_pb2.CreateWorldResponse(world_name='foo'),
        dm_env_rpc_pb2.JoinWorldResponse(specs=_SAMPLE_SPEC),
        dm_env_rpc_pb2.LeaveWorldResponse(),
        dm_env_rpc_pb2.DestroyWorldRequest()
    ])

    with self.assertRaisesRegex(ValueError,
                                'DmEnvAdaptor already has attribute'):
      _ = dm_env_adaptor.create_and_join_world(
          connection,
          create_world_settings={},
          join_world_settings={},
          extensions={'step': object()})

    connection.send.assert_has_calls([
        mock.call(dm_env_rpc_pb2.CreateWorldRequest()),
        mock.call(dm_env_rpc_pb2.JoinWorldRequest(world_name='foo')),
        mock.call(dm_env_rpc_pb2.LeaveWorldRequest()),
        mock.call(dm_env_rpc_pb2.DestroyWorldRequest(world_name='foo'))
    ])

  def test_created_but_failed_to_join_world(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(
        side_effect=(
            dm_env_rpc_pb2.CreateWorldResponse(world_name='Damogran_01'),
            error.DmEnvRpcError(status_pb2.Status(message='Failed to Join.')),
            dm_env_rpc_pb2.DestroyWorldResponse()))

    with self.assertRaisesRegex(error.DmEnvRpcError, 'Failed to Join'):
      _ = dm_env_adaptor.create_and_join_world(
          connection, create_world_settings={}, join_world_settings={})

    connection.send.assert_has_calls([
        mock.call(dm_env_rpc_pb2.CreateWorldRequest()),
        mock.call(dm_env_rpc_pb2.JoinWorldRequest(world_name='Damogran_01')),
        mock.call(dm_env_rpc_pb2.DestroyWorldRequest(world_name='Damogran_01'))
    ])

  def test_created_and_joined_but_adaptor_failed(self):
    connection = mock.MagicMock()
    connection.send = mock.MagicMock(
        side_effect=(
            dm_env_rpc_pb2.CreateWorldResponse(world_name='Damogran_01'),
            dm_env_rpc_pb2.JoinWorldResponse(specs=_SAMPLE_SPEC),
            dm_env_rpc_pb2.LeaveWorldResponse(),
            dm_env_rpc_pb2.DestroyWorldResponse()))

    with self.assertRaisesRegex(ValueError, 'Unsupported observations'):
      _ = dm_env_adaptor.create_and_join_world(
          connection,
          create_world_settings={},
          join_world_settings={},
          requested_observations=['invalid_observation'])

    connection.send.assert_has_calls([
        mock.call(dm_env_rpc_pb2.CreateWorldRequest()),
        mock.call(dm_env_rpc_pb2.JoinWorldRequest(world_name='Damogran_01')),
        mock.call(dm_env_rpc_pb2.LeaveWorldRequest()),
        mock.call(dm_env_rpc_pb2.DestroyWorldRequest(world_name='Damogran_01'))
    ])


if __name__ == '__main__':
  absltest.main()
