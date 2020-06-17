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
"""Tests for dm_env_flatten_utils."""

import collections

from absl.testing import absltest
from dm_env_rpc.v1 import dm_env_flatten_utils


class FlattenUtilsTest(absltest.TestCase):

  def test_flatten(self):
    input_dict = {
        'foo': {
            'bar': 1,
            'baz': False
        },
        'fiz': object(),
    }
    expected = {
        'foo.bar': 1,
        'foo.baz': False,
        'fiz': object(),
    }
    self.assertSameElements(expected,
                            dm_env_flatten_utils.flatten_dict(input_dict, '.'))

  def test_unflatten(self):
    input_dict = {
        'foo.bar.baz': True,
        'fiz.buz': 1,
        'foo.baz': 'val',
        'buz': {}
    }
    expected = {
        'foo': {
            'bar': {
                'baz': True
            },
            'baz': 'val'
        },
        'fiz': {
            'buz': 1
        },
        'buz': {},
    }
    self.assertSameElements(
        expected, dm_env_flatten_utils.unflatten_dict(input_dict, '.'))

  def test_unflatten_different_separator(self):
    input_dict = {'foo::bar.baz': True, 'foo.bar::baz': 1}
    expected = {'foo': {'bar.baz': True}, 'foo.bar': {'baz': 1}}
    self.assertSameElements(
        expected, dm_env_flatten_utils.unflatten_dict(input_dict, '::'))

  def test_flatten_unflatten(self):
    input_output = {
        'foo': {
            'bar': 1,
            'baz': False
        },
        'fiz': object(),
    }
    self.assertSameElements(
        input_output,
        dm_env_flatten_utils.unflatten_dict(
            dm_env_flatten_utils.flatten_dict(input_output, '.'), '.'))

  def test_flatten_with_key_containing_separator_raises_error(self):
    with self.assertRaisesRegex(ValueError, 'foo.bar'):
      dm_env_flatten_utils.flatten_dict({'foo.bar': True}, '.')

  def test_invalid_flattened_dict_raises_error(self):
    input_dict = collections.OrderedDict((
        ('foo.bar', True),
        ('foo', 'invalid_value_for_sub_key'),
    ))
    with self.assertRaisesRegex(ValueError, 'Duplicate key'):
      dm_env_flatten_utils.unflatten_dict(input_dict, '.')

  def test_sub_tree_has_value_raises_error(self):
    input_dict = collections.OrderedDict((
        ('branch', 'should_not_have_value'),
        ('branch.leaf', True),
        ))
    with self.assertRaisesRegex(ValueError,
                                "Sub-tree 'branch' has already been assigned"):
      dm_env_flatten_utils.unflatten_dict(input_dict, '.')

  def test_empty_dict_values_flatten(self):
    input_dict = {
        'foo': {},
        'bar': {
            'baz': {}
        },
    }
    expected = {
        'foo': {},
        'bar.baz': {},
    }
    self.assertSameElements(expected,
                            dm_env_flatten_utils.flatten_dict(input_dict, '.'))


if __name__ == '__main__':
  absltest.main()
