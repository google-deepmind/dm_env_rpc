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
"""Tests Properties extension."""

import contextlib

from absl.testing import absltest
from dm_env import specs
import mock
import numpy as np

from google.protobuf import any_pb2
from google.rpc import status_pb2
from google.protobuf import text_format
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import error
from dm_env_rpc.v1.extensions import properties
from dm_env_rpc.v1.extensions import properties_pb2


def _create_property_request_key(text_proto):
  extension_message = any_pb2.Any()
  extension_message.Pack(
      text_format.Parse(text_proto, properties_pb2.PropertyRequest()))
  return dm_env_rpc_pb2.EnvironmentRequest(
      extension=extension_message).SerializeToString()


def _pack_property_response(text_proto):
  extension_message = any_pb2.Any()
  extension_message.Pack(
      text_format.Parse(text_proto, properties_pb2.PropertyResponse()))
  return dm_env_rpc_pb2.EnvironmentResponse(extension=extension_message)

# Set of expected requests and associated responses for mock connection.
_EXPECTED_REQUEST_RESPONSE_PAIRS = {
    _create_property_request_key('read_property { key: "foo" }'):
        _pack_property_response(
            'read_property { value: { int32s: { array: 1 } } }'),
    _create_property_request_key("""write_property {
               key: "bar"
               value: { strings { array: "some_value" } }
             }"""):
        _pack_property_response('write_property {}'),
    _create_property_request_key('read_property { key: "bar" }'):
        _pack_property_response(
            'read_property { value: { strings: { array: "some_value" } } }'),
    _create_property_request_key('list_property { key: "baz" }'):
        _pack_property_response("""list_property {
                    values: {
                      is_readable:true
                      spec { name: "baz.fiz" dtype:UINT32 shape: 2 shape: 2 }
                    }}"""),
    _create_property_request_key('list_property {}'):
        _pack_property_response("""list_property {
                 values: { is_readable:true spec { name: "foo" dtype:INT32 }
                           description: "This is a documented integer" }
                 values: { is_readable:true
                           is_writable:true
                           spec { name: "bar" dtype:STRING } }
                 values: { is_listable:true spec { name: "baz" } }
               }"""),
    _create_property_request_key('read_property { key: "bad_property" }'):
        dm_env_rpc_pb2.EnvironmentResponse(
            error=status_pb2.Status(message='invalid property request.'))
}


@contextlib.contextmanager
def _create_mock_connection():
  """Helper to create mock dm_env_rpc connection."""
  with mock.patch.object(dm_env_rpc_connection,
                         'dm_env_rpc_pb2_grpc') as mock_grpc:

    def _process(request_iterator):
      for request in request_iterator:
        yield _EXPECTED_REQUEST_RESPONSE_PAIRS[request.SerializeToString()]

    mock_stub_class = mock.MagicMock()
    mock_stub_class.Process = _process
    mock_grpc.EnvironmentStub.return_value = mock_stub_class
    yield dm_env_rpc_connection.Connection(mock.MagicMock())


class PropertiesTest(absltest.TestCase):

  def test_read_property(self):
    with _create_mock_connection() as connection:
      extension = properties.PropertiesExtension(connection)
      self.assertEqual(1, extension['foo'])

  def test_write_property(self):
    with _create_mock_connection() as connection:
      extension = properties.PropertiesExtension(connection)
      extension['bar'] = 'some_value'
      self.assertEqual('some_value', extension['bar'])

  def test_list_property(self):
    with _create_mock_connection() as connection:
      extension = properties.PropertiesExtension(connection)
      property_specs = extension.specs('baz')
      self.assertLen(property_specs, 1)

      property_spec = property_specs['baz.fiz']
      self.assertTrue(property_spec.readable)
      self.assertFalse(property_spec.writable)
      self.assertFalse(property_spec.listable)
      self.assertEqual(
          specs.Array(shape=(2, 2), dtype=np.uint32), property_spec.spec)

  def test_root_list_property(self):
    with _create_mock_connection() as connection:
      extension = properties.PropertiesExtension(connection)
      property_specs = extension.specs()
      self.assertLen(property_specs, 3)
      self.assertTrue(property_specs['foo'].readable)
      self.assertTrue(property_specs['bar'].readable)
      self.assertTrue(property_specs['bar'].writable)
      self.assertTrue(property_specs['baz'].listable)

  def test_invalid_spec_request_on_listable_property(self):
    with _create_mock_connection() as connection:
      extension = properties.PropertiesExtension(connection)
      property_specs = extension.specs()
      self.assertTrue(property_specs['baz'].listable)
      self.assertIsNone(property_specs['baz'].spec)

  def test_invalid_request(self):
    with _create_mock_connection() as connection:
      extension = properties.PropertiesExtension(connection)
      with self.assertRaisesRegex(error.DmEnvRpcError,
                                  'invalid property request.'):
        _ = extension['bad_property']

  def test_property_description(self):
    with _create_mock_connection() as connection:
      extension = properties.PropertiesExtension(connection)
      property_specs = extension.specs()
      self.assertEqual('This is a documented integer',
                       property_specs['foo'].description)

  def test_property_print(self):
    with _create_mock_connection() as connection:
      extension = properties.PropertiesExtension(connection)
      property_specs = extension.specs()
      self.assertRegex(
          str(property_specs['foo']),
          (r'PropertySpec\(key=foo, readable=True, writable=False, '
           r'listable=False, spec=.*, '
           r'description=This is a documented integer\)'))


if __name__ == '__main__':
  absltest.main()
