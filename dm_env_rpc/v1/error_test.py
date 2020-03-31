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
"""Tests for dm_env_rpc error module."""

import pickle

from absl.testing import absltest
from google.rpc import code_pb2
from google.rpc import status_pb2
from dm_env_rpc.v1 import error


class ErrorTest(absltest.TestCase):

  def testSimpleError(self):
    message = status_pb2.Status(
        code=code_pb2.INVALID_ARGUMENT, message='A test error.')
    exception = error.DmEnvRpcError(message)

    self.assertEqual(code_pb2.INVALID_ARGUMENT, exception.code)
    self.assertEqual('A test error.', exception.message)
    self.assertEqual(str(message), str(exception))

  def testPickleUnpickle(self):
    exception = error.DmEnvRpcError(status_pb2.Status(
        code=code_pb2.INVALID_ARGUMENT, message='foo.'))
    pickled = pickle.dumps(exception)
    unpickled = pickle.loads(pickled)

    self.assertEqual(code_pb2.INVALID_ARGUMENT, unpickled.code)
    self.assertEqual('foo.', unpickled.message)


if __name__ == '__main__':
  absltest.main()
