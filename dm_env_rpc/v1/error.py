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

"""Provides custom Pythonic errors for dm_env_rpc error messages."""


class DmEnvRpcError(Exception):
  """A dm_env_rpc custom exception.

  Wraps a google.rpc.Status message as a Python Exception class.
  """

  def __init__(self, status_proto):
    super().__init__()
    self._status_proto = status_proto

  @property
  def code(self):
    return self._status_proto.code

  @property
  def message(self):
    return self._status_proto.message

  def __str__(self):
    return str(self._status_proto)

  def __reduce__(self):
    return (DmEnvRpcError, (self._status_proto,))
