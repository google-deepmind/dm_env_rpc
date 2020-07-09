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
"""Compliance test base classes for dm_env_rpc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_env_rpc.v1.compliance import create_destroy_world
from dm_env_rpc.v1.compliance import join_leave_world
from dm_env_rpc.v1.compliance import reset
from dm_env_rpc.v1.compliance import reset_world
from dm_env_rpc.v1.compliance import step

CreateDestroyWorld = create_destroy_world.CreateDestroyWorld
JoinLeaveWorld = join_leave_world.JoinLeaveWorld
Reset = reset.Reset
ResetWorld = reset_world.ResetWorld
Step = step.Step
