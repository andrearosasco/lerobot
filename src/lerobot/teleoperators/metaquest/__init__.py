#!/usr/bin/env python

# Copyright 2024 Istituto Italiano di Tecnologia. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .metaquest_yarp.configuration_metaquest import MetaQuestConfig
from .metaquest_keyboard.configuration_metaquest_keyboard import BimanualKeyboardConfig
from .metaquest_rail.metaquest import MetaQuestRail, MetaQuestRailConfig
try:
    from .metaquest_yarp.metaquest import MetaQuest
except ImportError as e:
    _yarp_import_error = e
    class MetaQuest:
        def __init__(self, *args, **kwargs):
            raise _yarp_import_error
from .metaquest_keyboard.metaquest_keyboard import BimanualKeyboard
