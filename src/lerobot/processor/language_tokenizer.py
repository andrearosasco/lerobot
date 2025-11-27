#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Shared language tokenization processor step for policies with language conditioning."""

from dataclasses import dataclass, field
from typing import Any

from transformers import AutoTokenizer

from lerobot.processor import (
    EnvTransition,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


@dataclass
@ProcessorStepRegistry.register(name="language_tokenizer")
class LanguageTokenizerStep(ProcessorStep):
    """
    Tokenizes task text for language-conditioned policies.
    
    Converts task strings to token IDs and attention masks that can be
    processed by language encoders (CLIP, BERT, T5, etc.).
    
    This is a shared processor step used by multiple policies (ACT, Diffusion, etc.)
    that support language conditioning.
    """
    
    tokenizer_name: str = "openai/clip-vit-base-patch32"
    max_length: int = 77
    task_key: str = "task"
    
    # Internal tokenizer instance
    tokenizer: Any = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        
        # Get task text from complementary data
        task = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if task is None:
            raise ValueError(f"Task not found in complementary data with key '{self.task_key}'")
        
        # Tokenize
        tokenized = self.tokenizer(
            task,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Add to observation (keep batch dim so downstream batching doesn't drop it)
        observation = dict(transition.get(TransitionKey.OBSERVATION, {}))
        observation[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"]
        observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].bool()
        
        transition[TransitionKey.OBSERVATION] = observation
        
        return transition
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Add language token features to the feature dictionary."""
        # This step adds new features but doesn't modify existing ones
        return features
