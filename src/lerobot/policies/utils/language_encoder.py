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
"""Shared language encoding utilities for language-conditioned policies."""

import torch
from torch import Tensor, nn


class LanguageEncoder(nn.Module):
    """
    Shared language encoder supporting CLIP, BERT, and T5 models.
    
    Handles loading pretrained models, encoding tokens, and applying
    different pooling strategies (cls, mean, max).
    
    Args:
        encoder_type: Type of encoder to use ("clip", "bert", or "t5")
        model_name: HuggingFace model name (e.g., "openai/clip-vit-base-patch32")
        pooling_strategy: How to pool token embeddings ("cls", "mean", or "max")
        freeze: Whether to freeze the encoder weights
    """
    
    def __init__(
        self,
        encoder_type: str,
        model_name: str,
        pooling_strategy: str = "cls",
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.pooling_strategy = pooling_strategy
        self.freeze = freeze
        
        # Load the appropriate encoder
        if encoder_type == "clip":
            from transformers import CLIPTextModel
            self.encoder = CLIPTextModel.from_pretrained(model_name)
        elif encoder_type == "bert":
            from transformers import BertModel
            self.encoder = BertModel.from_pretrained(model_name)
        elif encoder_type == "t5":
            from transformers import T5EncoderModel
            self.encoder = T5EncoderModel.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        
        # Freeze if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get embedding dimension from encoder config
        self.embedding_dim = self.encoder.config.hidden_size
    
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Encode language tokens and pool to get a single embedding vector.
        
        Args:
            input_ids: (B, max_length) tensor of token IDs
            attention_mask: (B, max_length) tensor of attention mask
            
        Returns:
            (B, embedding_dim) tensor of pooled language embeddings
        """
        # Convert to appropriate dtypes
        input_ids_long = input_ids.long()
        attention_mask_long = attention_mask.long()
        attention_mask_bool = attention_mask.bool()
        
        # Encode
        if self.freeze:
            self.encoder.eval()
            with torch.no_grad():
                encoder_outputs = self.encoder(
                    input_ids=input_ids_long,
                    attention_mask=attention_mask_long,
                )
        else:
            encoder_outputs = self.encoder(
                input_ids=input_ids_long,
                attention_mask=attention_mask_long,
            )
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            # Use pooled output if available (CLIP, BERT), else first token
            if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
                lang_emb = encoder_outputs.pooler_output
            else:
                lang_emb = encoder_outputs.last_hidden_state[:, 0]
        elif self.pooling_strategy == "mean":
            # Mean pooling over sequence (considering attention mask)
            token_embeddings = encoder_outputs.last_hidden_state
            input_mask_expanded = attention_mask_bool.unsqueeze(-1).expand(token_embeddings.size()).float()
            lang_emb = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        elif self.pooling_strategy == "max":
            # Max pooling over sequence
            token_embeddings = encoder_outputs.last_hidden_state
            lang_emb = token_embeddings.masked_fill(
                ~attention_mask_bool.unsqueeze(-1), 
                float('-inf')
            ).max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return lang_emb


class LanguageProjection(nn.Module):
    """
    Projects and normalizes language embeddings for integration into policy models.
    
    Args:
        encoder_dim: Dimension of the language encoder output
        projection_dim: Intermediate projection dimension (if None, uses model_dim)
        model_dim: Target dimension to match the policy's internal representation
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        encoder_dim: int,
        projection_dim: int | None,
        model_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Use model_dim if projection_dim not specified
        proj_dim = projection_dim if projection_dim is not None else model_dim
        
        # First projection from encoder to intermediate dimension
        self.projection = nn.Linear(encoder_dim, proj_dim)
        
        # Second projection from intermediate to model dimension (if needed)
        if proj_dim != model_dim:
            self.to_model = nn.Linear(proj_dim, model_dim)
        else:
            self.to_model = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)
    
    def forward(self, lang_emb: Tensor) -> Tensor:
        """
        Project and normalize language embeddings.
        
        Args:
            lang_emb: (B, encoder_dim) tensor of language embeddings
            
        Returns:
            (B, model_dim) tensor of projected and normalized embeddings
        """
        x = self.projection(lang_emb)
        x = self.to_model(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x


def filter_language_encoder_from_state_dict(state_dict: dict[str, Tensor], prefix: str = "language_encoder") -> dict[str, Tensor]:
    """
    Filter out language encoder weights from a state dict.
    
    Useful when saving/loading checkpoints with frozen language encoders
    to avoid safetensors aliasing issues.
    
    Args:
        state_dict: The state dict to filter
        prefix: The prefix to look for (default: "language_encoder")
        
    Returns:
        Filtered state dict without language encoder keys
    """
    return {
        key: value 
        for key, value in state_dict.items() 
        if not key.startswith(prefix) and not key.startswith(f"model.{prefix}")
    }
