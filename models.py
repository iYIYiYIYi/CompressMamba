import torch
import torch.nn as nn
from torch.nn.functional import softmax
import os
import random
import numpy as np

from typing import Dict, List, Optional, Sequence, Union
import json
from pathlib import Path
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
import sentencepiece as spm

class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, padding_side: str='left', **kwargs):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=sep_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)

    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.
        """
        return self._vocab_str_to_int

# model_type in ['single', 'double', 'sentencepiece']
def get_tokenizer(seq_len, vsize, model_type='single'):

    if model_type == 'sentencepiece':
        spm_model_path = f'./tokenizers/bpe_{vsize}.model'
        sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        tokenizer = sp
    elif model_type == 'single':
        tokenizer = CharacterTokenizer(characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=seq_len,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            use_fast=True,  # use fast Rust implementation
            padding_side='left',  # since HyenaDNA is causal, we pad on the left
        )
    elif model_type == 'double':
        characters = ['A', 'C', 'G', 'T', 'N', 'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
        tokenizer = CharacterTokenizer(characters=characters,  # add DNA characters, N is uncertain
            model_max_length=seq_len,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            use_fast=True,  # use fast Rust implementation
            padding_side='left',  # since HyenaDNA is causal, we pad on the left
        )
    
    tokenizer_args = {
        'return_tensors': 'pt',
        'max_length': seq_len,
        # 'truncation': True,
        'padding': 'max_length',
        'add_special_tokens': False,
    }
    return tokenizer, tokenizer_args

from standalone_compressmamba import CompressMamba

def get_CompressMamba(seq_len, vocab_size, model_type, head_dim, mamba_blocks=8, expand=4, d_model=512, output_heads=1):
    tokenizer, tokenizer_args = get_tokenizer(seq_len, vocab_size, model_type)
    v_size = vocab_size
    if model_type == 'sentencepiece':
        v_size = tokenizer.get_piece_size()
    else:
        v_size = tokenizer.vocab_size
    model = CompressMamba(v_size, v_size, mamba_blocks=mamba_blocks,
                                   head_dim=head_dim, expand=expand, d_model=d_model, output_heads=output_heads)
    return model, tokenizer, tokenizer_args

from utils.attention_layer import SelfAttentionLayer
import torch.nn.functional as F
class DeepDNA_Attn(nn.Module):
    def __init__(self, tokenizer):
        super(DeepDNA_Attn, self).__init__()
        self.tokenizer = tokenizer
        self.conv1 = nn.Conv1d(in_channels=tokenizer.vocab_size,  
                                out_channels=1024,
                                kernel_size=24,
                                stride=1,
                                padding=24 // 2,  # 'same' padding in TensorFlow
                            )  
        # self.conv1 = nn.Conv2d(5, 1024, kernel_size=(24,24), padding=24 // 2, stride=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)
        self.lstm1 = nn.LSTM(1024, 256, batch_first=True, dropout=0.1, num_layers=2)
        self.attn = SelfAttentionLayer(256)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, tokenizer.vocab_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x, global_position=0):
        param_type = self.conv1.weight.dtype
        x = F.one_hot(x, num_classes=self.tokenizer.vocab_size).to(param_type)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        x = self.lstm1(x)[0][:, -1, :]
        x, _ = self.attn(x)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return [x]
    
def get_DeepDNA_Attn(seq_len):
    tokenizer, tokenizer_args = get_tokenizer(seq_len, 0)
    return DeepDNA_Attn(tokenizer), tokenizer, tokenizer_args

class DeepDNA(nn.Module):
    def __init__(self, tokenizer):
        super(DeepDNA, self).__init__()
        self.tokenizer = tokenizer
        self.conv1 = nn.Conv1d(in_channels=self.tokenizer.vocab_size, 
                                out_channels=1024,
                                kernel_size=24,
                                stride=1,
                                padding=24 // 2, 
                            )  
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)
        self.lstm1 = nn.LSTM(1024, 256, batch_first=True, dropout=0.1, num_layers=2)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, tokenizer.vocab_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x, global_position=0):
        param_type = self.conv1.weight.dtype
        x = F.one_hot(x, num_classes=self.tokenizer.vocab_size).to(param_type)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        x = self.lstm1(x)[0][:, -1, :]
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return [x]
    
def get_DeepDNA(seq_len):
    tokenizer, tokenizer_args = get_tokenizer(seq_len, 0)
    return DeepDNA(tokenizer), tokenizer, tokenizer_args

def get_DeepDNA_DB(seq_len):
    tokenizer, tokenizer_args = get_tokenizer(seq_len, 0, 'double')
    return DeepDNA(tokenizer), tokenizer, tokenizer_args

class CompressAttn(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, num_layers=2):
        super(CompressAttn, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.positional_encoding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, src, global_position=0):
        position_encoding = self.positional_encoding(src)
        output = self.transformer_encoder(position_encoding)
        output = self.mlp(output.sum(dim=1))
        return [output]
    
def get_CompressAttn(seq_len, vocab_size):
    tokenizer, tokenizer_args = get_tokenizer(seq_len, vocab_size)
    return CompressAttn(tokenizer.vocab_size), tokenizer, tokenizer_args