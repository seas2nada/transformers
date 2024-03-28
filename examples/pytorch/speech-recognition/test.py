#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

test_text = "<|0.00|> 여기에서 13이 빠졌어요. 누구 남아요? MINUS9.72 남죠. <|5.88|>"
test_text_ = "<|0.00|>여기에서 13이 빠졌어요. 누구 남아요? MINUS9.72 남죠.<|5.88|>"
# test_text2 = "<|0.00|> 이쪽 동네에서 13 다시 뺏어 왔어요. <|2.90|>"
tokenizer = AutoTokenizer.from_pretrained('openai/whisper-large-v3')
tokenizer.set_prefix_tokens("korean", "transcribe", True)

encoded = tokenizer.encode(test_text)
encoded_ = tokenizer.encode(test_text_)
print(encoded)
print(encoded_)

decoded = tokenizer.decode(encoded, skip_special_tokens=True)
decoded_ = tokenizer.decode(encoded_, skip_special_tokens=True)
print(decoded)
print(decoded_)