# usage: python utils/average_models.py --model_dirs "/exps/whisper-large-KlecSpeech-fresh, /exps/whisper-large-KconfSpeech-max10000-batch256, /exps/whisper-large-KsponSpeech-max10000-batch256" --save_path /exps/WA_models/KlecKconfKspon/pytorch_model.bin
import os
import argparse

import torch
from arg_utils import str2bool, str2list

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
)
from transformers.models.whisper import WhisperForConditionalGeneration

def get_args():
    parser = argparse.ArgumentParser(description="ASR Evaluation Script")
    parser.add_argument("--model_dirs", type=str2list, help="/path/to/model1, /path/to/model2, ..., /path/to/modelN")
    parser.add_argument("--WA_type", type=str, default="uniform")
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="/exps/WA_models/model.bin")

    return parser.parse_args()

def main(args):
    num_models = 0
    total_params = {}
    for model_name_or_path in args.model_dirs:
        config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else model_name_or_path,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name_or_path,
            config=config,
        )

        for name, param in model.proj_out.named_parameters():
            if 'proj_out.weight' not in total_params:
                total_params['proj_out.weight'] = param.clone().detach()
            else:
                total_params['proj_out.weight'] += param

        for name, param in model.named_parameters():
            if name not in total_params:
                total_params[name] = param.clone().detach()
            else:
                total_params[name] += param
    
        num_models += 1

    for name in total_params:
        total_params[name] /= num_models
    
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

    # print(model_name_or_path)
    # exit()
    averaged_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, config=config)

    averaged_model.load_state_dict(total_params)
    torch.save(averaged_model.state_dict(), args.save_path)

if __name__ == "__main__":
    args = get_args()
    main(args)