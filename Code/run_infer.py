# -*- encoding: utf-8 -*-
"""
@File    :   run_infer.py
@Time    :   2021/10/13 23:45:30
@Author  :   Zhifeng Li
@Contact :   zaaachary_li@163.com
@Desc    :   加载模型推断结果

protoqa_evaluator evaluate --similarity_function exact_match targets.jsonl predictions.jsonl

protoqa_evaluator evaluate --similarity_function wordnet targets.jsonl predictions.jsonl

/home/zhifli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/evaluate/dev.crowdsourced.jsonl
/data1/zhifli/Models/output/gpt2_baseline/gpt2-large-1_16_4/prediction_300.jsonl

"""
import os
import argparse
import pdb
import logging
import json
from collections import Counter
from re import L
import sys

from nltk.corpus import stopwords
from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
import numpy as np
from data import DuReaderQG_Dataset

from data_io_util import dump_data
from model_wrapper import Generation_Model

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  


def prepare_inputs(args, raw_text, tokenizer, device):
    answer, passage = raw_text

    source_inputs = tokenizer.encode(answer, passage, 
                max_length=args.max_src_len, truncation='longest_first')

    input_ids = torch.tensor(source_inputs, dtype=torch.long, device=device).unsqueeze(0)
        
    faeture_dict = {'input_ids':input_ids}

    return faeture_dict

def run_beam_search(args, dataset, tokenizer, beam_nums, sample_nums, model, device):
    max_length = args.eval_length
    all_info_lsit = []

    for index, example in tqdm(enumerate(dataset), total=len(dataset)):
        # prepare input
        example_ids, raw_text = example
        info = {"example_id": example_ids, 'inputs': raw_text}
        feature_dict = prepare_inputs(args, raw_text, tokenizer, device)
        # run model
        outputs = model.generate(
            num_beams=beam_nums, num_return_sequences=sample_nums, 
            max_length=max_length,
            output_scores = True,
            return_dict_in_generate= True,
            **feature_dict
        )
        outputs_token = outputs['sequences']
        question_token = outputs_token[0]
        question = tokenizer.decode(question_token, skip_special_tokens=True)
        info['question'] = question.replace(' ','')
        all_info_lsit.append(info)
    return all_info_lsit

def load_model(model_path, args):
    logging.info(f'load model from <{model_path}>')
    model = Generation_Model.load_from_checkpoint(
        PTM_name_or_path=args.PTM_name_or_path,
        checkpoint_path=model_path
        )
    tokenizer = model.get_tokenizer()
    return model, tokenizer


def main(args):
    if args.device >= 0:
        torch.cuda.set_device(args.device)
        device = 'cuda'
    else:       # -1 
        device = 'cpu'
        
    ckpt_path = args.model_path
    output_path = args.output_path
        
    model, tokenizer = load_model(ckpt_path, args)
    model.to(device)

    # dev/test Dataset
    logging.info(f'load dataset from <{args.input_path}>')
    dataset = DuReaderQG_Dataset(
        args.input_path, None, None,
        tokenizer, "predict", evaluate=True
    )

    # run generation
    logging.info(f'run generation')
    all_info_lsit = run_beam_search(args, dataset, tokenizer, args.beam_nums, args.beam_nums, model, device)

    logging.info(f'save result to < {output_path} >')
    dump_data(all_info_lsit, output_path, mode='json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_method", choices=['beam_search'], default='beam_search')
    parser.add_argument("--beam_nums", type=int, default=1)
    parser.add_argument("--eval_length", type=int, default=30)
    parser.add_argument("--max_src_len", type=int, default=128,
        help='Bart source_length, GPT2 seq_len')
    
    parser.add_argument("--experiment", type=str, default='')
    parser.add_argument("--device", type=int, required=True, help='-1 means cpu')

    parser.add_argument("--model_path", type=str, required=True, help='model ckpt or ckpt dir')
    parser.add_argument("--output_path", type=str, default=None, help='empty or a dir')
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--PTM_name_or_path", type=str, required=True)
    
    args = parser.parse_args()
    logging.info(args)
    main(args)