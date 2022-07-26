import os
import time
import json
import logging
import random
from multiprocessing import Pool, cpu_count     # https://docs.python.org/3/library/multiprocessing.html
from collections import OrderedDict
from itertools import chain
import pickle

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader, sampler
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import sys
sys.path.append('../')
from data_io_util import load_data, dump_data


class DuReaderQG_Dataset(Dataset):

    def __init__(self, 
        dataset_path, max_src_len=None, max_tgt_len=None, 
        tokenizer=None, dataset_type='train', evaluate=False,
        ):

        super().__init__()
        self.evaluate = evaluate
        self.raw_examples = []
        self.labels = []
        self.examples = []
        self.scores = []
        self.example_ids = []

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.load_data()
        if not evaluate:
            self.convert_tokens_to_ids()

    def load_data(self):
        if os.path.isfile(self.dataset_path):
            target = self.dataset_path
        else:
            target = os.path.join(self.dataset_path, f"{self.dataset_type}.jsonl")
        data_list = load_data(target, mode='jsonl')
        if not self.evaluate:
            for data in data_list:
                context = data['context']
                question = data['question']
                answer = data['answer']
                # example_id = data['id']
                question = self.transform_question(question)
                self.raw_examples.append([[answer, context], question])
        else:
            for data in data_list:
                context = data['context']
                answer = data['answer']
                example_id = data['id']        
                self.raw_examples.append([example_id, [answer, context]])
        logging.info(f"{self.dataset_type} dataset loaded")

    class Convert:
        # tokenize and get the max len   for multiprocessing
        def __init__(self, 
            tokenizer, 
            evaluate,
            max_src_len=50, max_tgt_len=50
            ):
            self.tokenizer = tokenizer
            self.evaluate = evaluate
            self.max_src_len = max_src_len   # for bart encoder
            self.max_tgt_len = max_tgt_len   # for gpt2 & bart decoder

        def __call__(self, raw_example):
            try:
                return self._bart(raw_example)
            except:
                logging.warning(raw_example)
                exit()

        def _bart(self, raw_example):
            source, target = raw_example
            answer, passage = source
            source_inputs = self.tokenizer(answer, passage, 
                max_length=self.max_src_len, padding='max_length', truncation='longest_first')
            target_inputs = self.tokenizer.encode(target,
                max_length=self.max_tgt_len, padding='max_length', truncation='longest_first')

            target_inputs.insert(0, self.tokenizer.sep_token_id)
            feature_dict = source_inputs
            feature_dict['decoder_input_ids'] = target_inputs
            return feature_dict
    
    def convert_tokens_to_ids(self):
        '''
        make input and label
        tokenized_examples: 
        -gpt2 [question, answer] 
        -bart [
            source: <CLS> answer <SEP> passage <SEP>
            target: <SEP> <CLS> question <SEP>
            ]
        '''
        
        logging.info(f"tokenizing {self.dataset_type} examples")
        logging.info(f'data format {self.raw_examples[0]}')
        
        
        now = time.time()
        with Pool(processes=min(8, cpu_count())) as pool:
            tokenized_examples = pool.map(
                self.Convert(self.tokenizer, self.evaluate, self.max_src_len, self.max_tgt_len),
                self.raw_examples)
        logging.info(f"start {min(8, cpu_count())} processes, cost {time.time() - now}")
        
        if self.evaluate:
            # TODO
            pass
        else:
            for feature_dict in tokenized_examples:
                label = feature_dict['decoder_input_ids'][1:]
                decoder_input_ids = feature_dict['decoder_input_ids'][:-1]# no need to input the last token 
                feature_dict['decoder_input_ids'] = decoder_input_ids
                self.examples.append(feature_dict)

                # mask question part
                for index, token in enumerate(label):
                    if token == self.tokenizer.pad_token_id:
                        label[index] = -100
                self.labels.append(label)

    def make_dataloader(self, batch_size):
        if self.dataset_type == "train":
            data_sampler = RandomSampler(self)
        else:
            data_sampler = SequentialSampler(self)

        dataloader = DataLoader(self, sampler=data_sampler, batch_size=batch_size, num_workers=4, collate_fn=self.collate_fn) # TODO

        return dataloader

    @staticmethod
    def transform_question(origin:str):
        origin = origin.replace('?', "？")
        if not origin.endswith('？'):
            return origin + '？'
        else:
            return origin

    def collate_fn(self, batch):

        pad_token_id = self.tokenizer.pad_token_id
        max_inputids_len = 0
        max_dec_inputids_len = 0
        for example in batch:
            input_ids = example['input_ids']
            decoder_input_ids = example['decoder_input_ids']
            for index, token in enumerate(input_ids):
                if token == pad_token_id:
                    max_inputids_len = max(max_inputids_len, index)
                    break
            else:
                max_inputids_len = len(input_ids)
            for index, token in enumerate(decoder_input_ids):
                if token == pad_token_id:
                    max_dec_inputids_len = max(max_dec_inputids_len, index)
                    break
        input_ids = []
        masks = []
        target_ids = []
        labels = []
        for example in batch:
            input_ids.append(example['input_ids'][:max_inputids_len])
            masks.append(example['attention_mask'][:max_inputids_len])
            target_ids.append(example["decoder_input_ids"][:max_dec_inputids_len])
            labels.append(example['labels'][:max_dec_inputids_len])
        
        input_ids = torch.stack(input_ids)
        masks = torch.stack(masks)
        target_ids = torch.stack(target_ids)
        labels = torch.stack(labels)
                
        batch = {
            "input_ids": input_ids,
            "attention_mask": masks,
            "decoder_input_ids": target_ids,
            "labels": labels
        }
        return batch

    def __len__(self):
        if self.evaluate:
            return len(self.raw_examples)
        else:
            return len(self.examples)

    def __getitem__(self, idx):
        if self.evaluate:
            return self.raw_examples[idx]
            # return {'input_ids': torch.tensor(self.examples[idx])}
        else:
            feature_dict= {}
            for key, value in self.examples[idx].items():
                feature_dict[key] = torch.tensor(value)
            feature_dict['labels'] = torch.tensor(self.labels[idx])
            return feature_dict



if __name__ == "__main__":
    from transformers import BertTokenizer
    dataset_path = "/SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/QuestionGeneration/DuReaderQG/dev.json"
    tokenizer = BertTokenizer.from_pretrained("/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/cpt-base")
    
    dataset = DuReaderQG_Dataset(
        dataset_path, 
        max_src_len=40, max_tgt_len=45, 
        tokenizer=tokenizer, 
        dataset_type='train', evaluate=False,
    )
    batch = [dataset[0],dataset[33]]
    # dataset.collate_fn(batch)
    dataloader = dataset.make_dataloader(4)
    for batch in dataloader:
        print(batch)
        # import pdb; pdb.set_trace()
    