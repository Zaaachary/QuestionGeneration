# -*- encoding: utf-8 -*-
'''
@File    :   data_io_util.py
@Time    :   2022/03/29 12:14:44
@Author  :   Zhifeng Li
@Contact :   li_zaaachary@163.com
@Desc    :   
'''
import json
from collections import OrderedDict

def load_data(file_path, mode='plain'):
    '''
    load data from jsonl/tsf or plain
    mode: jsonl / tsf / plain
    '''
    result = []
    f = open(file_path, 'r', encoding='utf-8')

    if mode == 'json':
        result =  json.load(f, object_pairs_hook=OrderedDict)
    else:
        while True:
            line = f.readline()
            if not line:
                break
            if mode == 'jsonl':
                # print(line)
                line = json.loads(line, object_pairs_hook=OrderedDict)
                result.append(line)
            elif mode == 'tsf':
                line = line.strip('\n').split('\t')
                result.append(line)
            elif mode == 'plain':
                line = line.strip()
                result.append(line)
    f.close()
    return result


def dump_data(target, file_path, mode='json'):
    f = open(file_path, 'w', encoding='utf-8')
    if mode == 'json':
        json.dump(target, f, ensure_ascii=False, indent=2)
    elif mode == 'tsf':
        for line in target:
            line = list(map(str, line))
            f.write('\t'.join(line) + '\n')
    elif mode == 'jsonl':
        for line in target:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line+'\n')
    elif mode == 'csv':
        for line in target:
            line = list(map(str, line))
            line = [item.replace(',',' ') for item in line]
            f.write(','.join(line) + '\n')
            
    f.close()