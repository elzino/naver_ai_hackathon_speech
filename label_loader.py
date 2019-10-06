"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-

def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    ban_list = [65, 132, 200, 306, 435, 488, 646, 662, 722, 745]
    with open(label_path, 'r') as f:
        for _, line in enumerate(f):
            if line[0] == '#': 
                continue

            index, char, _ = line.strip().split('\t')
            char = char.strip()

            index_num = int(index)
            count = 0
            if index_num in ban_list:       # data 전처리 65: ), 132: (, 200: ^, 306: ', 435: >,  488: /, 646: ;, 662: blank, 722: ㄴ,  745: \
                continue            
            for i in range(0, len(ban_list)) : 
                if ban_list[i] > index_num : 
                    break
                count += 1
            index_num -= count
            char2index[char] = index_num
            index2char[index_num] = char

    return char2index, index2char
