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

original2index = dict() # [original] = id


def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    '''
    * 데이터 전처리 *
    특수 케이스 삭제
    65: ), 132: (, 200: ^, 306: ', 488: /, 662: blank, 722: ㄴ, 745: \ 
    
    특수문자 물음표 -> 일반 물음표
    >(435) -> ?(123)
    ;(646) -> ?(123)
    ？(712) -> ?(123)
    '''

    ban_list = [65, 132, 200, 306, 488, 662, 722, 745]
    SPECIAL_QUESTION_MARK_LIST = [435, 646, 712]
    QUESTION_MARK_INDEX = 123
    index = 0

    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#': 
                continue

            org_index, char, freq = line.strip().split('\t')
            char = char.strip()

            original_index = int(org_index)

            if original_index in ban_list:
                original2index[original_index] = -1
                continue

            if original_index in SPECIAL_QUESTION_MARK_LIST:
                original2index[original_index] = QUESTION_MARK_INDEX
                continue

            original2index[original_index] = index
            char2index[char] = index
            index2char[index] = char

            index += 1

    return char2index, index2char
