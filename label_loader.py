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
    
    20 이하 (132개) 삭제
    쓴 옷 밑 > 꽝 삭 줍 윙 앗 헛 째 쑥 캡 클 쭈 e 멜 콕 ; 몽 j \ 댈 옥 흘 뽑 랴 끈 엊 겅 압 떠 띄 즉 튜 푼
    뜰 핀 잖 팁 흡 곤 쨈 렷 떄 탐 눕 농 칼 쇠 잊 칭 섬 쉐 눌 흰 츠 홓 례 늄 횟 탔 뭣 겸 팥 듣 둔 f 헤 벼 령 퀘 뤄 푸
    칵 짧 및 쥬 징 ㄴ 썬 짠 앤 롬 렁 팡 r 흠 뗗 뷰 섞 빕 힘 짝 렵 꺾 값 쁘 렬 셧 냅 챙 털 랫 횡 쭉 첫 덕 딜 묶 묘 풀
    듬 탈 눠 걱 몰 옮 d 딸 퀴 씹 쌍 묻 놨 둬 컴 빅 였 얄 큼 란

    특수 케이스 삭제
    65: ), 132: (, 200: ^, 306: ', 488: /, 662: blank
    
    특수문자 물음표 -> 일반 물음표
    ？(712) -> ?(123)
    '''
    ban_list = [65, 132, 200, 306, 488, 662]
    SPECIAL_QUESTION_MARK_INDEX = 712
    QUESTION_MARK_INDEX = 123
    index = 0

    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#': 
                continue

            org_index, char, freq = line.strip().split('\t')
            char = char.strip()

            original_index = int(org_index)

            if int(freq) <= 20 and original_index not in [0, 818, 819]:  # 0 : PAD_TOKEN, 818 : SOS_TOKEN, 819: EOS_TOKEN
                original2index[original_index] = -1
                continue

            if original_index in ban_list:
                original2index[original_index] = -1
                continue

            if original_index is SPECIAL_QUESTION_MARK_INDEX:
                original2index[SPECIAL_QUESTION_MARK_INDEX] = QUESTION_MARK_INDEX
                continue

            original2index[original_index] = index
            char2index[char] = index
            index2char[index] = char

            index += 1

    return char2index, index2char
