# NAVER AI HACKATHON 2019 - Speech To Text
We participated in [NAVER_AI_HACKTHON 2019](https://github.com/Naver-AI-Hackathon/AI-Speech) and ranked 9th(77.0155) as a two-man team(Hangbok Coding).
* 박승일(Seungil Park) [github repo](https://psi9730.github.io/)
* 이진호(Jinho Lee)    [github repo](https://github.com/elzino)

## Final Leaderboard
![finale-leader-board](docs/final-board.png)

## Features
* Seq2Seq(bidirectional LSTM encoder, unidiredtional LSTM encoder with Bahdanau Attention)
* spec-augmented log melspectrogram
* beam-search
* labelsmoothing
* Multi step learning rate
* ensemble (but not used for best model)
* data preprocessing (delete blank and special characters)
## How to RUN

### Docker
```bash
$ docker build -t model:0.0 .
$ docker run -i --name model model:0.0
$ docker exec -i -t model /bin/bash
$ ./run.sh
```

### NSML
Login with nsml first, and run commands as follows:
```bash
$ ./run.sh # for local training
$ ./run_nsml.sh  # for NSML training
$ nsml submit (sessionName) (checkpoint) # for submit
```

## Hyperparameters
| Help        | default           |  |
| ------------- |:--------:| ------:|
| hidden size of model | 256 | --hiden_size | 
| size of embedding dimesion | 64 | --embedding_size |
| number of layers of encoder | 4 | --encoder_layer_size |
| number of layers of decoder | 3 | --decoder_layer_size |
| batch size | 32 | --batch_size |
| learning rate | 1e-04 | --lr |
| teachr forcing | 0.5 | --teacher_forcing|
| maximum characters of sentence | 80 | --max_len |

## Reference
1. [Listen and spell](https://arxiv.org/abs/1508.01211)
2. [label smoothing](https://arxiv.org/abs/1906.02629)
3. [spec augment](https://arxiv.org/abs/1904.08779)

## License
Copyright 2019 hangbok coding.
```
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```