FROM floydhub/pytorch:1.1.0-gpu.cuda9cudnn7-py3.45
LABEL authors="Seungil Park <psi9730@gmail.com>, Jinho Lee <elzinomaster@gmail.com>"

ENV APP_PATH /naver-ai
RUN apt-get install libsndfile1
RUN mkdir -p $APP_PATH
WORKDIR $APP_PATH

RUN pip install git+https://github.com/elzino/naver_ai_hackathon_speech.git
RUN pip install git+https://github.com/n-CLAIR/nsml-local

ADD . $APP_PATH
