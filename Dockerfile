FROM floydhub/pytorch:0.3.0-gpu.cuda8cudnn6-py3.17
MAINTAINER Seungil Park <psi9730@gmail.com>

RUN apt-get update
ENV APP_PATH /naver-ai
RUN mkdir -p $APP_PATH
WORKDIR $APP_PATH

RUN pip install git+https://github.com/elzino/naver_ai_hackathon_speech.git
RUN pip install -r requirements.txt

ADD . $APP_PATH