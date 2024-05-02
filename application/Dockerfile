FROM python:3.9
LABEL authors="loke"

WORKDIR .

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install -r requirements.txt


CMD ["python","app.py"]