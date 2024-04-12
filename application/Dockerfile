FROM python:3.9
LABEL authors="loke"

WORKDIR /srv
ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


CMD ["python","app.py"]