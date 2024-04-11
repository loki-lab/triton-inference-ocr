FROM nvcr.io/nvidia/tritonserver:22.07-py3


# Prepare environment
WORKDIR /srv
ADD ./requirements.txt ./requirements.txt
ADD model_repository ./model_repository

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Install requirements
RUN pip install -r requirements.txt
