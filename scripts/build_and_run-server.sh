#build server
docker build ./ -f Dockerfile -t asia.gcr.io/mles-class-04/text-detection-triton:latest

#run server
docker run --shm-size=1g --rm --net=host -v ${PWD}/model_repository:/models asia.gcr.io/mles-class-04/text-detection-triton:latest tritonserver --model-repository=/models

