#build application
docker build . -f Dockerfile -t gradio-ocr-application:latest

#run application
docker run --rm -p 8080:7860 --name gradio-ocr-application-container-1 gradio-ocr-application
