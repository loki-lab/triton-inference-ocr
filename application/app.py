import gradio as gr
from client import Client


class Application:
    def __init__(self, client):
        self.interface = gr.Interface
        self.client = client

    def run(self):
        app = self.interface(self.client.inference, inputs=["image"], outputs=["text", "text"])
        app.launch()


if __name__ == "__main__":
    url = "192.168.252.128:8000"
    infer = Client(url)
    gradio_app = Application(infer)
    gradio_app.run()
