import gradio as gr
from client import Client
import numpy as np


class Application:
    def __init__(self, client1, client2):
        self.gr = gr
        self.client1 = client1
        self.client2 = client2

    def inference_form1(self, event):
        input_img = event["composite"]
        input_img = np.asarray(input_img)
        img_crop, list_text = self.client1.inference_e2e(input_img)

        return list_text

    def inference_form2(self, event):
        input_img = event["composite"]
        input_img = np.asarray(input_img)
        img_crop, list_text = self.client2.inference_e2e(input_img)

        return list_text

    def run(self):
        with self.gr.Blocks() as app:
            with self.gr.Blocks():
                with gr.Tab("English"):
                    with gr.Tab("English Detection"):
                        self.gr.Interface(self.client1.inference_e2e, inputs=["image"], outputs=["image", "text"])
                    with gr.Tab("Process image"):
                        im = self.gr.ImageEditor(type="numpy", image_mode="RGB")
                        self.gr.Interface(self.inference_form1, inputs=im, outputs=["text"])

                with gr.Tab("Japanese"):
                    with gr.Tab("Japanese Detection"):
                        self.gr.Interface(self.client2.inference_e2e, inputs=["image"], outputs=["image", "text"])
                    with gr.Tab("Process image"):
                        im = self.gr.ImageEditor(type="numpy", image_mode="RGB")
                        self.gr.Interface(self.inference_form2, inputs=im, outputs=["text"])

            with self.gr.Blocks():
                with self.gr.Tab("Confident score"):
                    self.gr.Markdown("plot")

        app.launch(share=True, server_port=8080)


if __name__ == "__main__":
    url1 = "34.16.181.202:8000"
    url2 = "34.121.147.116:8000"
    infer1 = Client(url1)
    infer2 = Client(url2)
    gradio_app = Application(infer1, infer2)
    gradio_app.run()
