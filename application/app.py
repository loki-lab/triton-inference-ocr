from client import Client
import gradio as gr
import numpy as np
import pandas as pd


plot = gr.BarPlot(x="text", y="score", vertical=False)


class Application:
    def __init__(self, client1, client2):
        self.gr = gr
        self.client1 = client1
        self.client2 = client2

    def inference_form1(self, event):
        input_img = event["composite"]
        input_img = np.asarray(input_img)
        img_crop, list_text, rec_scores = self.client1.inference_e2e(input_img)
        df = pd.DataFrame({"text": list_text, "score": rec_scores[0]})

        return list_text, df

    def inference_form2(self, event):
        input_img = event["composite"]
        input_img = np.asarray(input_img)
        img_crop, list_text, rec_scores = self.client2.inference_e2e(input_img)
        df = pd.DataFrame({"text": list_text, "score": rec_scores[0]})

        return list_text, df

    def inference_form3(self, inputs):
        img, text, score = self.client1.inference_e2e(inputs)
        df = pd.DataFrame({"text": text, "score": score[0]})

        return img, text, df

    def inference_form4(self, inputs):
        img, text, score = self.client2.inference_e2e(inputs)
        df = pd.DataFrame({"text": text, "score": score[0]})
        print(text)
        print(score[0])

        return img, text, df

    def run(self):
        with self.gr.Blocks() as app:
            with self.gr.Blocks():
                with gr.Tab("English"):
                    with gr.Tab("English Detection"):
                        self.gr.Interface(self.inference_form3, inputs=["image"],
                                          outputs=["image", "text", plot])

                    with gr.Tab("Process image"):
                        im = self.gr.ImageEditor(type="numpy", image_mode="RGB")
                        self.gr.Interface(self.inference_form1, inputs=im, outputs=["text", plot])

                with gr.Tab("Japanese"):
                    with gr.Tab("Japanese Detection"):
                        self.gr.Interface(self.inference_form4, inputs=["image"],
                                          outputs=["image", "text", plot])
                    with gr.Tab("Process image"):
                        im = self.gr.ImageEditor(type="numpy", image_mode="RGB")
                        self.gr.Interface(self.inference_form2, inputs=im, outputs=["text", plot])

        app.launch(server_name="0.0.0.0", server_port=8000, debug=True, share=True)


if __name__ == "__main__":
    url1 = "34.125.26.33:8000"
    url2 = "34.66.163.162:8000"

    infer1 = Client(url1)
    infer2 = Client(url2)
    gradio_app = Application(infer1, infer2)
    gradio_app.run()
