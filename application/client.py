import tritonclient.http as httpclient
import numpy as np
from utils import draw_img


class Client:
    def __init__(self, url):
        self.url = url

    def inference_e2e(self, img):
        img = np.asarray([img])

        client = httpclient.InferenceServerClient(url=self.url)

        inputs = httpclient.InferInput("INPUT", shape=img.shape, datatype="UINT8")
        inputs.set_data_from_numpy(img, binary_data=False)

        outputs = httpclient.InferRequestedOutput("rec_texts", binary_data=False)
        bbox_outputs = httpclient.InferRequestedOutput("det_bboxes", binary_data=False)

        results = client.infer(model_name="pp_ocr", inputs=[inputs], outputs=[outputs, bbox_outputs])
        inference_texts = results.as_numpy('rec_texts')
        inference_bboxes = results.as_numpy('det_bboxes')
        img = draw_img(img[0], list_points=list(inference_bboxes[0]))

        return img, list(inference_texts[0])

    def inference_rec(self, img_crop):
        img_crop = np.asarray([img_crop])

        client = httpclient.InferenceServerClient(url=self.url)

        inputs = httpclient.InferInput("x", shape=img_crop.shape, datatype="FP32")
        inputs.set_data_from_numpy(img_crop, binary_data=False)

        outputs = httpclient.InferRequestedOutput("rec_texts", binary_data=False)

        results = client.infer(model_name="rec_pp", inputs=[inputs], outputs=[outputs])
        inference_texts = results.as_numpy('rec_texts')

        return img_crop, list(inference_texts[0])