from rknnlite.api import RKNNLite as RKNN

class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNN()
        rknn.load_rknn(model_path)
        ret = rknn.init_runtime()
        self.rknn = rknn

    def run(self, inputs):
        if not self.rknn:
            print("ERROR: RKNN has been released")
            return []

        inputs = [inputs] if not isinstance(inputs, (list, tuple)) else inputs
        result = self.rknn.inference(inputs=inputs)
        return result

    def release(self):
        self.rknn.release()
        self.rknn = None
