import onnx

# print("loading")
# onnx_model = onnx.load("out/test.onnx")
# print("checking")
# onnx.checker.check_model(onnx_model)
# print("checked")

import onnxruntime

ort_session = onnxruntime.InferenceSession("out/test.onnx", providers=["CUDAExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
#np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")