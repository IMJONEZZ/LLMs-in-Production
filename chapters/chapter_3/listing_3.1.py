import copy
import torch
import torch.ao.quantization as q

# deep copy the original model as quantization is done in place
model_to_quantize = copy.deepcopy(model_fp32)
model_to_quantize.eval()

# get mappings - note use “qnnpack” for ARM and “fbgemm” for x86 CPU
model_to_quantize.qconfig = q.get_default_qconfig("qnnpack")

# prepare
prepared_model = q.prepare(model_to_quantize)

# calibrate - you’ll want to use representative (validation) data.
with torch.inference_mode():
    for x in dataset:
        prepared_model(x)

# quantize
model_quantized = q.convert(prepared_model)
