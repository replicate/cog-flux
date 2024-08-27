
# Imports flux-schnell model from predict.py
# from flux.util import load_flow_model
# flux = load_flow_model("flux-schnell", device="cuda")
from predict import SchnellPredictor
import torch

p = SchnellPredictor()
p.setup()

p.flux = torch.compile(p.flux)

# Torch.compiles flux model

# actually we're just going to do this in the predictor, aren't we. this is more of a scratchpad

# range of batch dimension sizes in the input from 3808 -> 4096, or to 8100 if we go full range


# 1024:1024
# torch.Size([1, 16, 128, 128])
# torch.Size([1, 4096, 64])
# _
# 1344:768
# torch.Size([1, 16, 168, 96])
# torch.Size([1, 4032, 64])
# _
# 1536:640
# torch.Size([1, 16, 192, 80])
# torch.Size([1, 3840, 64])
# _
# 1216:832
# torch.Size([1, 16, 152, 104])
# torch.Size([1, 3952, 64])
# _
# 832:1216
# torch.Size([1, 16, 104, 152])
# torch.Size([1, 3952, 64])
# _
# 896:1088
# torch.Size([1, 16, 112, 136])
# torch.Size([1, 3808, 64])
# _
# 1088:896
# torch.Size([1, 16, 136, 112])
# torch.Size([1, 3808, 64])
# _
# 768:1344
# torch.Size([1, 16, 96, 168])
# torch.Size([1, 4032, 64])
# _
# 640:1536
# torch.Size([1, 16, 80, 192])
# torch.Size([1, 3840, 64])
# _
# 1440:1440
# torch.Size([1, 16, 180, 180])
# torch.Size([1, 8100, 64])
