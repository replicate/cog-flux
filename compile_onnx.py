
from flux.util import load_flow_model
import torch



flux = load_flow_model('flux-dev')

def get_rand_inputs():
    # second img and img_id dimension is dynamic (=h * w / 256), everything else is fixed 
    img = torch.randn((1, 4096, 64), dtype=torch.bfloat16, device="cuda")
    img_ids = torch.rand((1, 4096, 3), dtype=torch.float32, device="cuda")

    txt = torch.randn((1, 512, 4096), dtype=torch.bfloat16, device="cuda")
    txt_ids = torch.rand((1, 512, 3), dtype=torch.float32, device="cuda")

    vec = torch.randn((1, 768), dtype=torch.bfloat16, device="cuda")
    guidance = torch.randn((1), dtype=torch.bfloat16, device="cuda") 
    timestep = torch.randn((1), dtype=torch.bfloat16, device="cuda")
    torch.cat((txt_ids, img_ids), dim=1)

    # don't have to be passed as dict, can also just do basic args in this order: 
    # img: Tensor,
    # img_ids: Tensor,
    # txt: Tensor,
    # txt_ids: Tensor,
    # timesteps: Tensor,
    # y: Tensor, - this is "vec"
    # guidance: Tensor | None = None,
    return (img, img_ids, txt, txt_ids, timestep, vec, guidance)
    # return {"img": img, "img_id": img_id, "txt": txt, "txt_ids": txt_ids, "vec": vec, "guidance": guidance, "timestep": timestep}


# model opt stuff
with torch.inference_mode():
    #flux = torch.compile(flux, 'max-autotune', dynamic=False, fullgraph=False)

    inputs = get_rand_inputs()

    input_names = ["img", "img_ids", "txt", "txt_ids", "timestep", "y", "guidance"]
    output_names = ["pred"]
    dynamic_axes = {"img": {1: "h*w"}, "img_ids": {1: "h*w"}, "pred": {1: "h*w"}}
    path = "./out/test.onnx"

    torch.onnx.export(flux,
                    inputs,
                    path,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes
                    )