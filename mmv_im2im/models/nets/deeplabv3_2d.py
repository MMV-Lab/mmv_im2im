# ADAPTED FROM https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html
# and https://discuss.pytorch.org/t/how-to-modify-deeplabv3-and-fcn-models-for-grayscale-images/52688

import torch
from mmv_im2im.utils.misc import parse_config


class Net(torch.nn.Module):
    def __init__(
        self,
        backbone,
        in_channels: int = 3,
        num_classes: int = 0,
        aux_loss: bool = None,
    ):
        super().__init__()
        params = {"progress": False, "num_classes": num_classes, "aux_loss": aux_loss}
        info = {"module_name": "torchvision.models.segmentation", "params": params}
        if backbone == "deeplabv3_resnet50":
            info["func_name"] = "deeplabv3_resnet50"
            self.net = parse_config(info)
        elif backbone == "deeplabv3_resnet101":
            info["func_name":"deeplabv3_resnet101"]
            self.net = parse_config(info)
        elif backbone == "deeplabv3_mobilenet_v3_large":
            info["func_name":"deeplabv3_mobilenet_v3_large"]
            self.net = parse_config(info)
        if in_channels != 3:
            self.net.backbone.conv1 = torch.nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat["out"]
