# ADAPTED FROM https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html  # noqa E501
# and https://discuss.pytorch.org/t/how-to-modify-deeplabv3-and-fcn-models-for-grayscale-images/52688  # noqa E501

import torch
from mmv_im2im.utils.misc import parse_config_func_without_params


class Net(torch.nn.Module):
    def __init__(
        self,
        backbone,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
        in_channels: int = 3,
        num_classes: int = 21,
        aux_loss: bool = None,
    ):
        super().__init__()
        params = {
            "progress": False,
            "num_classes": num_classes,
            "aux_loss": aux_loss,
            "pretrained": pretrained,
            "pretrained_backbone": pretrained_backbone,
        }
        info = {"module_name": "torchvision.models.segmentation", "params": params}
        if backbone == "deeplabv3_resnet50":
            info["func_name"] = "deeplabv3_resnet50"
            my_func = parse_config_func_without_params(info)
            self.net = my_func(**info["params"])
        elif backbone == "deeplabv3_resnet101":
            info["func_name"] = "deeplabv3_resnet101"
            my_func = parse_config_func_without_params(info)
            self.net = my_func(**info["params"])
        elif backbone == "deeplabv3_mobilenet_v3_large":
            info["func_name"] = "deeplabv3_mobilenet_v3_large"
            my_func = parse_config_func_without_params(info)
            self.net = my_func(**info["params"])
        if in_channels != 3:
            self.net.backbone.conv1 = torch.nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    def forward(self, x):
        y_hat = self.net(x)
        return y_hat["out"]
