import functools
import torch.nn as nn

from monai.networks.blocks import ResidualUnit, Convolution
from monai.networks.nets import UNet

from mmv_im2im.utils.misc import (
    parse_config_func_without_params,
    parse_config_func,
    parse_config,
)


class preset_generator_resent(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        n_down_blocks,
        n_res_blocks,
        nf=64,
        norm_layer="INSTANCE",
        use_dropout=None,
    ):
        super().__init__()
        if spatial_dims == 2:
            first_layer = functools.partial(nn.ReflectionPad2d)
        elif spatial_dims == 3:
            first_layer = functools.partial(nn.ReplicationPad3d)
        else:
            raise NotImplementedError("only 2d and 3d data are supported")

        model = [
            first_layer(3),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=nf,
                norm=norm_layer,
                kernel_size=7,
                act="RELU",
                padding=0,
            ),
        ]

        # add downsampling layers
        for i in range(n_down_blocks):
            mult = 2**i
            model += [
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=nf * mult,
                    out_channels=nf * mult * 2,
                    kernel_size=3,
                    strides=2,
                    padding=1,
                    norm=norm_layer,
                    act="RELU",
                )
            ]

        # add ResNet blocks
        mult = 2**n_down_blocks
        for i in range(n_res_blocks):
            model += [
                ResidualUnit(
                    spatial_dims=spatial_dims,
                    in_channels=nf * mult,
                    out_channels=nf * mult,
                    kernel_size=3,
                    strides=1,
                    norm=norm_layer,
                    dropout=use_dropout,
                    act="RELU",
                )
            ]

        # add upsampling blocks
        for i in range(n_down_blocks):
            mult = 2 ** (n_down_blocks - i)
            model += [
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=nf * mult,
                    out_channels=int(nf * mult / 2),
                    kernel_size=3,
                    strides=2,
                    padding=1,
                    output_padding=1,
                    norm=norm_layer,
                    act="RELU",
                    is_transposed=True,
                )
            ]

        # add final layer
        model += [
            first_layer(3),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=nf,
                out_channels=out_channels,
                kernel_size=7,
                conv_only=True,
                padding=0,
            ),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class generator_encoder_decoder(nn.Module):
    def __init__(self, model_info):
        super().__init__()
        # initial block
        init_block = parse_config_func(model_info["init_block"])
        model = [init_block]

        # downsampling blocks
        down_block = parse_config_func_without_params(model_info["down_block"])
        down_channels = model_info["down_block"]["params"].pop("channels")
        prev_channel = model_info["init_block"]["params"]["out_channel"]
        for this_channel in down_channels:
            model.append(
                down_block(
                    in_channels=prev_channel,
                    out_channels=this_channel,
                    **model_info["down_block"]["params"],
                )
            )

        # residual block
        res_block = parse_config_func_without_params(model_info["res_block"])
        prev_channel = down_channels[-1]
        res_channels = model_info["res_block"]["params"].pop("channels")
        for this_channel in res_channels:
            model.append(
                res_block(
                    in_channels=prev_channel,
                    out_channels=this_channel,
                    **model_info["res_block"]["params"],
                )
            )

        # up blocks
        up_block = parse_config_func_without_params(model_info["up_block"])
        prev_channel = res_channels[-1]
        up_channels = model_info["up_block"]["params"].pop("channels")
        for this_channel in up_channels:
            model.append(
                up_block(
                    in_channels=prev_channel,
                    out_channels=this_channel,
                    **model_info["up_block"]["params"],
                )
            )

        # final block
        final_block = parse_config_func(model_info["final_block"])
        model.append(final_block)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def define_generator(model_info):
    if "type" in model_info:
        if model_info["type"] == "predefined_resnet":
            net = preset_generator_resent(**model_info["params"])
        elif model_info["type"] == "predefined_unet":
            net = UNet(**model_info["params"])
        elif model_info["type"] == "generic_encoder_decoder":
            net = generator_encoder_decoder(**model_info)
    else:
        net = parse_config(model_info)

    return net


class patch_discriminator(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        nf,
        n_layers,
        norm_layer,
        return_features: bool = False,
    ):
        super().__init__()

        blocks = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=nf,
                kernel_size=4,
                strides=2,
                padding=1,
                conv_only=True,
            ),  # no norm for first layer
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            blocks += [
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=nf * nf_mult_prev,
                    out_channels=nf * nf_mult,
                    kernel_size=4,
                    strides=2,
                    padding=1,
                    norm=norm_layer,
                    act=("leakyrelu", {"negative_slope": 0.2, "inplace": True}),
                )
            ]

        blocks += [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=nf * nf_mult,
                out_channels=1,
                kernel_size=4,
                strides=1,
                padding=1,
                conv_only=True,
            )
        ]

        self.return_features = return_features
        if return_features:
            self.n_blocks = len(blocks)
            for i in range(self.n_blocks):
                setattr(self, "block_{}".format(i), nn.Sequential(*blocks[i]))
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x):
        if self.return_features:
            result = [x]
            for i in range(self.n_blocks):
                block = getattr(self, "block_{}".format(i))
                result.append(block(result[-1]))

            return result[1:]  # except for the input
        else:
            return self.model(x)


class multiscale_discriminator(nn.Module):
    def __init__(self, num_discriminator, **kwargs):
        super().__init__()

        for i in range(num_discriminator):
            setattr(self, "Scale_{}".format(str(i)), patch_discriminator(**kwargs))
        self.n_D = num_discriminator
        self.spatial_dimenstion = kwargs["spatial_dims"]

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, "Scale_{}".format(i))(x))
            if i != self.n_D - 1:
                if self.spatial_dimenstion == 2:
                    x = nn.AvgPool2d(
                        kernel_size=3, padding=1, stride=2, count_include_pad=False
                    )(x)
                elif self.spatial_dimenstion == 3:
                    x = nn.AvgPool3d(
                        kernel_size=3, padding=1, stride=2, count_include_pad=False
                    )(x)
        return result


def define_discriminator(model_info):
    if model_info["type"] == "predefined_basic":
        net = patch_discriminator(**model_info["params"])
    elif model_info["type"] == "predefined_multiscale":
        net = multiscale_discriminator(**model_info["params"])
    else:
        raise NotImplementedError("only predefined discriminator is supported")

    return net
