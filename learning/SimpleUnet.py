import torch
import torch.nn as nn

from functools import partial


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1,
                 use_batch_norm=True,
                 num_convs=2):
        super(ConvBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.module_dict = nn.ModuleDict()
        for i in range(num_convs):
            conv_block = nn.Conv3d(in_channels=in_channels if i == 0 else out_channels,
                                   out_channels=out_channels,
                                   kernel_size=k_size,
                                   stride=stride,
                                   padding=padding)
            self.module_dict[f"conv_{i}"] = conv_block
            if use_batch_norm:
                self.module_dict[f"bn_{i}"] = nn.BatchNorm3d(num_features=out_channels)
            self.module_dict[f"prelu_{i}"] = nn.PReLU()

    def forward(self, x):
        for k, op in self.module_dict.items():
            # print(k, x.shape)
            x = op(x)
            # print(k, x.shape)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2, num_feat_maps=16, num_convs=2):
        """
        Model depth correspond to the number of convolution block, which is one more than the max pool
        x_0 -> conv_0 -> maxpool_0 -> x_1
        x_1 -> conv_1 -> maxpool_1 -> x_2
        ...
        x_d-1 -> conv_d -> x_d

        Therefore, we have 5xi, 4 convs and 3MP
        """
        super(EncoderBlock, self).__init__()
        self.num_feat_maps = num_feat_maps
        self.num_conv_blocks = num_convs
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            # Compute output size
            feat_map_channels = self.num_feat_maps * 2 ** depth
            self.module_dict[f"block_{depth}"] = ConvBlock(in_channels=in_channels,
                                                           out_channels=feat_map_channels,
                                                           num_convs=num_convs)
            in_channels = feat_map_channels
            if depth < model_depth - 1:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        downsampling_features = []
        for k, op in self.module_dict.items():
            # print(k, x.shape)
            if k.startswith("block"):
                x = op(x)
                downsampling_features.append(x)
                # print([x.shape for x in downsampling_features])
            elif k.startswith("max_pooling"):
                x = op(x)
            # print(k, x.shape)
            # print()
        return x, downsampling_features


# Just to get default values => less verbose
class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=2, padding=1, output_padding=1):
        super(ConvTranspose, self).__init__()
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=in_channels,
                                                   out_channels=out_channels,
                                                   kernel_size=k_size,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

    def forward(self, x):
        return self.conv3d_transpose(x)


class DecoderBlock(nn.Module):
    """
    Constructs back the input like grids from the condensed representation and the intermediate values.

    Since : x_0 -> conv_0 -> maxpool_0 -> x_1
    We define CT_i as the one which has output at depth i.
    """

    def __init__(self, out_channels, model_depth=4, num_feat_maps=16, max_decode=0, num_convs=2):
        super(DecoderBlock, self).__init__()
        self.num_feat_maps = num_feat_maps
        self.module_dict = nn.ModuleDict()

        # Only decode until a certain depth of max_decode
        for depth in range(model_depth - 2, max_decode - 1, - 1):
            # This is the size expected at this depth.
            # The input and output for deconv will be twice that, so after concat we get three times that
            feat_map_channels = self.num_feat_maps * 2 ** depth
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
            self.module_dict[f"deconv_{depth}"] = self.deconv
            self.conv = ConvBlock(in_channels=feat_map_channels * 3,
                                  out_channels=feat_map_channels,
                                  num_convs=num_convs)
            self.module_dict[f"block_{depth}"] = self.conv
            if depth == max_decode:
                self.final_conv = ConvBlock(in_channels=feat_map_channels, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, downsampling_features):
        """
        :param x: inputs
        :param downsampling_features: feature maps from encoder path
        :return: output
        """

        for k, op in self.module_dict.items():
            # print(k, x.shape)
            if k.startswith("deconv"):
                x = op(x)
                # If the input has a shape that is not a power of 2, we need to pad when deconvoluting
                *_, dx, dy, dz = [a - b for a, b in zip((downsampling_features[int(k[-1])].shape), (x.shape))]
                pad_size = [0, dz, 0, dy, 0, dx]
                x = torch.nn.functional.pad(x, pad=pad_size, mode='constant', value=0)
                x = torch.cat((downsampling_features[int(k[-1])], x), dim=1)
            else:
                x = op(x)
            # print(k, x.shape)
        return x


def bi_pred_head(x):
    categorical_slices_x = x[..., :3, :, :, :]
    regression_slices_x = x[..., 3:, :, :, :]
    softm = nn.functional.softmax(categorical_slices_x, dim=1)
    output = torch.cat([softm, regression_slices_x], dim=1)
    return output


class UnetModel(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels_decoder=32,
                 predict_mse=False,
                 model_depth=4,
                 num_feature_map=16,
                 ):
        super(UnetModel, self).__init__()
        self.num_feat_maps = num_feature_map
        self.encoder = EncoderBlock(in_channels=in_channels,
                                    model_depth=model_depth,
                                    num_feat_maps=self.num_feat_maps)
        self.decoder = DecoderBlock(out_channels=out_channels_decoder,
                                    model_depth=model_depth,
                                    num_feat_maps=self.num_feat_maps)
        out_channels_network = 5 if predict_mse else 3

        mid_channels = max(out_channels_decoder // 2, 5)
        self.conv3d_final = nn.Sequential(
            nn.Conv3d(in_channels=out_channels_decoder, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=mid_channels, out_channels=out_channels_network, kernel_size=1),
        )
        if predict_mse:
            self.final_act = bi_pred_head
        else:
            self.final_act = partial(nn.functional.softmax, dim=1)

    def forward(self, x):
        mid, downsampling_features = self.encoder(x)
        x = self.decoder(mid, downsampling_features=downsampling_features)
        x = self.conv3d_final(x)
        out = self.final_act(x)
        return out


class SimpleHalfUnetModel(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels_decoder=128,
                 out_channels=9,
                 classif_nano=False,
                 model_depth=4,
                 num_feature_map=16,
                 max_decode=2,
                 num_convs=2
                 ):
        super(SimpleHalfUnetModel, self).__init__()
        self.num_feat_maps = num_feature_map
        self.encoder = EncoderBlock(in_channels=in_channels,
                                    model_depth=model_depth,
                                    num_feat_maps=self.num_feat_maps,
                                    num_convs=num_convs)
        self.decoder = DecoderBlock(out_channels=out_channels_decoder,
                                    model_depth=model_depth,
                                    num_feat_maps=self.num_feat_maps,
                                    max_decode=max_decode,
                                    num_convs=num_convs)

        self.classif_nano = classif_nano
        self.final_conv = nn.Conv3d(in_channels=out_channels_decoder,
                                    out_channels=out_channels + 1 if classif_nano else out_channels,
                                    kernel_size=1)

    def forward(self, x):
        mid, downsampling_features = self.encoder(x)
        x = self.decoder(mid, downsampling_features=downsampling_features)
        x = self.final_conv(x)
        x[0, 0, ...] = torch.sigmoid(x[0, 0, ...])
        if self.classif_nano:
            x[0, -1, ...] = torch.sigmoid(x[0, -1, ...])
        return x


if __name__ == '__main__':
    # in_shape = (16,) * 3
    in_shape = (53, 73, 58)
    grid_em = torch.ones((1, 1, *in_shape), dtype=torch.float32)

    # model = UnetModel(in_channels=1)
    # out = model(grid_em)
    # print(out.shape)

    model = SimpleHalfUnetModel(in_channels=1,
                                model_depth=4,
                                num_convs=3,
                                max_decode=2,
                                num_feature_map=32)
    a = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    print(a)
    # out = model(grid_em)
    # print(out.shape)
    # print(model.encoder.module_dict)
