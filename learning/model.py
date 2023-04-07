import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, use_batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv3d_a = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                  stride=stride, padding=padding)
        self.conv3d_b = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                                  stride=stride, padding=padding)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.conv3d_a(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.conv3d_b(x)
        x = F.elu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2, num_feat_maps=16):
        super(EncoderBlock, self).__init__()
        self.num_feat_maps = num_feat_maps
        self.num_conv_blocks = 2
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                # print(in_channels, feat_map_channels)
                self.conv_block = ConvBlock(in_channels=in_channels,
                                            out_channels=feat_map_channels,
                                            use_batch_norm=depth < 2)
                self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth < model_depth - 1:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        downsampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                # print([x.shape for x in downsampling_features])
                if k.endswith("1"):
                    downsampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)
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
    The final size of the map is
    """

    def __init__(self, out_channels, model_depth=4, num_feat_maps=16):
        super(DecoderBlock, self).__init__()
        self.num_conv_blocks = 2
        self.num_feat_maps = num_feat_maps
        # user nn.ModuleDict() to store ops and the fact that the order is kept
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps
            self.deconv = ConvTranspose(in_channels=feat_map_channels * 4, out_channels=feat_map_channels * 4)
            self.module_dict["deconv_{}".format(depth)] = self.deconv
            for i in range(self.num_conv_blocks):
                if i == 0:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 6, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
                else:
                    self.conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=feat_map_channels * 2)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv
            if depth == 0:
                self.final_conv = ConvBlock(in_channels=feat_map_channels * 2, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, downsampling_features):
        """
        :param x: inputs
        :param downsampling_features: feature maps from encoder path
        :return: output
        """

        for k, op in self.module_dict.items():
            if k.startswith("deconv"):
                x = op(x)
                # If the input has a shape that is not a power of 2, we need to pad when deconvoluting
                *_, dx, dy, dz = [a - b for a, b in zip((downsampling_features[int(k[-1])].shape), (x.shape))]
                pad_size = [0, dz, 0, dy, 0, dx]
                x = torch.nn.functional.pad(x, pad=pad_size, mode='constant', value=0)
                x = torch.cat((downsampling_features[int(k[-1])], x), dim=1)
            elif k.startswith("conv"):
                x = op(x)
            else:
                x = op(x)
        return x


def unet_from_hparams(hparams, load_weights=False):
    model_depth = hparams.get('argparse', 'model_depth')
    num_feature_map = hparams.get('argparse', 'num_feature_map')
    unet = UnetModel(model_depth=model_depth,
                     num_feature_map=num_feature_map)
    if load_weights:
        unet.load_weights(hparams=hparams)
    return unet


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
                 num_feature_map=16, ):
        super(UnetModel, self).__init__()
        self.num_feat_maps = num_feature_map
        self.encoder = EncoderBlock(in_channels=in_channels,
                                    model_depth=model_depth,
                                    num_feat_maps=self.num_feat_maps)
        self.decoder = DecoderBlock(out_channels=out_channels_decoder,
                                    model_depth=model_depth,
                                    num_feat_maps=self.num_feat_maps)
        mid_channels = max(out_channels_decoder // 2, 5)
        out_channels_network = 5 if predict_mse else 3
        self.conv3d_final = nn.Sequential(
            nn.Conv3d(in_channels=out_channels_decoder, out_channels=mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=mid_channels, out_channels=out_channels_network, kernel_size=1),
        )
        if predict_mse:
            self.final_act = partial(nn.functional.softmax, dim=1)
        else:
            self.final_act = bi_pred_head

    def forward(self, x):
        mid, downsampling_features = self.encoder(x)
        x = self.decoder(mid, downsampling_features=downsampling_features)
        x = self.conv3d_final(x)
        out = self.final_act(x)
        return out


if __name__ == '__main__':
    model = UnetModel(in_channels=1)
    in_shape = (16,) * 3
    grid_em, grid_target = torch.ones((1, 1, *in_shape), dtype=torch.float32), \
        torch.ones((1, 3, *in_shape), dtype=torch.float32)
    out = model(grid_em)
    print(out.shape)
    # out, loss = model.persistent_training_call(grid_prot, grid_hd, 0)
    # out, loss = model.persistent_training_call(grid_prot, grid_pl, 1)
