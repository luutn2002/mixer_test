from os.path import join as pjoin

import configs
from einops import rearrange
from numpy import load, arange
from torch import cat, from_numpy, swapaxes, mean, no_grad
from torch.nn import Linear, GELU, LayerNorm, Conv2d, ConvTranspose2d, Module, ModuleDict, Sigmoid, BatchNorm2d, Threshold, Sequential
from torch.nn.init import xavier_normal_

# from collections import OrderedDict

TOK_FC_0 = "token_mixing/Dense_0"
TOK_FC_1 = "token_mixing/Dense_1"
CHA_FC_0 = "channel_mixing/Dense_0"
CHA_FC_1 = "channel_mixing/Dense_1"
PRE_NORM = "LayerNorm_0"
POST_NORM = "LayerNorm_1"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return from_numpy(weights)


class MlpBlock(Module):
    def __init__(self, input_dim, hidden_dim):
        super(MlpBlock, self).__init__()
        self.fc0 = Linear(input_dim, hidden_dim)
        self.fc1 = Linear(hidden_dim, input_dim)
        self.act_fn = GELU()

    def forward(self, x):
        x = self.fc0(x)
        x = self.act_fn(x)
        x = self.fc1(x)
        return x


class MixerBlock(Module):
    def __init__(self, config):
        super(MixerBlock, self).__init__()
        self.token_mlp_block = MlpBlock(config.token_input_dim, config.token_hidden_dim)
        self.channel_mlp_block = MlpBlock(config.channel_input_dim, config.channel_hidden_dim)
        self.pre_norm = LayerNorm(config.layer_norm_dim, eps=1e-6)
        self.post_norm = LayerNorm(config.layer_norm_dim, eps=1e-6)

    def forward(self, x):
        h = x
        x = self.pre_norm(x)
        x = x.transpose(-1, -2)
        x = self.token_mlp_block(x)
        x = x.transpose(-1, -2)
        x = x + h

        h = x
        x = self.post_norm(x)
        x = self.channel_mlp_block(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
        ROOT = f"MixerBlock_{n_block}"
        with no_grad():
            self.token_mlp_block.fc0.weight.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_0, "kernel")]).t())
            self.token_mlp_block.fc1.weight.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_1, "kernel")]).t())
            self.token_mlp_block.fc0.bias.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_0, "bias")]).t())
            self.token_mlp_block.fc1.bias.copy_(
                np2th(weights[pjoin(ROOT, TOK_FC_1, "bias")]).t())

            self.channel_mlp_block.fc0.weight.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_0, "kernel")]).t())
            self.channel_mlp_block.fc1.weight.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_1, "kernel")]).t())
            self.channel_mlp_block.fc0.bias.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_0, "bias")]).t())
            self.channel_mlp_block.fc1.bias.copy_(
                np2th(weights[pjoin(ROOT, CHA_FC_1, "bias")]).t())

            self.pre_norm.weight.copy_(np2th(weights[pjoin(ROOT, PRE_NORM, "scale")]))
            self.pre_norm.bias.copy_(np2th(weights[pjoin(ROOT, PRE_NORM, "bias")]))
            self.post_norm.weight.copy_(np2th(weights[pjoin(ROOT, POST_NORM, "scale")]))
            self.post_norm.bias.copy_(np2th(weights[pjoin(ROOT, POST_NORM, "bias")]))


class DenseMixerBlock(Module):
    def __init__(self,
                 config,
                 mixer_npz_data):
        super(DenseMixerBlock, self).__init__()

        """
        This layer concatennate result over each layer of mixer.
        """

        #self.using_gg_mixer_block = config.using_gg_mixer_block

        if config.unit_of_dense > 12:
            raise ValueError(f"Dense unit in config file is too high,\
                                    limit is 11, receiving:{config.unit_of_dense}")

        #self.units = config.unit_of_dense
        self.unit_mixer = MixerBlock(config)

        if config.using_gg_mixer_block:
            data = load(mixer_npz_data)
            module_list = {}

            for i in range(config.unit_of_dense):
                self.unit_mixer.load_from(data, i)
                module_list[f"mixer_block_{i}"] = self.unit_mixer

            self.mixer_blocks = ModuleDict(module_list)
            # print(self.mixer_blocks)

        else:
            module_list = {}
            for i in range(config.unit_of_dense):
                module_list[f"mixer_block_{i}"] = self.unit_mixer

            self.mixer_blocks = ModuleDict(module_list)
            # print(self.mixer_blocks)

    def forward(self, x):

        y = self.mixer_blocks["mixer_block_0"](x)
        result = cat((x.unsqueeze(1), y.unsqueeze(1)), 1)
        for i in range(1, len(self.mixer_blocks)):
            y = self.mixer_blocks[f"mixer_block_{i}"](y)
            result = cat((result, y.unsqueeze(1)), 1)

        return result


def glorot_normal_initialize(x):
    xavier_normal_(x.weight)
    return x


class DenseTransMiSS(Module):
    def __init__(self,
                 config,
                 mixer_npz_data):
        super(DenseTransMiSS, self).__init__()

        self.config = config
        self.stem = Conv2d(config.input_image_channel,
                           config.pre_patch_hidden_size,
                           config.batch_size,
                           stride=config.batch_size)

        self.stem = glorot_normal_initialize(self.stem)

        self.conv_transpose = ConvTranspose2d(config.multiplied_dimension,
                                              config.transpose_segmentate_dimension,
                                              config.batch_size,
                                              stride=self.config.batch_size)

        self.conv_transpose = glorot_normal_initialize(self.conv_transpose)

        self.stem_2 = Conv2d(config.transpose_segmentate_dimension,
                             config.conv_segmentate_dimension,
                             config.batch_size,
                             padding='same')

        self.stem_2 = glorot_normal_initialize(self.stem_2)

        self.head = Linear(config.conv_segmentate_dimension,
                           config.output_mask_channel)

        self.head = glorot_normal_initialize(self.head)

        self.dense_mixer_block = DenseMixerBlock(config, mixer_npz_data)

        #self.pre_head_layer_norm = LayerNorm((13, 196, 768))
        self.pre_head_batch_norm = BatchNorm2d(config.multiplied_dimension)
        #self.final_activation = Sigmoid()
        #self.thresh = Threshold(0.8, 0.0)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)

        x = self.stem(x)
        wide = x.size(-2)
        x = swapaxes(x, -1, -3)  # change to channel last data
        x = rearrange(x, 'n h w c -> n (h w) c')
        x = self.dense_mixer_block(x)
        #x = self.pre_head_layer_norm(x)
        x = swapaxes(x, -1, -2)
        x = rearrange(x, 'n h w c -> n (h w) c')
        x = swapaxes(x, -1, -2)
        #x = mean(x, axis=1) # sum through concatenated dimesion
        x = rearrange(x, 'b  (w h) c -> b w h c', w=wide)  # restore original form
        x = swapaxes(x, -1, -3)  # back to channel first data
        x = self.pre_head_batch_norm(x)
        #x = mean(x, axis=0)  
        #x = x.unsqueeze(0)  # unsqueeze to receive 4 dimension tensor with batch size = 1
        x = self.conv_transpose(x)
        x = self.stem_2(x)
        x = swapaxes(x, -1, -3)  # swap axes to get channel last form to affine transform over last dimesion which is channel
        x = self.head(x)  # result
        x = swapaxes(x, -1, -3)
        #x = self.final_activation(x)

        return x


def get_model(config):
    return DenseTransMiSS(config, config.mixer_np_data)

class MultiPerpectiveMixer(Module):
    def __init__(self,
                 config):
        super(MultiPerpectiveMixer, self).__init__()
        
        config_gen, config_16, config_32, config_56, config_112 = config
        
        self.prepatch_16 = Conv2d(config_gen.input_image_channel,
                                   config_16.pre_patch_hidden_size,
                                   config_16.batch_size,
                                   stride=config_16.batch_size)
        
        self.prepatch_32 = Conv2d(config_gen.input_image_channel,
                                   config_32.pre_patch_hidden_size,
                                   config_32.batch_size,
                                   stride=config_32.batch_size)
        
        self.prepatch_56 = Conv2d(config_gen.input_image_channel,
                                   config_56.pre_patch_hidden_size,
                                   config_56.batch_size,
                                   stride=config_56.batch_size)
        
        self.prepatch_112 = Conv2d(config_gen.input_image_channel,
                                   config_112.pre_patch_hidden_size,
                                   config_112.batch_size,
                                   stride=config_112.batch_size)
        
        self.unit_16 = MixerBlock(config_16)
        self.unit_32 = MixerBlock(config_32)
        self.unit_56 = MixerBlock(config_56)
        self.unit_112 = MixerBlock(config_112)
        
        self.block_16 = Sequential(*[self.unit_16 for _ in arange(config_16.unit_count)])
        self.block_32 = Sequential(*[self.unit_32 for _ in arange(config_32.unit_count)])
        self.block_56 = Sequential(*[self.unit_56 for _ in arange(config_56.unit_count)])
        self.block_112 = Sequential(*[self.unit_112 for _ in arange(config_112.unit_count)])
        
        self.norm_16 = LayerNorm(config_16.layer_norm_dim)
        self.norm_32 = LayerNorm(config_32.layer_norm_dim)
        self.norm_56 = LayerNorm(config_56.layer_norm_dim)
        self.norm_112 = LayerNorm(config_112.layer_norm_dim)
        
        self.classifier = Linear(config_gen.classifier_input, config_gen.num_class)
        
    def forward(self, x):
        
        x16 = self.prepatch_16(x)
        x32 = self.prepatch_32(x)
        x56 = self.prepatch_56(x)
        x112 = self.prepatch_112(x)
        
        x16 = swapaxes(x16, -1, -3)
        x16 = rearrange(x16, 'n h w c -> n (h w) c')
        x16 = self.block_16(x16)
        x16 = self.norm_16(x16)
        x16 = mean(x16, axis=-2)
        
        x32 = swapaxes(x32, -1, -3)
        x32 = rearrange(x32, 'n h w c -> n (h w) c')
        x32 = self.block_32(x32)
        x32 = self.norm_32(x32)
        x32 = mean(x32, axis=-2)
        
        x56 = swapaxes(x56, -1, -3)
        x56 = rearrange(x56, 'n h w c -> n (h w) c')
        x56 = self.block_56(x56)
        x56 = self.norm_56(x56)
        x56 = mean(x56, axis=-2)
        
        x112 = swapaxes(x112, -1, -3)
        x112 = rearrange(x112, 'n h w c -> n (h w) c')
        x112 = self.block_112(x112)
        x112 = self.norm_112(x112)
        x112 = mean(x112, axis=-2)
        
        x = cat((x16, x32, x56, x112), 1)
        
        return self.classifier(x)
        
        
        
        