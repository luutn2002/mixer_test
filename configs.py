# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ml_collections import ConfigDict
import os
from utils import get_image_aug

def get_mixer_b16_config():
    """Returns Mixer-B/16 configuration."""
    config = ConfigDict()
    
    #mlp block config
    config.token_input_dim = 196
    config.channel_input_dim = 768
    config.token_hidden_dim = 384
    config.channel_hidden_dim = 3072
    config.layer_norm_dim = 768
    
    #dense-mix block config
    config.using_gg_mixer_block = False
    config.unit_of_dense = 12
    
    #prepatch config
    config.input_image_channel = 3
    config.pre_patch_hidden_size = 768
    config.batch_size = 16
    
    #transposition config
    config.transpose_segmentate_dimension = 50
    config.conv_segmentate_dimension = 10
    config.mixer_np_data = "npz_pretrained_model/imagenet1k_Mixer-B_16.npz"
    config.output_mask_channel = 1
    config.multiplied_dimension = (config.unit_of_dense+1)*config.pre_patch_hidden_size
    return config

def get_512_modified_mixer_b16_config():
    """Returns Mixer-B/16 configuration."""
    config = ConfigDict()
    
    #mlp block config
    config.token_input_dim = 196
    config.channel_input_dim = 768
    config.token_hidden_dim = 384
    config.channel_hidden_dim = 3072
    config.layer_norm_dim = 768
    
    #dense-mix block config
    config.using_gg_mixer_block = False
    config.unit_of_dense = 12
    
    #prepatch config
    config.input_image_channel = 3
    config.pre_patch_hidden_size = 768
    config.batch_size = 16
    
    #transposition config
    config.transpose_segmentate_dimension = 100
    config.conv_segmentate_dimension = 50
    config.mixer_np_data = "npz_pretrained_model/imagenet1k_Mixer-B_16.npz"
    config.output_mask_channel = 34
    config.multiplied_dimension = (config.unit_of_dense+1)*config.pre_patch_hidden_size
    return config

                
def get_medical_dataset_config():
    
    config = ConfigDict()
    
    config.img_dir = os.path.join(os.getcwd()+'/sessile-main-Kvasir-SEG/images/')
    config.mask_dir = os.path.join(os.getcwd() + '/sessile-main-Kvasir-SEG/masks/')
    
    #Apply augmentation that does not affect the mask here
    config.image_only_transform = get_image_aug()
    
    #Apply augmentation that does not affect the image here
    config.mask_only_transform = None 
    
    #Enable simple augmentation for both image and mask
    config.enable_parallel_transform = True
    
    return config

def get_medical_dataloader_config():
    
    config = ConfigDict()
    
    config.batch_size = 14
    config.validation_split = .2
    config.shuffle_dataset = True
    config.random_seed= 42
    
    return config

def get_cityscapes_config():
    config = ConfigDict()
    
    config.data_dir = './cityscapes'
    config.data_mode = 'fine'
    config.data_target_type = 'semantic'
    config.batch_size = 50

    return config

def get_MultiPerceptiveMixer_config_224():
    config_gen = ConfigDict()
    config_16 = ConfigDict()
    config_32 = ConfigDict()
    config_56 = ConfigDict()
    config_112 = ConfigDict()
    
    config_16.pre_patch_hidden_size = 768
    config_16.batch_size = 16
    config_16.token_input_dim = 196
    config_16.channel_input_dim = config_16.pre_patch_hidden_size
    config_16.token_hidden_dim = 384
    config_16.channel_hidden_dim = 3072
    config_16.layer_norm_dim = config_16.channel_input_dim
    config_16.unit_count = 12
    
    config_32.pre_patch_hidden_size = 768
    config_32.batch_size = 32
    config_32.token_input_dim = 49
    config_32.channel_input_dim = config_32.pre_patch_hidden_size
    config_32.token_hidden_dim = 96
    config_32.channel_hidden_dim = 768
    config_32.layer_norm_dim = config_32.channel_input_dim
    config_32.unit_count = 12
    
    config_56.pre_patch_hidden_size = 768
    config_56.batch_size = 56
    config_56.token_input_dim = 16
    config_56.channel_input_dim = config_56.pre_patch_hidden_size
    config_56.token_hidden_dim = 32
    config_56.channel_hidden_dim = 256
    config_56.layer_norm_dim = config_56.channel_input_dim
    config_56.unit_count = 12
    
    config_112.pre_patch_hidden_size = 768
    config_112.batch_size = 112
    config_112.token_input_dim = 4
    config_112.channel_input_dim = config_112.pre_patch_hidden_size
    config_112.token_hidden_dim = 8
    config_112.channel_hidden_dim = 64
    config_112.layer_norm_dim = config_112.channel_input_dim
    config_112.unit_count = 12
    
    config_gen.input_image_channel = 3
    config_gen.num_class = 1000
    config_gen.classifier_input = config_16.pre_patch_hidden_size + config_32.pre_patch_hidden_size\
                                + config_56.pre_patch_hidden_size + config_112.pre_patch_hidden_size
    
    return config_gen, config_16, config_32, config_56, config_112

def get_imagenet_config():
    config = ConfigDict()
    
    config.train_data_dir = '/home/ccl/MixerPyramid/imagenet/train'
    config.val_data_dir = '/home/ccl/MixerPyramid/imagenet/val'
    config.batch_size = 200

    return config