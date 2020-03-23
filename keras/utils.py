import numpy as np
import os
import os.path as osp
import argparse

Config ={}
# you should replace it with your own root_path
Config['root_path'] = 'I:/Data/polyvore_outfits_hw/polyvore_outfits'
#Config['root_path'] = '/home/ubuntu/Desktop/Data/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''


Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 100
Config['batch_size'] = 8

Config['learning_rate'] = 0.001
Config['num_workers'] = 5

