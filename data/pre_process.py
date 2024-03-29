import os
from glob import glob

import cdflib
import numpy as np

from commons.Human36 import Human36mDataset
from commons.camera import world_to_camera, project_to_2d, image_coordinates,wrap



subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
output_filename = 'data_3d_h36m'
output_filename_2d = 'data_2d_h36m_gt'
def pre_process():
    output={}
    for subject in subjects:
        output[subject] = {}
        file_list = glob('./data/'+ subject +'/MyPoseFeatures/D3_Positions/*.cdf')
        for f in file_list:
            action = os.path.splitext(os.path.basename(f))[0]

            if subject == 'S11' and action == 'Directions':
                continue  # Discard corrupted video

            # Use consistent naming convention
            canonical_name = action.replace('TakingPhoto', 'Photo') \
                .replace('WalkingDog', 'WalkDog')

            hf = cdflib.CDF(f)
            positions = hf['Pose'].reshape(-1, 32, 3)
            positions /= 1000  # Meters instead of millimeters
            output[subject][canonical_name] = positions.astype('float32')

    print('Saving...')
    np.savez_compressed(output_filename, positions_3d=output)
    print('Done.')
    print('')
    print('Computing ground-truth 2D poses...')
    dataset = Human36mDataset(output_filename + '.npz')
    output_2d_poses = {}
    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_2d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
            output_2d_poses[subject][action] = positions_2d

    print('Saving...')
    metadata = {
        'num_joints': dataset.skeleton().num_joints(),
        'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
    }
    np.savez_compressed(output_filename_2d, positions_2d=output_2d_poses, metadata=metadata)

    print('Done.')


#pre_process()