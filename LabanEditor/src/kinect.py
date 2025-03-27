# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import os,math,copy
import numpy as np
import torch
import settings
from tqdm import tqdm

import os
import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel

from pathlib import Path
from scipy.spatial.transform import Rotation as Rscipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# a joint point, 'ts' stands for tracking status
jType = np.dtype({'names':['x', 'y', 'z','ts'],'formats':[float,float,float,int]})

# a body
bType = np.dtype({'names':[ 'timeS',                # milliseconds
                            'filled',               # filled gap
                            'spineB', 'spineM',     # meter
                            'neck', 'head',
                            'shoulderL', 'elbowL', 'wristL', 'handL', # tracked=2, inferred=1, nottracked=0
                            'shoulderR', 'elbowR', 'wristR', 'handR',
                            'hipL', 'kneeL', 'ankleL', 'footL',
                            'hipR', 'kneeR', 'ankleR', 'footR',
                            'spineS', 'handTL', 'thumbL', 'handTR', 'thumbR'],
                'formats':[ int,
                            bool,
                            jType, jType,
                            jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType, jType]})

#------------------------------------------------------------------------------
# AMASS Joint Mapping to Kinect Body Format
AMASS_TO_KINECT_MAP = {
    "spineB": 0, "spineM": 3,
    "neck": 12, "head": 15,
    "shoulderL": 16, "elbowL": 18, "wristL": 20, "handL": 22,  # Left arm
    "shoulderR": 17, "elbowR": 19, "wristR": 21, "handR": 37,  # Right arm
    "hipL": 1, "kneeL": 4, "ankleL": 7, "footL": 10,  # Left leg
    "hipR": 2, "kneeR": 5, "ankleR": 8, "footR": 11,  # Right leg
    "spineS": 6,"handTL": 34, "thumbL": 35, "handTR": 49, "thumbR": 50  # Hands
}
# AMASS_TO_KINECT_MAP = {
#     'spineB': 0, 'hipL': 1, 'hipR': 2, 'spineM': 3,
#     'kneeL': 4, 'kneeR': 5, 'spineS': 6, 'ankleL': 7,
#     'ankleR': 8, 'neck': 9, 'footL': 10, 'footR': 11,
#     'head': 12, 'shoulderL': 13, 'shoulderR': 14, 'elbowL': 15,
#     'elbowR': 16, 'wristL': 17, 'wristR': 18, 'handL': 19,
#     'handR': 20
# }


#------------------------------------------------------------------------------
# load kinect data file an return as a bType array
#
def loadKinectDataFile(filePath, fFillGap = False):
    if not os.path.isabs(filePath):
        print('input file ' + os.path.basename(filePath) + ' does not exist.')
        exit()

    f = open(filePath)

    kinectData = []
    idx = 0
    currentTime = 0
    startTime = 0
    lastTime = 0

    line = f.readline()
    while line != '':
        temp = line.split(',')
        if len(temp) < 1+25*3:
            break
        currentTime = int(float(temp[0]))

        tempBody = np.zeros(1, dtype=bType)
        tempBody['filled'] = False
        if (idx == 0):
            tempBody['timeS'] = 1
            startTime = currentTime
        else:
            # from kinect data
            if currentTime > 10e7:
                cnt = ((currentTime - lastTime) / 10000) / 30
                if (cnt < 1):
                    cnt = 1
                tempBody['timeS'] = kinectData[-1][0][0] + cnt*33
            # yamada data
            else:
                tempBody['timeS'] = (currentTime - startTime)

        # get joints
        for j in range(0, 25):
            tempPoint = np.zeros(1, dtype=jType)
            tempPoint['x'] =  float(temp[1+j*4])
            tempPoint['y'] =  float(temp[2+j*4])
            tempPoint['z'] =  float(temp[3+j*4])
            tempPoint['ts'] = int(float(temp[4+j*4]))
            tempBody[0][j+2] = tempPoint

        # fill time gap when needed
        if ((fFillGap == True) and (idx > 0)):
            timeGap = (currentTime - lastTime) / 10000
            if (timeGap > 40):
                cnt = int(timeGap/30)
                if (settings.fVerbose):
                    print('index ' + str(i) + ' is ' + str(timeGap) + 'ms')
                    print('adding ' + str(cnt) + ' frame')

                refPoseA = kinectData[-1][0]
                refPoseB = tempBody[0]

                for j in range(1, cnt):
                    extraBody = np.zeros(1, dtype=bType)
                    # add time first
                    extraBody['timeS'] = 1 + 33*idx
                    extraBody['filled'] = True
                    # then add joints
                    # do a linear interpolation between two poses (refPoseB and refPoseA). If error margins are 
                    # important, replace this interpolation with a joint corrected approach
                    for k in range(2, 27):
                        xGap = (refPoseB[k][0] - refPoseA[k][0])
                        yGap = (refPoseB[k][1] - refPoseA[k][1])
                        zGap = (refPoseB[k][2] - refPoseA[k][2])

                        extraPoint = np.zeros(1,dtype=jType)
                        extraPoint['x'] =  refPoseA[k][0] + (xGap * float(j) / float(cnt))
                        extraPoint['y'] =  refPoseA[k][1] + (yGap * float(j) / float(cnt))
                        extraPoint['z'] =  refPoseA[k][2] + (zGap * float(j) / float(cnt))
                        extraPoint['ts'] = 0
                        extraBody[0][k] = extraPoint

                    kinectData.append(extraBody)
                    idx += 1
            elif timeGap < 30:
                pass
                #print str(i) + ' is ' + 'smaller than 30ms! ' + str(timeGap) + 'ms'

        kinectData.append(tempBody)

        lastTime = currentTime
        idx += 1
        line = f.readline()

    f.close()

    return kinectData

# Kinect-compatible data types
jType = np.dtype({'names': ['x', 'y', 'z', 'ts'], 'formats': [float, float, float, int]})

# a body
bType = np.dtype({'names':[ 'timeS',                # milliseconds
                            'filled',               # filled gap
                            'spineB', 'spineM',     # meter
                            'neck', 'head',
                            'shoulderL', 'elbowL', 'wristL', 'handL', # tracked=2, inferred=1, nottracked=0
                            'shoulderR', 'elbowR', 'wristR', 'handR',
                            'hipL', 'kneeL', 'ankleL', 'footL',
                            'hipR', 'kneeR', 'ankleR', 'footR',
                            'spineS', 'handTL', 'thumbL', 'handTR', 'thumbR', 
                            'T', 'R',
                            ],
                'formats':[ int,
                            bool,
                            jType, jType,
                            jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType,
                            jType, jType, jType, jType, jType,
                            (float, 3), (float, 3)]})
# **AMASS to Kinect Joint Mapping**

from scipy.spatial.transform import Rotation as Rscipy
import numpy as np
from tqdm import tqdm
import torch
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rscipy

def convert_to_kinect_coords(joint_positions, trans, rot):
    """
    Converts AMASS coordinate system (Y-up) to Kinect coordinate system (Z-up).
    
    Args:
        joint_positions (np.array): (N, 52, 3) global joint positions.
        trans (np.array): (N, 3) global translations.
        rot (np.array): (N, 3) global rotations in Euler angles.

    Returns:
        joint_positions, trans, rot transformed to Kinect format.
    """
    num_frames = joint_positions.shape[0]

    # ✅ Swap Y & Z axes for Kinect format (Z-up)
    joint_positions[:, :, [1, 2]] = joint_positions[:, :, [2, 1]]  
    joint_positions[:, :, 1] *= -1  # Keep Z positive

    trans[:, [1, 2]] = trans[:, [2, 1]]
    trans[:, 1] *= -1  # Keep Z positive

    # ✅ Convert Euler angles to match Kinect system
    for i in range(num_frames):
        r = Rscipy.from_euler('xyz', rot[i], degrees=False).as_matrix()
        r_kinect = r[:, [0, 2, 1]] * np.array([1, 1, -1])  # Reorder axes
        rot[i] = Rscipy.from_matrix(r_kinect).as_euler('xyz', degrees=False)  

    return joint_positions, trans, rot

def loadAMASSData(filePath, fps=30, device='cpu'):
    """
    Loads AMASS motion data and converts it to Kinect-compatible format using PyTorch optimizations.
    
    Args:
        filePath (str): Path to the AMASS dataset (.pt).
        fps (int): Frames per second (default is 30).
        device (str): Computation device ('cpu' or 'cuda').
        
    Returns:
        List of Kinect-compatible motion frames.
    """
    # ✅ Load dataset
    ds = torch.load(filePath)
    
    # ✅ Load SMPL-H Model
    support_dir = Path(filePath).resolve().parents[4]  # Adjust for correct model path
    model_path = os.path.join(support_dir, 'body_models/smplh/male/model.npz')
    bm = BodyModel(bm_fname=model_path, num_betas=16).to(device)

    # ✅ Convert pose, translation & betas into tensors (efficient batch processing)
    pose_torch = torch.tensor(ds['pose'], dtype=torch.float32, device=device)  # (N, 156)
    trans_torch = torch.tensor(ds['trans'], dtype=torch.float32, device=device)  # (N, 3)
    betas_torch = torch.tensor(ds['betas'][:16], dtype=torch.float32, device=device).reshape(1, -1)  # Ensure (1, 16)

    num_frames = pose_torch.shape[0]
    
    # ✅ Extract root orientation and body pose
    root_orient = pose_torch[:, :3]  # (N, 3)  -> Euler angles (axis-angle)
    pose_body = pose_torch[:, 3:66]  # (N, 63)

    # ✅ Compute global joint positions in batch
    smpl_output = bm(pose_body=pose_body,
                     root_orient=root_orient,
                     trans=trans_torch)

    joint_positions = smpl_output.Jtr  # (N, 52, 3)

    # ✅ Convert results to numpy (batch processing)
    joint_positions_np = joint_positions.cpu().numpy()  # (N, 52, 3)
    trans_np = trans_torch.cpu().numpy()  # (N, 3)
    root_np = root_orient.cpu().numpy()  # (N, 3)  ✅ Kept in Euler angles

    # ✅ Convert to Kinect Coordinate System BEFORE setting relative translation
    joint_positions_np, trans_np, root_np = convert_to_kinect_coords(joint_positions_np, trans_np, root_np)

    # ✅ Set first frame translation to zero (relative)
    trans_orig = trans_np[0].copy()
    trans_np -= trans_orig

    # ✅ Keep root rotation relative to first frame
    root_orig = root_np[0].copy()
    relative_root_np = root_np - root_orig  # ✅ Euler angles remain relative
    rotation_matrix = Rscipy.from_euler('xyz', root_orig, degrees=False).as_matrix()  # (3,3)


    # ✅ Convert batch data into motion_data format
    motion_data = []
    start_time = 0

    for idx in tqdm(range(min(num_frames, 1000)), desc="Processing Frames"):
        tempBody = np.zeros(1, dtype=bType)
        tempBody['filled'] = False

        if idx == 0:
            tempBody['timeS'] = 0
            start_time = idx * (1000 / fps)
        else:
            tempBody['timeS'] = int(start_time + (idx * (1000 / fps)))

        # ✅ Store joint positions in Kinect format
        for kinect_joint, amass_idx in AMASS_TO_KINECT_MAP.items():
            tempPoint = np.zeros(1, dtype=jType)
            tempPoint['x'], tempPoint['y'], tempPoint['z'] =  rotation_matrix @ joint_positions_np[idx, amass_idx] #+ trans_orig   # Corrected Coordinates
            tempPoint['ts'] = 2  # Fully tracked
            
            tempBody[0][kinect_joint] = tempPoint

        tempBody[0]['T'] = trans_np[idx]  # ✅ Store transformed global translation
        tempBody[0]['R'] = relative_root_np[idx]   # ✅ Store transformed global rotation (Euler)

        motion_data.append(tempBody)

    # ✅ Visualize 5 frames
    # plot_skeleton(motion_data, num_frames=5)

    return motion_data


# def loadAMASSData(filePath, fps=30, device='cpu'):
#     """
#     Loads AMASS motion data and converts it to Kinect-compatible format using PyTorch optimizations.
    
#     Args:
#         filePath (str): Path to the AMASS dataset (.pt).
#         fps (int): Frames per second (default is 30).
#         device (str): Computation device ('cpu' or 'cuda').
        
#     Returns:
#         List of Kinect-compatible motion frames.
#     """
#     # ✅ Load dataset
#     ds = torch.load(filePath)
    
#     # ✅ Load SMPL-H Model
#     support_dir = Path(filePath).resolve().parents[4]  # Adjust for correct model path
#     model_path = os.path.join(support_dir, 'body_models/smplh/male/model.npz')
#     bm = BodyModel(bm_fname=model_path, num_betas=16).to(device)

#     # ✅ Convert pose, translation & betas into tensors (efficient batch processing)
#     pose_torch = torch.tensor(ds['pose'], dtype=torch.float32, device=device)  # (N, 156)
#     trans_torch = torch.tensor(ds['trans'], dtype=torch.float32, device=device)  # (N, 3)
#     betas_torch = betas_torch = torch.tensor(ds['betas'][:16], dtype=torch.float32, device=device).reshape(1, -1)  # Ensure (1, 16)
#   # (1, 16)

#     num_frames = pose_torch.shape[0]
    
#     # ✅ Extract root orientation and body pose
#     root_orient = pose_torch[:, :3]  # (N, 3)
#     pose_body = pose_torch[:, 3:66]  # (N, 63)

#     # ✅ Normalize global translation & rotation to the first frame
#     trans_orig = trans_torch[0]  # (3,)
#     root_orig = root_orient[0]  # (3,)

#     relative_trans = trans_torch - trans_orig  # Normalize translation
#     relative_root = root_orient - root_orig  # Normalize rotation

#     # ✅ Compute global joint positions in batch
#     smpl_output = bm(pose_body=pose_body,
#                      root_orient=root_orient,
#                      trans=trans_torch)
#                     #  betas=betas_torch.expand(num_frames, -1))

#     joint_positions = smpl_output.Jtr  # (N, 52, 3)

    
#     # ✅ Convert results to numpy (batch processing)
#     joint_positions_np = joint_positions.cpu().numpy() 
#     rotation_matrix = Rscipy.from_euler('xyz', root_orig, degrees=False).as_matrix()

#     relative_trans_np = relative_trans.cpu().numpy()
#     relative_root_np = relative_root.cpu().numpy()

#     # ✅ Convert batch data into motion_data format
#     motion_data = []
#     start_time = 0

#     for idx in tqdm(range(min(num_frames,100)), desc="Processing Frames"):
#         tempBody = np.zeros(1, dtype=bType)
#         tempBody['filled'] = False

#         if idx == 0:
#             tempBody['timeS'] = 1
#             start_time = idx * (1000 / fps)
#         else:
#             tempBody['timeS'] = int(start_time + (idx * (1000 / fps)))

#         # ✅ Store joint positions in Kinect format
#         for kinect_joint, amass_idx in AMASS_TO_KINECT_MAP.items():
#             tempPoint = np.zeros(1, dtype=jType)
#             tempPoint['x'], tempPoint['y'], tempPoint['z'] =  joint_positions_np[idx, amass_idx] 
#             tempPoint['ts'] = 2  # Fully tracked
           
#             tempBody[0][kinect_joint] = tempPoint
#             tempBody[0]['T'] = relative_trans_np[idx]  # Store global translation
#             tempBody[0]['R'] = relative_root_np[idx]   # Store global rotation

#         motion_data.append(tempBody)

#     # ✅ Visualize 5 frames
#     # plot_skeleton(motion_data, num_frames=5)

#     return motion_data


def apply_transformation(joint_positions, T, R):
    """
    Applies global translation (T) and rotation (R) to joint positions.
    
    Args:
        joint_positions (dict): Dictionary of joint names and their local 3D positions.
        T (np.array): Global translation (3,)
        R (np.array): Global rotation (3,) (Euler angles)

    Returns:
        transformed_positions (dict): Dictionary of transformed joint positions.
    """

    # Convert Euler angles (R) to rotation matrix
    rotation_matrix = Rscipy.from_euler('xyz', R, degrees=False).as_matrix()

    transformed_positions = {}
    for joint, pos in joint_positions.items():
        pos = np.array(pos).reshape(3, 1)  # Ensure it's column vector
        rotated_pos = rotation_matrix @ pos  # Apply rotation
        transformed_positions[joint] = (rotated_pos.flatten() ).tolist()  # Apply translation

    return transformed_positions


def plot_skeleton(frames, num_frames=4):
    """
    Visualizes multiple skeleton frames as subplots with transformations applied.
    
    Args:
        frames (list): List of motion frames containing Kinect joint positions, T, and R.
        num_frames (int): Number of frames to visualize.
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Define Kinect skeleton connections
    skeleton_connections = [
        ('spineB', 'spineM'), ('spineM', 'spineS'), ('spineS', 'neck'), ('neck', 'head'),
        ('spineS', 'shoulderL'), ('shoulderL', 'elbowL'), ('elbowL', 'wristL'), ('wristL', 'handL'),
        ('spineS', 'shoulderR'), ('shoulderR', 'elbowR'), ('elbowR', 'wristR'), ('wristR', 'handR'),
        ('spineB', 'hipL'), ('hipL', 'kneeL'), ('kneeL', 'ankleL'), ('ankleL', 'footL'),
        ('spineB', 'hipR'), ('hipR', 'kneeR'), ('kneeR', 'ankleR'), ('ankleR', 'footR')
    ]

    # Select evenly spaced frames

    for i in range(num_frames):
        ax = fig.add_subplot(1, num_frames, i + 1, projection='3d')
        frame = frames[10*i]
        
        # Use structured array indexing (original approach)
        joint_positions = {joint: (frame[joint]['x'], frame[joint]['y'], frame[joint]['z']) 
                            for joint in frame.dtype.names[2:-2]}
        T = frame['T']  # Global translation
        R = frame['R']  # Global rotation
    
        # # Extract joint positions & transformations
        # joint_positions = {joint: (frame[joint]['x'], frame[joint]['y'], frame[joint]['z']) for joint in frame.dtype.names[2:]}
        # T = frame['spineB']['T']  # Global translation
        # R = frame['spineB']['R']  # Global rotation

        # Apply global transformation
        joint_positions_transformed = apply_transformation(joint_positions, T, R)

        # Convert coordinate system (swap Y-Z, invert Z)
        for joint in joint_positions_transformed:
            x, y, z = joint_positions_transformed[joint]
            joint_positions_transformed[joint] = [x, y, z]

        # Plot skeleton connections
        for joint_start, joint_end in skeleton_connections:
            if joint_start in joint_positions_transformed and joint_end in joint_positions_transformed:
                start_pos = joint_positions_transformed[joint_start]
                end_pos = joint_positions_transformed[joint_end]
                ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_pos[2], end_pos[2]], 'bo-', linewidth=2)

        # Add joint names
        # for joint, pos in joint_positions_transformed.items():
            # ax.text(pos[0], pos[1], pos[2], joint, fontsize=6, color='red')

        ax.set_xlim([-1, 1])
        ax.set_ylim([0, 2])
        ax.set_zlim([0, 2])  # Upright visualization

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=90, azim=-90)
        ax.set_title(f'Frame {i*5}')

    plt.show()

