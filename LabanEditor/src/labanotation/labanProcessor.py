# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import math
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
import cv2
#------------------------------------------------------------------------------
# Normalize a 1 dimension vector
#
def norm1d(a):
    if len(a.shape) != 1:
        return -1
    else:
        l = a.shape[0]
        s = 0 #sum
        for i in range(0,l):
            s+=a[i]**2
        s = math.sqrt(s)
        v = np.zeros(l)
        for i in range(0,l):
            v[i] = a[i]/s
        return v
    
#------------------------------------------------------------------------------
# Converting a vector from Cartesian coordianate to spherical
# theta:0~180 (zenith), phi: 0~180 for left, 0~-180 for right (azimuth)
#
def to_sphere(vec):
    """
    Converts a 3D vector into spherical coordinates (r, theta, phi).
    """
    x, y, z = vec
    r = np.linalg.norm(vec)
    
    if r == 0:
        return (0.0, 0.0, 0.0)
    
    theta = np.degrees(np.arccos(z / r))  # Elevation (0° to 180°)
    phi = np.degrees(np.arctan2(y, x))    # Azimuth (-180° to 180°)
    
    return (r, theta, phi)

def calculate_base_rotation(joint):
    shL = np.zeros(3)
    shR = np.zeros(3)
    spM = np.zeros(3)

    shL[0] = joint[0]['shoulderL']['x']
    shL[1] = joint[0]['shoulderL']['y']
    shL[2] = joint[0]['shoulderL']['z']
    shR[0] = joint[0]['shoulderR']['x']
    shR[1] = joint[0]['shoulderR']['y']
    shR[2] = joint[0]['shoulderR']['z']

    spM[0] = joint[0]['spineM']['x']
    spM[1] = joint[0]['spineM']['y']
    spM[2] = joint[0]['spineM']['z']

    # convert kinect space to spherical coordinate
    # 1. normal vector of plane defined by shoulderR, shoulderL and spineM
    sh = np.zeros((3,3))
    v1 = shL-shR
    v2 = [0,-1,0]# spM-shR
    sh[0] = np.cross(v2,v1)#x axis
    sh[1] = v1#y axis
    sh[2] = np.cross(sh[0],sh[1])#z axis
    nv = np.zeros((3,3))
    nv[0] = norm1d(sh[0])
    nv[1] = norm1d(sh[1])
    nv[2] = norm1d(sh[2])
    # 2. generate the rotation matrix for
    # converting point from kinect space to euculid space, then sphereical
    base_rotation = np.transpose(nv)
    return base_rotation

def compute_local_spherical(joints):
    """
    Computes local spherical coordinates (relative to the parent) for key joints.

    Args:
        joints (dict): 3D positions of skeleton joints.

    Returns:
        List of spherical coordinate tuples (r, theta, phi).
    """
    parent_relations = {
        "elbowR": "shoulderR",
        "elbowL": "shoulderL",
        "wristR": "elbowR",
        "wristL": "elbowL",
        "kneeR": "hipR",
        "kneeL": "hipL",
        "ankleR": "kneeR",
        "ankleL": "kneeL",
        "footR": "ankleR",
        "footL": "ankleL",
        "head": "chest",
        "torso": "pelvis"
    }

    spherical_coords = []
    
    for joint, parent in parent_relations.items():
        if joint in joints and parent in joints:
            local_vec = joints[joint] - joints[parent]  # Get vector relative to parent
            spherical_coords.append(to_sphere(local_vec))  # Convert to spherical

        else:
            print("error")
            spherical_coords.append((0, 0, 0))  # Handle missing joints safely
    
    return spherical_coords

#------------------------------------------------------------------------------
# Transform origin from kinect-base to shoulder-base, 
# convert position information to angle/direction+level
# Replace LabaProcessor::(FindDriectionXOZ, FindLevelYOZ, FindLevelXOY)
#import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rscipy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rscipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
def visualize_skeleton_and_spherical(joints, spherical_coords, highlight_joint=None, highlight_parent=None):
    """
    Visualizes the full skeleton alongside spherical coordinates.

    - Left: The full skeleton with highlighted joints.
    - Right: The spherical coordinate plot.
    """
    fig = plt.figure(figsize=(14, 6))

    # ✅ **3D Skeleton Visualization**
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Skeleton with Highlighted Joint")

    # Skeleton Connections
    skeleton_connections = [
        ('pelvis', 'spineB'), ('spineB', 'spineM'), ('spineM', 'chest'), ('chest', 'head'),

        ('chest', 'shoulderL'), ('shoulderL', 'elbowL'), ('elbowL', 'wristL'),
        ('chest', 'shoulderR'), ('shoulderR', 'elbowR'), ('elbowR', 'wristR'),

        ('pelvis', 'hipL'), ('hipL', 'kneeL'), ('kneeL', 'ankleL'), ('ankleL', 'footL'),
        ('pelvis', 'hipR'), ('hipR', 'kneeR'), ('kneeR', 'ankleR'), ('ankleR', 'footR')
    ]

    # **Plot each joint connection**
    for j1, j2 in skeleton_connections:
        color = 'b' if (j1 != highlight_joint and j2 != highlight_joint) else 'g'  # Highlight selected joints
        ax1.plot([joints[j1][0], joints[j2][0]], 
                 [joints[j1][1], joints[j2][1]], 
                 [joints[j1][2], joints[j2][2]], color + 'o-', linewidth=2)

    # **Highlight Specific Joint**
    if highlight_joint and highlight_parent:
        ax1.scatter(*joints[highlight_joint], color='r', s=100, label=f"Joint: {highlight_joint}")
        ax1.scatter(*joints[highlight_parent], color='orange', s=100, label=f"Parent: {highlight_parent}")

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=30, azim=60)
    ax1.legend()

    # ✅ **Spherical Coordinates Plot**
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_title("Local Joint Rotations (Spherical)")

    joint_names = ["elR", "elL", "wrR", "wrL", "knR", "knL", "anR", "anL", "head", "torso"]
    
    for i, (r, theta, phi) in enumerate(spherical_coords):
        ax2.scatter(np.radians(phi), theta, s=r * 50, label=joint_names[i])  # Use phi for angle, theta for height

    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)  # Clockwise
    ax2.set_rlabel_position(0)
    ax2.legend(loc='upper right', fontsize=8)

    plt.show()

import numpy as np

def raw2sphere(joint, base_rotation=None, base_translation=None):
    """
    Converts skeleton's 3D positions into spherical coordinates, adapted to Labanotation's structure.

    Args:
        joint (dict): Kinect joint data.
        base_rotation (np.array): Rotation matrix aligning skeleton to global frame.
        base_translation (np.array): Translation vector.

    Returns:
        dict: Dictionary with spherical coordinates categorized by Labanotation sections.
    """

    # ✅ **Extract Joints and Organize by Labanotation Column Structure**
    joints = {
        "pelvis": np.array([joint[0]['spineB']['x'], joint[0]['spineB']['y'], joint[0]['spineB']['z']]),
        "spineM": np.array([joint[0]['spineM']['x'], joint[0]['spineM']['y'], joint[0]['spineM']['z']]),
        "chest": np.array([joint[0]['spineS']['x'], joint[0]['spineS']['y'], joint[0]['spineS']['z']]),
        "neck": np.array([joint[0]['neck']['x'], joint[0]['neck']['y'], joint[0]['neck']['z']]),
        "head": np.array([joint[0]['head']['x'], joint[0]['head']['y'], joint[0]['head']['z']]),

        # Arms
        "shoulderL": np.array([joint[0]['shoulderL']['x'], joint[0]['shoulderL']['y'], joint[0]['shoulderL']['z']]),
        "shoulderR": np.array([joint[0]['shoulderR']['x'], joint[0]['shoulderR']['y'], joint[0]['shoulderR']['z']]),
        "elbowL": np.array([joint[0]['elbowL']['x'], joint[0]['elbowL']['y'], joint[0]['elbowL']['z']]),
        "elbowR": np.array([joint[0]['elbowR']['x'], joint[0]['elbowR']['y'], joint[0]['elbowR']['z']]),
        "wristL": np.array([joint[0]['wristL']['x'], joint[0]['wristL']['y'], joint[0]['wristL']['z']]),
        "wristR": np.array([joint[0]['wristR']['x'], joint[0]['wristR']['y'], joint[0]['wristR']['z']]),

        # Legs
        "hipL": np.array([joint[0]['hipL']['x'], joint[0]['hipL']['y'], joint[0]['hipL']['z']]),
        "hipR": np.array([joint[0]['hipR']['x'], joint[0]['hipR']['y'], joint[0]['hipR']['z']]),
        "kneeL": np.array([joint[0]['kneeL']['x'], joint[0]['kneeL']['y'], joint[0]['kneeL']['z']]),
        "kneeR": np.array([joint[0]['kneeR']['x'], joint[0]['kneeR']['y'], joint[0]['kneeR']['z']]),
        "ankleL": np.array([joint[0]['ankleL']['x'], joint[0]['ankleL']['y'], joint[0]['ankleL']['z']]),
        "ankleR": np.array([joint[0]['ankleR']['x'], joint[0]['ankleR']['y'], joint[0]['ankleR']['z']]),
        "footL": np.array([joint[0]['footL']['x'], joint[0]['footL']['y'], joint[0]['footL']['z']]),
        "footR": np.array([joint[0]['footR']['x'], joint[0]['footR']['y'], joint[0]['footR']['z']])
    }

    # ✅ **Define Parent-Child Relationships for Relative Movement**
    parent_relations = {
        "elbowR": "shoulderR",
        "elbowL": "shoulderL",
        "wristR": "elbowR",
        "wristL": "elbowL",
        "kneeR": "hipR",
        "kneeL": "hipL",
        "ankleR": "kneeR",
        "ankleL": "kneeL",
        "footR": "ankleR",
        "footL": "ankleL",
        "head": "neck",
        "chest": "spineM",
        "shoulderR":"chest",
        "shoulderL":"chest"
        
        
    }

    # ✅ **Compute Spherical Coordinates Using Local Transformations**
    conv = calculate_base_rotation(joint) # Rscipy.from_euler('xyz', base_rotation, degrees=False).as_matrix() if (base_rotation is not None) else calculate_base_rotation(joint)
    spherical_coords = [ to_sphere(np.dot(conv.T, (joints[joint] - joints[parent_relations[joint]])))
        for joint in parent_relations]

    return spherical_coords



    

#------------------------------------------------------------------------------
# replace LabaProcessor::CoordinateToLabanotation, FindDirectionsHML, FindDirectionsFSB
#
# Direction:
# forward--'f', rightforward--'rf', right--'r',rightbackward--'rb'
# backward--'b', leftbackward--'lb',left--'l',leftforward--'lf'
#
# Height:
# place high--'ph', high--'h', middle/normal--'m', low--'l', place low--'pl'
def coordinate2laban(theta, phi, joint_type, support_data=None, support=True):
    laban = ['Forward','Low']
    
    #find direction, phi, (-180,180]
    #forward
    # if joint_type!="support" or not support:#"arm" or "leg" or "head" of "foot":
    if (phi <= 22.5 and phi >= 0) or (phi < 0 and phi > -22.5):
        laban[0] = 'Forward'
    elif (phi <= 67.5 and phi > 22.5):
        laban[0] = 'Left Forward'
    elif (phi <= 112.5 and phi > 67.5):
        laban[0] = 'Left'
    elif (phi <= 157.5 and phi > 112.5):
        laban[0] = 'Left Backward'
    elif (phi <= -157.5 and phi > -180) or (phi <= 180 and phi > 157.5):
        laban[0] = 'Backward'
    elif (phi <= -112.5 and phi > -157.5):
        laban[0] = 'Right Backward'
    elif (phi <= -67.5 and phi > -112.5):
        laban[0] = 'Right'
    else:
        laban[0] = 'Right Forward'

    if joint_type!="support" and joint_type!="torso" :
        # place high
        if theta < 22.5:
             laban=['Place','High']
        # high
        elif theta < 67.5:
            laban[1] = 'High'
        # normal/mid
        elif theta < 112.5: 
            laban[1] = 'Normal'
        # low
        elif theta < 157.5:
            laban[1] = 'Low'
        # place low
        else:
            laban = ['Place','Low']
        
        if joint_type!="body":
            # place high
            if theta < 15:
                laban=['Place','High']
            # high
            elif theta < 30:
                laban[1] = 'High'
            # normal/mid
            elif theta < 67.5: 
                laban[1] = 'Normal'
            # low
            elif theta <  112.5:
                laban[1] = 'Low'
            # place low
            
        
    if joint_type=="support":
        direction, support_type =support_data
        if support: 
            if theta < 90:
               laban[1] = 'Low o'
            # normal/mid
            elif theta < 120:  
                laban[1] = 'Normal o'
            # low
            else:
                laban[1] = 'High o'
        else:
            
            # laban[0]=direction
            
            # high
            if theta < 67.5:
                laban[1] = 'High'
            # normal/mid
            elif theta < 112.5: 
                laban[1] = 'Normal'
            # low
            else:
                laban[1] = 'Low'
        
    
    return laban


def detect_weight_support(jointFrames, i, base_translation_partial, base_rotation_partial, base_foot, STEP_THRESHOLD=0.01, JUMP_THRESHOLD=0.01, ROTATION_THRESHOLD=15):
    """
    Determines weight support by analyzing global translation and rotation.

    Args:
        jointFrames (dict): Contains global position and rotation per frame.
        i (int): Current frame index.
        base_translation_partial (np.array): Previous frame’s global translation.
        base_rotation_partial (np.array): Previous frame’s global rotation.
        STEP_THRESHOLD (float): Distance threshold to detect stepping.
        JUMP_THRESHOLD (float): Height threshold to detect jumping.
        ROTATION_THRESHOLD (float): Angle threshold to detect turns.

    Returns:
        dict: {'support': 'One Foot'/'Both Feet'/'Airborne'/'Other', 'step': True/False, 'jump': True/False, 'turn': True/False}
    """

    delta_T = np.linalg.norm(jointFrames[i]["T"][0] - base_translation_partial)  # Translation change
    delta_R = np.linalg.norm(jointFrames[i]["R"][0] - base_rotation_partial)  # Rotation change
    vertical_move = jointFrames[i]["T"][0][1] - base_translation_partial[1]  # Vertical movement (Y-axis)
    side_move= jointFrames[i]["T"][0][0] - base_translation_partial[0]
    depth_move= jointFrames[i]["T"][0][2] - base_translation_partial[2]
    support_type = [  "Place","Stand","Both",]
    
    # **Check if weight is on one or both feet**
    footL_y = jointFrames[i]["footL"][0][1]
    footR_y = jointFrames[i]["footR"][0][1]
    ground_contact_L = abs(footL_y-base_foot) < 0.05  # Close to ground
    ground_contact_R = abs(footR_y-base_foot) < 0.05
    base_translation_partial = jointFrames[i]["T"][0]

    if depth_move>STEP_THRESHOLD/2:
            if side_move>STEP_THRESHOLD/2:  
                support_type[0]= "Right Forward"
            elif side_move<-STEP_THRESHOLD/2:
                support_type[0]= "Left Forward"
            else:
                support_type[0]= "Forward"
    elif depth_move<-STEP_THRESHOLD/2:
        if side_move>STEP_THRESHOLD/2:  
            support_type[0]= "Right Backward"
        elif side_move<-STEP_THRESHOLD/2:
            support_type[0]=  "Left Backward"
        else:
            support_type[0]= "Backward"
    elif side_move<-STEP_THRESHOLD/2:
        support_type[0]=  "Left"
    elif side_move>STEP_THRESHOLD/2:
        support_type[0]=  "Right"
    else:
        support_type[0]=  "Place"
    
    if vertical_move > JUMP_THRESHOLD :
        support_type[1]= "Jump"
        if not ground_contact_L and ground_contact_R:
            support_type[2]= "Right"
        elif not ground_contact_R and ground_contact_L:
            support_type[2]= "Left"
        else:
             support_type[2]= "Both"
        
    elif vertical_move < -JUMP_THRESHOLD:
        support_type[1]= "Squat Down"
        
        support_type[2]= "Both"
        
    elif ground_contact_L and not ground_contact_R:
            support_type [2]= "Left"
    elif ground_contact_R and not ground_contact_L:
            support_type [2]= "Right"
  
    angle_rot=min(np.rad2deg(delta_R), 360-np.rad2deg(delta_R) )// ROTATION_THRESHOLD 
    
    rotation=0.0
    if abs(angle_rot)>0:  # Convert degrees to radians
        rotation=angle_rot*ROTATION_THRESHOLD
        base_rotation_partial = jointFrames[i]["R"][0]


    return support_type, rotation, base_translation_partial, base_rotation_partial



#------------------------------------------------------------------------------
#
def LabanKeyframeToScript(idx, time, dur, laban_score):
    """
    Generates a Labanotation keyframe script with full-body motion capture.
    """
    strScript = f"#{idx}\nStart Time:{time}\nDuration:{dur}\n"

    # Staff-based order
    strScript += f"Left Elbow:{laban_score[0][0]}:{laban_score[0][1]}\n"
    strScript += f"Left Wrist:{laban_score[1][0]}:{laban_score[1][1]}\n"
    strScript += f"Left Body:{laban_score[2][0]}:{laban_score[2][1]}\n"
    strScript += f"Left Ankle:{laban_score[3][0]}:{laban_score[3][1]}\n"
    strScript += f"Left Foot:{laban_score[4][0]}:{laban_score[4][1]}\n"
    strScript += f"Left Knee:{laban_score[5][0]}:{laban_score[5][1]}\n"
    strScript += f"Right Knee:{laban_score[6][0]}:{laban_score[6][1]}\n"
    strScript += f"Right Foot:{laban_score[7][0]}:{laban_score[7][1]}\n"
    strScript += f"Right Ankle:{laban_score[8][0]}:{laban_score[8][1]}\n"
    strScript += f"Right Body:{laban_score[9][0]}:{laban_score[9][1]}\n"
    strScript += f"Right Wrist:{laban_score[10][0]}:{laban_score[10][1]}\n"
    strScript += f"Right Elbow:{laban_score[11][0]}:{laban_score[11][1]}\n"
    strScript += f"Head:{laban_score[12][0]}:{laban_score[12][1]}\n"
    strScript += f"Support:{laban_score[13]}\n"
    strScript += f"Rotation:ToLeft:{laban_score[14]}\n"



    return strScript


#------------------------------------------------------------------------------
#
def toScript(timeS, all_laban):
    if (all_laban == None):
        return ""

    strScript = ""
    cnt = len(all_laban)
    for j in range(cnt):
        if j == 0:
            time = 1
        else:
            time = int(timeS[j])

        if j == (cnt - 1):
            dur = '-1'
        else:
            dur = '1'

        strScript += LabanKeyframeToScript(j, time, dur, all_laban[j])

    return strScript

