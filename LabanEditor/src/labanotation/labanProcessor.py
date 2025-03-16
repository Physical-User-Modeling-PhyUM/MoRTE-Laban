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
def to_sphere(a):
    x = a[0]
    y = a[1]
    z = a[2]
    r = math.sqrt(x**2+y**2+z**2)
    if r == 0:
        return (0.0,0.0,0.0)
    theta = math.degrees(math.acos(z/r))
    # phi is from -180~+180
    if x != 0.0:
        phi = math.degrees(math.atan(y/x))
    else:
        if y > 0:
            phi = 90
        else:
            phi = -90
    if x < 0 and y > 0:
        phi = 180+phi
    elif x < 0 and y < 0:
        phi = -180+phi
    else:
        phi = phi
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
    v2 = spM-shR
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


#------------------------------------------------------------------------------
# Transform origin from kinect-base to shoulder-base, 
# convert position information to angle/direction+level
# Replace LabaProcessor::(FindDriectionXOZ, FindLevelYOZ, FindLevelXOY)
#
import numpy as np

def raw2sphere(joint, base_rotation=None):
    """
    Converts Kinect joint positions into spherical coordinates, respecting Labanotation's
    motion principles and staff structure.
    """
    # Extract **shoulders** (base for arm movements)
    shL = np.array([joint[0]['shoulderL']['x'], joint[0]['shoulderL']['y'], joint[0]['shoulderL']['z']])
    shR = np.array([joint[0]['shoulderR']['x'], joint[0]['shoulderR']['y'], joint[0]['shoulderR']['z']])

    # Extract **torso & pelvis**
    spine = np.array([joint[0]['spineM']['x'], joint[0]['spineM']['y'], joint[0]['spineM']['z']])
    chest = np.array([joint[0]['spineS']['x'], joint[0]['spineS']['y'], joint[0]['spineS']['z']])
    pelvis = np.array([joint[0]['spineB']['x'], joint[0]['spineB']['y'], joint[0]['spineB']['z']])

    # Extract **head position relative to spine**
    head = np.array([joint[0]['head']['x'], joint[0]['head']['y'], joint[0]['head']['z']]) - spine

    # Extract **torso movement relative to pelvis**
    torso = chest - pelvis  # Approximate torso motion (spine bending, leaning)

    # Extract **arm movements**
    elbowR = np.array([joint[0]['elbowR']['x'] - shR[0], joint[0]['elbowR']['y'] - shR[1], joint[0]['elbowR']['z'] - shR[2]])
    elbowL = np.array([joint[0]['elbowL']['x'] - shL[0], joint[0]['elbowL']['y'] - shL[1], joint[0]['elbowL']['z'] - shL[2]])
    wristR = np.array([joint[0]['wristR']['x'] - joint[0]['elbowR']['x'], joint[0]['wristR']['y'] - joint[0]['elbowR']['y'], joint[0]['wristR']['z'] - joint[0]['elbowR']['z']])
    wristL = np.array([joint[0]['wristL']['x'] - joint[0]['elbowL']['x'], joint[0]['wristL']['y'] - joint[0]['elbowL']['y'], joint[0]['wristL']['z'] - joint[0]['elbowL']['z']])

    # Extract **leg movements**
    hipR = np.array([joint[0]['hipR']['x'], joint[0]['hipR']['y'], joint[0]['hipR']['z']])
    hipL = np.array([joint[0]['hipL']['x'], joint[0]['hipL']['y'], joint[0]['hipL']['z']])
    kneeR = np.array([joint[0]['kneeR']['x'] - hipR[0], joint[0]['kneeR']['y'] - hipR[1], joint[0]['kneeR']['z'] - hipR[2]])
    kneeL = np.array([joint[0]['kneeL']['x'] - hipL[0], joint[0]['kneeL']['y'] - hipL[1], joint[0]['kneeL']['z'] - hipL[2]])
    ankleR = np.array([joint[0]['ankleR']['x'] - joint[0]['kneeR']['x'], joint[0]['ankleR']['y'] - joint[0]['kneeR']['y'], joint[0]['ankleR']['z'] - joint[0]['kneeR']['z']])
    ankleL = np.array([joint[0]['ankleL']['x'] - joint[0]['kneeL']['x'], joint[0]['ankleL']['y'] - joint[0]['kneeL']['y'], joint[0]['ankleL']['z'] - joint[0]['kneeL']['z']])
    base_rotation=None  
    if base_rotation is None:
        conv = calculate_base_rotation(joint)
    else:
        conv = base_rotation

    # Convert all extracted positions to spherical coordinates
    elRdeg, elLdeg = to_sphere(np.dot(conv.T, elbowR)), to_sphere(np.dot(conv.T, elbowL))
    wrRdeg, wrLdeg = to_sphere(np.dot(conv.T, wristR)), to_sphere(np.dot(conv.T, wristL))
    knRdeg, knLdeg = to_sphere(np.dot(conv.T, kneeR)), to_sphere(np.dot(conv.T, kneeL))
    anRdeg, anLdeg = to_sphere(np.dot(conv.T, ankleR)), to_sphere(np.dot(conv.T, ankleL))
    head_deg = to_sphere(np.dot(conv.T, head))
    torso_deg = to_sphere(np.dot(conv.T, torso))  # Convert torso movement

    return (elRdeg, elLdeg, wrRdeg, wrLdeg, knRdeg, knLdeg, anRdeg, anLdeg, head_deg, torso_deg)


# def raw2sphere(joint, base_rotation=None, base_translation=None):
#     """
#     Converts a skeleton's 3D positions into spherical coordinates (direction & level)
#     for Labanotation.

#     Args:
#         joint (dict): Kinect joint data.
#         base_rotation (np.array): 3x3 rotation matrix aligning skeleton to global frame.
#         base_translation (np.array): Translation vector for world-space positioning.

#     Returns:
#         Spherical coordinates for:
#         - Right & Left Elbow
#         - Right & Left Wrist
#         - Right & Left Knee
#         - Right & Left Ankle
#         - Head
#         - Torso
#     """
#     # ✅ Extract **Body Joints**
#     pelvis  = np.array([joint[0]['spineB']['x'], joint[0]['spineB']['y'], joint[0]['spineB']['z']])
#     spineM  = np.array([joint[0]['spineM']['x'], joint[0]['spineM']['y'], joint[0]['spineM']['z']])
#     chest   = np.array([joint[0]['spineS']['x'], joint[0]['spineS']['y'], joint[0]['spineS']['z']])
#     head    = np.array([joint[0]['head']['x'], joint[0]['head']['y'], joint[0]['head']['z']])

#     shoulderL = np.array([joint[0]['shoulderL']['x'], joint[0]['shoulderL']['y'], joint[0]['shoulderL']['z']])
#     shoulderR = np.array([joint[0]['shoulderR']['x'], joint[0]['shoulderR']['y'], joint[0]['shoulderR']['z']])
    
#     elbowL = np.array([joint[0]['elbowL']['x'], joint[0]['elbowL']['y'], joint[0]['elbowL']['z']])
#     elbowR = np.array([joint[0]['elbowR']['x'], joint[0]['elbowR']['y'], joint[0]['elbowR']['z']])

#     wristL = np.array([joint[0]['wristL']['x'], joint[0]['wristL']['y'], joint[0]['wristL']['z']])
#     wristR = np.array([joint[0]['wristR']['x'], joint[0]['wristR']['y'], joint[0]['wristR']['z']])

#     # ✅ Extract **Legs**
#     hipL   = np.array([joint[0]['hipL']['x'], joint[0]['hipL']['y'], joint[0]['hipL']['z']])
#     hipR   = np.array([joint[0]['hipR']['x'], joint[0]['hipR']['y'], joint[0]['hipR']['z']])

#     kneeL  = np.array([joint[0]['kneeL']['x'], joint[0]['kneeL']['y'], joint[0]['kneeL']['z']])
#     kneeR  = np.array([joint[0]['kneeR']['x'], joint[0]['kneeR']['y'], joint[0]['kneeR']['z']])

#     ankleL = np.array([joint[0]['ankleL']['x'], joint[0]['ankleL']['y'], joint[0]['ankleL']['z']])
#     ankleR = np.array([joint[0]['ankleR']['x'], joint[0]['ankleR']['y'], joint[0]['ankleR']['z']])

   
#     rotation_matrix = Rscipy.from_euler('xyz', base_rotation, degrees=False).as_matrix()
#     # Convert axis-angle to 3x3 matrix

#     # pelvis = rotation_matrix @ pelvis - base_translation
#     # spineM = rotation_matrix @ spineM
#     # chest  = rotation_matrix @ chest- base_translation
#     # head   = rotation_matrix @ head- base_translation
#     # shoulderL = rotation_matrix @ shoulderL- base_translation
#     # shoulderR = rotation_matrix @ shoulderR- base_translation
#     # elbowL = rotation_matrix @ elbowL- base_translation
#     # elbowR = rotation_matrix @ elbowR- base_translation
#     # wristL = rotation_matrix @ wristL- base_translation
#     # wristR = rotation_matrix @ wristR- base_translation
#     # hipL   = rotation_matrix @ hipL- base_translation
#     # hipR   = rotation_matrix @ hipR- base_translation
#     # kneeL  = rotation_matrix @ kneeL- base_translation
#     # kneeR  = rotation_matrix @ kneeR- base_translation
#     # ankleL = rotation_matrix @ ankleL- base_translation
#     # ankleR = rotation_matrix @ ankleR- base_translation

#     # ✅ Convert to **Spherical Coordinates** for Labanotation
#     elRdeg = to_sphere(elbowR - shoulderR)
#     elLdeg = to_sphere(elbowL - shoulderL)
#     wrRdeg = to_sphere(wristR - (elbowR- shoulderR))
#     wrLdeg = to_sphere(wristL - (elbowL- shoulderL))
    
#     knRdeg = to_sphere(kneeR - hipR)
#     knLdeg = to_sphere(kneeL - hipL)
#     anRdeg = to_sphere(ankleR - (kneeR-hipR))
#     anLdeg = to_sphere(ankleL - (kneeL-hipR))

#     head_deg = to_sphere(head - (chest - pelvis))
#     torso_deg = to_sphere(chest - pelvis)

#     return elRdeg, elLdeg, wrRdeg, wrLdeg, knRdeg, knLdeg, anRdeg, anLdeg, head_deg, torso_deg



#------------------------------------------------------------------------------
# replace LabaProcessor::CoordinateToLabanotation, FindDirectionsHML, FindDirectionsFSB
#
# Direction:
# forward--'f', rightforward--'rf', right--'r',rightbackward--'rb'
# backward--'b', leftbackward--'lb',left--'l',leftforward--'lf'
#
# Height:
# place high--'ph', high--'h', middle/normal--'m', low--'l', place low--'pl'
def coordinate2laban(theta, phi):
    laban = ['Normal', 'Forward']
    
    #find direction, phi, (-180,180]
    #forward
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
    
    # find height, theta, [0,180]
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

    return laban



#------------------------------------------------------------------------------
#
def LabanKeyframeToScript(idx, time, dur, laban_score):
    """
    Generates a Labanotation keyframe script with full-body motion capture.
    """

    strScript = f"#{idx}\nStart Time:{time}\nDuration:{dur}\n"

    # Extract values from `laban_score`, which is a list of (direction, level) pairs
    strScript += f"Head:{laban_score[8][0]}:{laban_score[8][1]}\n"
    strScript += f"Torso:{laban_score[7][0]}:{laban_score[7][1]}\n"
    strScript += f"Right Elbow:{laban_score[0][0]}:{laban_score[0][1]}\n"
    strScript += f"Right Wrist:{laban_score[1][0]}:{laban_score[1][1]}\n"
    strScript += f"Left Elbow:{laban_score[2][0]}:{laban_score[2][1]}\n"
    strScript += f"Left Wrist:{laban_score[3][0]}:{laban_score[3][1]}\n"
    strScript += f"Right Knee:{laban_score[4][0]}:{laban_score[4][1]}\n"
    strScript += f"Right Foot:{laban_score[5][0]}:{laban_score[5][1]}\n"
    strScript += f"Left Knee:{laban_score[6][0]}:{laban_score[6][1]}\n"
    strScript += f"Left Foot:{laban_score[7][0]}:{laban_score[7][1]}\n"

    strScript += 'Rotation:ToLeft:0.0\n'

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

