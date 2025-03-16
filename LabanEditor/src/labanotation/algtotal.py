# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import sys, os, math, copy
import json
from math import sin, cos, sqrt, radians

import numpy as np
from decimal import Decimal
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.widgets import Slider, Cursor, Button
from matplotlib.backend_bases import MouseEvent
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from tqdm import tqdm 

try:
    from tkinter import messagebox
except ImportError:
    # Python 2
    import tkMessageBox as messagebox

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tool'))

import settings

from . import labanProcessor as lp

import kp_extractor as kpex

import accessory as ac
import wavfilter as wf
import cluster as cl

class Algorithm:
    algorithm = None
    ax = None

    jointFrames = []

    timeS = None
    all_laban = []

    unfilteredTimeS = None
    unfilteredLaban = []

    labandata = OrderedDict()

    line_ene = None
    vlines = None
    y_data = []
    points = []

    data_fps = 30
    dragging_sb = False
    dragging_point = None

    selectedFrame = 0
    selectedFrameMarker = None

    default_gauss_window_size = 31
    default_gauss_sigma = 5

    gauss_window_size = default_gauss_window_size
    gauss_sigma = default_gauss_sigma

    STEP_THRESHOLD = 0.1  # Minimum displacement to be a step (meters)
    JUMP_THRESHOLD = 0.2  # Minimum vertical displacement for jumps (meters)
    ROTATION_THRESHOLD = 15  # Minimum rotation for a turn (degrees)

    #------------------------------------------------------------------------------
    # Class initialization
    #
    def __init__(self, algorithm):
        self.algorithm = algorithm

    #------------------------------------------------------------------------------
    # reset class variables
    #
    def reset(self):
        self.jointFrames = []

        self.timeS = None
        self.all_laban = []

        self.unfilteredTimeS = None
        self.unfilteredLaban = []

        self.labandata = OrderedDict()

        self.line_ene = None
        self.vlines = None

        self.y_data = []
        self.points = []

        self.data_fps = 30
        self.dragging_sb = False
        self.dragging_point = None

        self.selectedFrame = 0
        self.selectedFrameMarker = None

    #------------------------------------------------------------------------------
    # convert joint data frames to labanotation
    #
    def convertToLabanotation(self, ax, jointD, forceReset,
                              base_rotation_style='every'):
        if (forceReset):
            self.reset()

        self.ax = ax

        self.jointFrames = copy.copy(jointD)

        cnt = len(jointD)

        self.data_fps = 30
        self.duration = jointD[cnt-1]['timeS'][0] if (cnt > 0) else 0.0

        # clear canvas
        if (self.ax != None):
            self.ax.clear()
            self.selectedFrameMarker = None

            self.line_ene = None
            self.vlines = None
            self.y_data = []
            self.points = []

        self.calculateUnfilteredLaban(base_rotation_style=base_rotation_style)

        return self.totalEnergy()

    #------------------------------------------------------------------------------
    # unfiltered labanotation
    

    def calculateUnfilteredLaban(self, base_rotation_style='every'):
        base_rotation = None
        if base_rotation_style == 'first':
            # try:
                # base_rotation=self.jointFrames[0]["R"]
                # base_translation=self.jointFrames[0]["T"]
            # except:
                base_rotation = lp.calculate_base_rotation(self.jointFrames[0])
                # base_translation=None
        cnt = len(self.jointFrames)

        # Store time information
        self.unfilteredTimeS = np.zeros(cnt)

        # Store spherical coordinate data for joints
        elR = np.zeros((cnt, 3))
        elL = np.zeros((cnt, 3))
        wrR = np.zeros((cnt, 3))
        wrL = np.zeros((cnt, 3))
        knR = np.zeros((cnt, 3))
        knL = np.zeros((cnt, 3))
        anR = np.zeros((cnt, 3))
        anL = np.zeros((cnt, 3))
        head = np.zeros((cnt, 3))
        torso = np.zeros((cnt, 3))

        for i in range(cnt):
            if base_rotation_style == 'every':
                
                # try:
                #     base_rotation=self.jointFrames[i]["R"]
                #     base_translation=self.jointFrames[i]["T"]
                # else:
                    base_rotation = lp.calculate_base_rotation(self.jointFrames[i])
                    # base_translation=None
            self.unfilteredTimeS[i] = self.jointFrames[i]['timeS'][0]

            (elR[i], elL[i], wrR[i], wrL[i], 
            knR[i], knL[i], anR[i], anL[i], head[i], torso[i]) = lp.raw2sphere(
                self.jointFrames[i], base_rotation=base_rotation
            )

        # Generate Labanotation for all frames
        self.unfilteredLaban = []

        for i in range(cnt):
            temp = []
            temp.append(lp.coordinate2laban(elR[i][1], elR[i][2]))  # Right Elbow
            temp.append(lp.coordinate2laban(wrR[i][1], wrR[i][2]))  # Right Wrist
            temp.append(lp.coordinate2laban(elL[i][1], elL[i][2]))  # Left Elbow
            temp.append(lp.coordinate2laban(wrL[i][1], wrL[i][2]))  # Left Wrist
            temp.append(lp.coordinate2laban(knR[i][1], knR[i][2]))  # Right Knee
            temp.append(lp.coordinate2laban(knL[i][1], knL[i][2]))  # Left Knee
            temp.append(lp.coordinate2laban(anR[i][1], anR[i][2]))  # Right Ankle
            temp.append(lp.coordinate2laban(anL[i][1], anL[i][2]))  # Left Ankle
            temp.append(lp.coordinate2laban(head[i][1], head[i][2])) # Head Direction
            temp.append(lp.coordinate2laban(torso[i][1], torso[i][2])) # Head Direction
            self.unfilteredLaban.append(temp)
    # Thresholds for detecting movement
   
    # def calculateUnfilteredLaban(self, base_rotation_style='every'):
    #     """
    #     Computes Labanotation symbols for **arms, legs, support, and rotation**.

    #     Args:
    #         base_rotation_style (str): 'first' → uses first frame's base rotation;
    #                                 'every' → recalculates it per frame.
    #     """
    #     base_rotation = None
    #     base_translation = None

    #     if base_rotation_style == 'first':
    #         try:
    #             base_rotation = self.jointFrames[0]["R"][0]
    #             base_translation = self.jointFrames[0]["T"][0]
    #         except KeyError:
    #             base_rotation = lp.calculate_base_rotation(self.jointFrames[0])
    #             base_translation = None

    #     cnt = len(self.jointFrames)


    #     # get hand position
    #     self.unfilteredTimeS = np.zeros(cnt)
       
    #     # ✅ Store spherical coordinate data for joints
    #     elR = np.zeros((cnt, 3))
    #     elL = np.zeros((cnt, 3))
    #     wrR = np.zeros((cnt, 3))
    #     wrL = np.zeros((cnt, 3))
    #     knR = np.zeros((cnt, 3))
    #     knL = np.zeros((cnt, 3))
    #     anR = np.zeros((cnt, 3))
    #     anL = np.zeros((cnt, 3))
    #     head = np.zeros((cnt, 3))
    #     torso = np.zeros((cnt, 3))

    #     # ✅ Store support (steps, jumps, turns)
    #     support = np.full(cnt, 'Stable', dtype=object)  # Default: stable support
    #     base_rotation_partial=self.jointFrames[0]["T"][0]
    #     base_translation_partial=self.jointFrames[0]["R"][0]
        
        
    #     for i in tqdm(range(cnt), desc="Processing Frames"):
    #         if base_rotation_style == 'every':
    #             try:
    #                 base_rotation = self.jointFrames[i]["R"][0]
    #                 base_translation = self.jointFrames[i]["T"][0]
    #             except KeyError:
    #                 base_rotation = lp.calculate_base_rotation(self.jointFrames[i])
    #                 base_translation = None

    #         self.unfilteredTimeS[i] = self.jointFrames[i]['timeS'][0]

    #         # ✅ Convert joints to spherical coordinates
    #         (elR[i], elL[i], wrR[i], wrL[i], 
    #         knR[i], knL[i], anR[i], anL[i], head[i], torso[i]) = lp.raw2sphere(
    #             self.jointFrames[i], base_rotation=base_rotation)#, base_translation=base_translation
    #         # )

    #         # ✅ Detect Steps, Jumps, and Turns
    #         if i > 0:
    #             delta_T = np.linalg.norm(self.jointFrames[i]["T"][0] -  base_translation_partial)  # Translation change
    #             delta_R = np.linalg.norm(self.jointFrames[i]["R"][0] -  base_rotation_partial)  # Rotation change
    #             vertical_move = self.jointFrames[i]["T"][0][1] - base_translation_partial[1]  # Vertical change (Y-axis)

    #             if delta_T > self.STEP_THRESHOLD:
    #                 support[i] = 'Step'
    #                 base_translation_partial=self.jointFrames[i]["T"][0]
    #             if vertical_move > self.JUMP_THRESHOLD:
    #                 support[i] = 'Jump'
    #             if delta_R > np.deg2rad(self.ROTATION_THRESHOLD):  # Convert degrees to radians
    #                 support[i] = 'Turn'
    #                 base_rotation_partial=self.jointFrames[i]["R"][0]

    #     # ✅ Convert to Labanotation
    #     self.unfilteredLaban = []

    #     for i in range(cnt):
    #         temp = []
    #         temp.append(lp.coordinate2laban(elR[i][1], elR[i][2]))  # Right Elbow
    #         temp.append(lp.coordinate2laban(wrR[i][1], wrR[i][2]))  # Right Wrist
    #         temp.append(lp.coordinate2laban(elL[i][1], elL[i][2]))  # Left Elbow
    #         temp.append(lp.coordinate2laban(wrL[i][1], wrL[i][2]))  # Left Wrist
    #         temp.append(lp.coordinate2laban(knR[i][1], knR[i][2]))  # Right Knee
    #         temp.append(lp.coordinate2laban(knL[i][1], knL[i][2]))  # Left Knee
    #         temp.append(lp.coordinate2laban(anR[i][1], anR[i][2]))  # Right Ankle
    #         temp.append(lp.coordinate2laban(anL[i][1], anL[i][2]))  # Left Ankle
    #         temp.append(lp.coordinate2laban(head[i][1], head[i][2]))  # Head Direction
    #         temp.append(lp.coordinate2laban(torso[i][1], torso[i][2]))  # Torso Movement
    #         temp.append(support[i])  # Support info (Step, Jump, Turn)

    #         self.unfilteredLaban.append(temp)

    #------------------------------------------------------------------------------
    # apply total energy algoritm to joint data frames and calculate labanotation
    #
    def totalEnergy(self):
        cnt = len(self.jointFrames)

        # **1️⃣ Allocate space for key body joints**
        handR, handL = np.zeros((cnt, 3)), np.zeros((cnt, 3))
        footR, footL = np.zeros((cnt, 3)), np.zeros((cnt, 3))
        head = np.zeros((cnt, 3))
        torso = np.zeros((cnt, 3))

        for i in range(cnt):
            # **Extract hand positions**
            handR[i] = [self.jointFrames[i]['wristR']['x'][0], 
                        self.jointFrames[i]['wristR']['y'][0], 
                        self.jointFrames[i]['wristR']['z'][0]]
            handL[i] = [self.jointFrames[i]['wristL']['x'][0], 
                        self.jointFrames[i]['wristL']['y'][0], 
                        self.jointFrames[i]['wristL']['z'][0]]

            # **Extract foot positions**
            footR[i] = [self.jointFrames[i]['ankleR']['x'][0], 
                        self.jointFrames[i]['ankleR']['y'][0], 
                        self.jointFrames[i]['ankleR']['z'][0]]
            footL[i] = [self.jointFrames[i]['ankleL']['x'][0], 
                        self.jointFrames[i]['ankleL']['y'][0], 
                        self.jointFrames[i]['ankleL']['z'][0]]

            # **Extract head position**
            head[i] = [self.jointFrames[i]['head']['x'][0], 
                    self.jointFrames[i]['head']['y'][0], 
                    self.jointFrames[i]['head']['z'][0]]

            # **Extract torso (Spine Mid) position**
            torso[i] = [self.jointFrames[i]['spineM']['x'][0], 
                        self.jointFrames[i]['spineM']['y'][0], 
                        self.jointFrames[i]['spineM']['z'][0]]

        # **2️⃣ Apply Gaussian filtering to smooth noise**
        gauss = wf.gaussFilter(self.gauss_window_size, self.gauss_sigma)

        handRF, handLF = wf.calcFilter(handR, gauss), wf.calcFilter(handL, gauss)
        footRF, footLF = wf.calcFilter(footR, gauss), wf.calcFilter(footL, gauss)
        headF = wf.calcFilter(head, gauss)
        torsoF = wf.calcFilter(torso, gauss)

        # **3️⃣ Compute velocities**
        handRv, handLv = ac.vel(self.unfilteredTimeS, handRF), ac.vel(self.unfilteredTimeS, handLF)
        footRv, footLv = ac.vel(self.unfilteredTimeS, footRF), ac.vel(self.unfilteredTimeS, footLF)
        headV = ac.vel(self.unfilteredTimeS, headF)
        torsoV = ac.vel(self.unfilteredTimeS, torsoF)

        # **4️⃣ Compute accelerations**
        handRa, handLa = ac.acc(self.unfilteredTimeS, handRv), ac.acc(self.unfilteredTimeS, handLv)
        footRa, footLa = ac.acc(self.unfilteredTimeS, footRv), ac.acc(self.unfilteredTimeS, footLv)
        headA = ac.acc(self.unfilteredTimeS, headV)
        torsoA = ac.acc(self.unfilteredTimeS, torsoV)

        # **5️⃣ Compute Separate Energy Functions**
        energy_hands = kpex.energy_function_ijcv(
            v_l=handLv, a_l=handLa,
            v_r=handRv, a_r=handRa
        )

        energy_feet = kpex.energy_function_ijcv(
            v_l=footLv, a_l=footLa,
            v_r=footRv, a_r=footRa
        )

        energy_head = kpex.energy_function_ijcv(
            v_l=headV, a_l=headA,
            v_r=headV, a_r=headA  # No left-right separation for head
        )

        energy_torso = kpex.energy_function_ijcv(
            v_l=torsoV, a_l=torsoA,
            v_r=torsoV, a_r=torsoA  # No left-right separation for torso
        )

        # **6️⃣ Detect keyframes where any energy function has a local minimum**
        indices_hands = kpex.gaussian_pecdec(energy_hands)
        indices_feet = kpex.gaussian_pecdec(energy_feet)
        indices_head = kpex.gaussian_pecdec(energy_head)
        indices_torso = kpex.gaussian_pecdec(energy_torso)

        # Combine keyframe indices (only unique values)
        all_indices = sorted(set(indices_hands) | set(indices_feet) | set(indices_head) | set(indices_torso))

        self.y_data = {
            'hands': energy_hands,
            'feet': energy_feet,
            'head': energy_head,
            'torso': energy_torso
        }

        self.points = {idx: (energy_hands[idx], energy_feet[idx], energy_head[idx], energy_torso[idx]) 
                    for idx in all_indices}

        # **7️⃣ Plot Separate Energy Graphs**
        if self.ax:
            self.ax.plot(energy_hands, color='red', label='Arms/Hands Energy')
            self.ax.plot(energy_feet, color='blue', label='Legs/Feet Energy')
            self.ax.plot(energy_head, color='green', label='Head Energy')
            self.ax.plot(energy_torso, color='purple', label='Torso Energy')

            self.ax.set_xlim((0, len(energy_hands) - 1))
            self.ax.set_ylim((min(min(energy_hands), min(energy_feet), min(energy_head), min(energy_torso)) - 0.5, 
                            max(max(energy_hands), max(energy_feet), max(energy_head), max(energy_torso)) + 0.5))

            legend_elements = [
                Line2D([0], [0], color='red', label='Arms/Hands Energy'),
                Line2D([0], [0], color='blue', label='Legs/Feet Energy'),
                Line2D([0], [0], color='green', label='Head Energy'),
                Line2D([0], [0], color='purple', label='Torso Energy'),
                Line2D([0], [0], marker='o', color='w', label='Key Frames', markerfacecolor='black', markersize=10)
            ]

            self.ax.legend(handles=legend_elements, bbox_to_anchor=(0, 1), loc=3, ncol=4)

        self.updateEnergyPlotAndLabanScore(True)
        self.highlightLabanotationRegions(self.unfilteredLaban, 
            (min(min(energy_hands), min(energy_feet), min(energy_head), min(energy_torso)) - 0.5, 
            max(max(energy_hands), max(energy_feet), max(energy_head), max(energy_torso)) + 0.5))

        return self.unfilteredTimeS, self.unfilteredLaban


    #------------------------------------------------------------------------------
    # plot different colors for each labanotation region.
    #
    def highlightLabanotationRegions(self, laban, y):
        if self.ax is None:
            return

        laban_sect = ac.split(laban)
        cnt = len(laban)

        indices_hands = kpex.gaussian_pecdec(self.y_data['hands'])
        indices_feet = kpex.gaussian_pecdec(self.y_data['feet'])
        indices_head = kpex.gaussian_pecdec(self.y_data['head'])
        indices_torso = kpex.gaussian_pecdec(self.y_data['torso'])

        all_indices = sorted(set(indices_hands) | set(indices_feet) | set(indices_head) | set(indices_torso))

        for i in range(len(laban_sect)):
            start = laban_sect[i][0]
            end = laban_sect[i][1]

            c = 'wheat' if start not in all_indices else 'tan'
            a = 0.4

            x_width = end - start + 0.5 if i < len(laban_sect) - 1 else cnt - start + 0.25

            p = patches.Rectangle((start-0.25, y[0]), x_width, y[1]-y[0], alpha=a, color=c)
            self.ax.add_patch(p)


    #------------------------------------------------------------------------------
    #
    def getLabanotationKeyframeData(self, idx, time, dur, laban):
        """
        Generates a structured Labanotation keyframe data dictionary for full-body motion.
        """
        data = OrderedDict()
        data["start time"] = [str(time)]
        data["duration"] = [str(dur)]
        
        # Extract motion direction and level for each body part
        data["head"] = [laban[8][0], laban[8][1]]
        data["torso"] = [laban[9][0], laban[9][1]]
        
        data["right elbow"] = [laban[0][0], laban[0][1]]
        data["right wrist"] = [laban[1][0], laban[1][1]]
        data["left elbow"] = [laban[2][0], laban[2][1]]
        data["left wrist"] = [laban[3][0], laban[3][1]]

        data["right knee"] = [laban[4][0], laban[4][1]]
        data["right foot"] = [laban[5][0], laban[5][1]]
        data["left knee"] = [laban[6][0], laban[6][1]]
        data["left foot"] = [laban[7][0], laban[7][1]]

        # If we have dynamic rotation data, update it; otherwise, keep default.
        data["rotation"] = ['ToLeft', '0']  # Update later if rotation is implemented.

        return data

    #------------------------------------------------------------------------------
    # update labanotation key frames
    def updateLaban(self, indices):
        self.labandata = OrderedDict()
        positions = []

        self.timeS = []
        self.all_laban = []

        # Ensure indices come from all energy functions
        indices_hands = kpex.gaussian_pecdec(self.y_data['hands'])
        indices_feet = kpex.gaussian_pecdec(self.y_data['feet'])
        indices_head = kpex.gaussian_pecdec(self.y_data['head'])
        indices_torso = kpex.gaussian_pecdec(self.y_data['torso'])

        all_indices = sorted(set(indices_hands) | set(indices_feet) | set(indices_head) | set(indices_torso))

        idx = 0
        cnt = len(all_indices)

        if cnt == 0:
            return

        for i in range(cnt):
            j = all_indices[i]

            time = int(self.unfilteredTimeS[j])
            dur = '-1' if j == (cnt - 1) else '1'

            # Store new time and laban
            self.timeS.append(time)
            self.all_laban.append(self.unfilteredLaban[j])

            positions.append("Position" + str(i))
            self.labandata[positions[idx]] = self.getLabanotationKeyframeData(idx, time, dur, self.unfilteredLaban[j])
            idx += 1


    #------------------------------------------------------------------------------
    # update energy markers and lines, and labanotation score
    #
    def updateEnergyPlotAndLabanScore(self, updateLabanScore=False):
        if (self.ax != None):
            if not self.points:
                return

            x, y = zip(*sorted(self.points.items()))

            if not self.line_ene:
                # Add new plots for each body part
                self.line_ene_hands, = self.ax.plot(self.y_data['hands'], '*', color='r', mew=3, markersize=14, markevery=list(x))
                self.line_ene_feet, = self.ax.plot(self.y_data['feet'], '*', color='b', mew=3, markersize=14, markevery=list(x))
                self.line_ene_head, = self.ax.plot(self.y_data['head'], '*', color='g', mew=3, markersize=14, markevery=list(x))
                self.line_ene_torso, = self.ax.plot(self.y_data['torso'], '*', color='purple', mew=3, markersize=14, markevery=list(x))
            else:
                # Update current plots
                self.line_ene_hands.set_data(range(len(self.y_data['hands'])), self.y_data['hands'])
                self.line_ene_feet.set_data(range(len(self.y_data['feet'])), self.y_data['feet'])
                self.line_ene_head.set_data(range(len(self.y_data['head'])), self.y_data['head'])
                self.line_ene_torso.set_data(range(len(self.y_data['torso'])), self.y_data['torso'])

                self.line_ene_hands.set_markevery(list(x))
                self.line_ene_feet.set_markevery(list(x))
                self.line_ene_head.set_markevery(list(x))
                self.line_ene_torso.set_markevery(list(x))

                self.ax.draw_artist(self.line_ene_hands)
                self.ax.draw_artist(self.line_ene_feet)
                self.ax.draw_artist(self.line_ene_head)
                self.ax.draw_artist(self.line_ene_torso)
                self.line_ene=True

            # plot vertical lines to denote labanotation keyframes
            xs = list(x)
            xs = np.array((xs, ) if np.isscalar(xs) else xs, copy=False)
            lims = self.ax.get_ylim()
            x_points = np.repeat(xs[:, None], repeats=3, axis=1).flatten()
            y_points = np.repeat(np.array(lims + (np.nan, ))[None, :], repeats=len(xs), axis=0).flatten()
            if not self.vlines:
                # Add new plot
                self.vlines, = self.ax.plot(x_points, y_points, scaley = False, color='g')
            else:
                # Update current plot
                self.vlines.set_data(x_points, y_points)
                self.ax.draw_artist(self.vlines)

            self.ax.figure.canvas.draw_idle()

        # update laban score
        if (updateLabanScore) and (self.points):
            tmp_indices, _ = zip(*sorted(self.points.items()))
            new_indices = list(tmp_indices)

            self.updateLaban(new_indices)

            settings.application.updateLaban(self.timeS, self.all_laban)

    #------------------------------------------------------------------------------
    #
    def add_point(self, x, y=None):
        if isinstance(x, MouseEvent):
            x, y = int(x.xdata), int(x.ydata)

        y_on_curve = self.y_data[x]
        self.points[x] = y_on_curve

        return x, y_on_curve

    #------------------------------------------------------------------------------
    #
    def remove_point(self, x, _):
        if x in self.points:
            self.points.pop(x)

    #------------------------------------------------------------------------------
    #
    def setSelectedFrameMarker(self):
        if (self.ax is None):
            return

        cnt = len(self.jointFrames)
        idx = self.selectedFrame
        if ((idx is None) or (idx < 0) or (idx >= cnt)):
            return

        time = idx
        padding = 1.0 / 6.0

        if (self.selectedFrameMarker is None):
            yy = self.ax.get_ylim()
            self.selectedFrameMarker = patches.Rectangle((time-padding, yy[0]), 2*padding, (yy[1]-yy[0]), alpha=0.5, color='purple')
            self.ax.add_patch(self.selectedFrameMarker)
        else:
            self.selectedFrameMarker.set_x(time-padding)

    #------------------------------------------------------------------------------
    #
    def findNearestFrameForTime(self, time):
        cnt = len(self.jointFrames)
        if (cnt == 0):
            return None

        timeMS = time

        # find the frame corresponding to the given time
        for idx in range(0, cnt):
            kt = self.unfilteredTimeS[idx]

            if (kt == timeMS):
                return idx
            elif (kt > timeMS):
                break

        # should not get here if idx == 0, but let's check anyway
        if (idx == 0):
            return idx

        # now that we have an index, determine which frame time is closest to
        dist1 = abs(kt - time)
        dist2 = abs(self.unfilteredTimeS[idx-1] - time)

        return idx if (dist1 < dist2) else (idx-1)

    #------------------------------------------------------------------------------
    #
    def saveToJSON(self):
        filePath = settings.checkFileAlreadyExists(settings.application.outputFilePathJson, fileExt=".json", fileTypes=[('json files', '.json'), ('all files', '.*')])
        if filePath is None:
            return

        file_name = os.path.splitext(os.path.basename(filePath))[0]

        labanjson = OrderedDict()
        labanjson[file_name] = self.labandata

        # Add energy functions to JSON
        labanjson["energy_hands"] = list(self.y_data['hands'])
        labanjson["energy_feet"] = list(self.y_data['feet'])
        labanjson["energy_head"] = list(self.y_data['head'])
        labanjson["energy_torso"] = list(self.y_data['torso'])

        try:
            with open(filePath, 'w') as file:
                json.dump(labanjson, file, indent=2)
                settings.application.logMessage(f"Labanotation json script saved to '{settings.beautifyPath(filePath)}'")
        except Exception as e:
            settings.application.logMessage(f"Exception saving Labanotation json script: {str(e)}")

    #------------------------------------------------------------------------------
    #
    def saveToTXT(self):
        filePath = settings.checkFileAlreadyExists(settings.application.outputFilePathTxt, fileExt=".txt", fileTypes=[('text files', '.txt'), ('all files', '.*')])
        if (filePath is None):
            return

        # save text script
        script = settings.application.labanotation.labanToScript(self.timeS, self.all_laban)

        try:
            with open(filePath,'w') as file:
                file.write(script)
                file.close()
                settings.application.logMessage("Labanotation text script was saved to '" + settings.beautifyPath(filePath) + "'")
        except Exception as e:
            strError = e
            settings.application.logMessage("Exception saving Labanotation text script to '" + settings.beautifyPath(filePath) + "': " + str(e))

    #------------------------------------------------------------------------------
    #
    def selectTime(self, time):
        time = time * (self.duration)
        self.selectedFrame = self.findNearestFrameForTime(time)
        self.setSelectedFrameMarker()

    #------------------------------------------------------------------------------
    # find point closest to mouse position
    #
    def find_neighbor_point(self, event):
        distance_threshold = 3.0
        nearest_point = None
        min_distance = math.sqrt(2 * (100 ** 2))
        for x, y in self.points.items():
            distance = math.hypot(event.xdata - x, event.ydata - y) # euclidian norm
            if distance < min_distance:
                min_distance = distance
                nearest_point = (x, y)
        if min_distance < distance_threshold:
            return nearest_point
        return None

    # -----------------------------------------------------------------------------
    # canvas click event
    #
    def onCanvasClick(self, event):
        if (event.xdata is None) or (event.ydata is None):
            return

        # callback method for mouse click event
        # left click
        if event.button == 1 and event.inaxes in [self.ax]:
            if event.dblclick:
                pass
            else:
                self.dragging_sb = True

                # map xdata to [0..1]
                xx = self.ax.get_xlim()
                p = (event.xdata) / (xx[1]-xx[0])

                # call application so that other graphs can be updated as well
                settings.application.selectTime(p)

        # right click
        elif event.button == 3 and event.inaxes in [self.ax]:
            point = self.find_neighbor_point(event)
            if point and event.dblclick:
                self.remove_point(*point)
            elif point:
                self.dragging_point = point
                self.remove_point(*point)
            else:
                self.add_point(event)

            self.updateEnergyPlotAndLabanScore(True)

    # -----------------------------------------------------------------------------
    # canvas click release event
    #
    def onCanvasRelease(self, event):
        if event.button == 1 and event.inaxes in [self.ax] and self.dragging_sb:
            self.dragging_sb = False
        if event.button == 3 and event.inaxes in [self.ax] and self.dragging_point:
            self.add_point(event)
            self.dragging_point = None
            self.updateEnergyPlotAndLabanScore(True)

    # -----------------------------------------------------------------------------
    # canvas move event
    #
    def onCanvasMove(self, event):
        if (not self.dragging_sb or event.xdata is None) and (not self.dragging_point):
            return

        if self.dragging_sb:
            # map xdata to [0..1]
            xx = self.ax.get_xlim()
            p = event.xdata / (xx[1]-xx[0])

            # call application so that other graphs can be updated as well
            settings.application.selectTime(p)
        else:
            self.remove_point(*self.dragging_point)
            self.dragging_point = self.add_point(event)
            self.updateEnergyPlotAndLabanScore()

    #------------------------------------------------------------------------------
    #

