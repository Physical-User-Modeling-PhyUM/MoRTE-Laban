
import pybullet as p
import pybullet_data
import time
import ezc3d
import numpy as np
mocap_data=r'C:\Users\berti\Mi_equipo\Documentos\Roberto\MASTER\TFM\MoRTE-Laban\dataset\00001_raw.c3d')
# Load C3D data
c3d = ezc3d.c3d(mocap_data)
markers = c3d['data']['points']  # (4, n_markers, n_frames)

# Start PyBullet GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Create a simple skeleton using spheres (representing joints)
marker_ids = []
n_markers = markers.shape[1]
for i in range(n_markers):
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
    marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere, basePosition=[0, 0, 0])
    marker_ids.append(marker_id)

# Animate the mocap data
n_frames = markers.shape[2]
for frame in range(n_frames):
    for i, marker_id in enumerate(marker_ids):
        x, y, z = markers[0, i, frame], markers[1, i, frame], markers[2, i, frame]
        if not np.isnan(x):  # Ignore missing markers
            p.resetBasePositionAndOrientation(marker_id, [x, y, z], [0, 0, 0, 1])
    time.sleep(1 / 120)  # Adjust playback speed (e.g., 120 FPS)

# Disconnect after animation
p.disconnect()
