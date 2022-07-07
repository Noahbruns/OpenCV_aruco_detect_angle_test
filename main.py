import cv2
from cv2 import aruco
from project import render
import numpy as np
import matplotlib.pyplot as plt

f = 300
sensor_size = 80

# Camera Intrisecs matrix 3D -> 2D
K = np.array([
    [f, 0, sensor_size/2, 0],
    [0, f, sensor_size/2, 0],
    [0, 0, 1, 0]
])

K_aruco = K[0:3, 0:3]

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

detected = 0.
wrong = 0
runs = 0.
steps = 500

def val(v):
    return -1 + float(2 * v) / (steps - 1)

error = np.zeros((steps, steps))

for i in range(steps):
    for j in range(steps):
        t = np.array([0, 0, 5000])
        r = np.array([val(i) * np.pi / 2, val(j) * np.pi / 2, 0])

        runs += 1

        img = render(K, t, r)
        corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters = parameters)

        if len(corners) != 1:
            continue

        if ids[0] != 0:
            wrong += 1
            continue

        detected += 1

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 320, K_aruco, np.zeros(5, dtype=np.float32))

        e = np.linalg.norm(rvec[0][0][0:2] - r[0:2])
        error[i, j] = np.max(1 - e / np.pi / 2, 0)

error = np.array(error)
error = error / error.max()
print("Detection rate", detected * 100 / runs)
print("Wrong detection rate", wrong * 100 / runs)

extent = (-90, 90, -90, 90)
ticks = np.linspace(-90, 90, 7)

plt.xticks(ticks)
plt.yticks(ticks)
plt.xlabel('roll')
plt.ylabel('pitch')

plt.imshow(error, extent=extent)
plt.savefig('plot')

