import cv2
from cv2 import aruco
import numpy as np

rect_size = 420
margin = 50
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# white background
marker = 255 * np.ones((rect_size, rect_size), dtype=np.uint8)
img_marker = aruco.drawMarker(aruco_dict, 0, rect_size - 2 * margin)
# add marker centered
marker[margin:-margin, margin:-margin] = img_marker

def RotMatrix(x, y, z):
    # Rotation matrices around the X,Y,Z axis
    RX = np.array([[1,           0,          0,     0],
                   [0, np.cos(x), -np.sin(x), 0],
                   [0, np.sin(x), np.cos(x),  0],
                   [0,           0,            0,   1]])

    RY = np.array([[np.cos(y), 0, np.sin(y), 0],
                   [0, 1,            0, 0],
                   [-np.sin(y), 0, np.cos(y), 0],
                   [0, 0,            0, 1]])

    RZ = np.array([[np.cos(z), -np.sin(z), 0, 0],
                   [np.sin(z), np.cos(z), 0, 0],
                   [0,            0, 1, 0],
                   [0,            0, 0, 1]])

    # Composed rotation matrix with (RX,RY,RZ)
    R = np.linalg.multi_dot([RX, RY, RZ])
    return R

def render(K, t, r):
    sensor_size = int(K[0][2] * 2)
    f = K[0][0]
    
    # Projection 2D -> 3D matrix
    A1 = np.array([
        [1, 0, -rect_size/2],
        [0, 1, -rect_size/2],
        [0, 0,    0],
        [0, 0,    1]
    ])

    R = RotMatrix(r[0], r[1], r[2])

    # Translation matrix on the Z axis
    T = np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, 1]
    ])

    M = K @ (T @ (R @ A1))

    out = cv2.warpPerspective(marker, M, (sensor_size, sensor_size), cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)
    
    #add noise
    noise = 5 * np.random.randn(sensor_size, sensor_size)
    noisy_image = out.astype(np.float32) + noise
    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image