# Copyright 2018 Christoph Heindl.
#
# Licensed under MIT License
# ============================================================
# from https://github.com/cheind/py-thin-plate-spline/blob/f6995795397118b7d0ac01aecd3f39ffbfad9dee/thinplate/numpy.py

import cv2
import numpy as np


class TPS:
    @staticmethod
    def fit(c, lambda_=0., reduced=False):
        n = c.shape[0]

        U = TPS.u(TPS.d(c, c))
        K = U + np.eye(n, dtype=np.float32) * lambda_

        P = np.ones((n, 3), dtype=np.float32)
        P[:, 1:] = c[:, :2]

        v = np.zeros(n+3, dtype=np.float32)
        v[:n] = c[:, -1]

        A = np.zeros((n+3, n+3), dtype=np.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T

        theta = np.linalg.solve(A, v) # p has structure w,a
        return theta[1:] if reduced else theta

    @staticmethod
    def d(a, b):
        return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))

    @staticmethod
    def u(r):
        return r**2 * np.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        x = np.atleast_2d(x)
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = np.concatenate((-np.sum(w, keepdims=True), w))
        b = np.dot(U, w)
        return a[0] + a[1]*x[:, 0] + a[2]*x[:, 1] + b


def uniform_grid(shape):
    """Uniform grid coordinates.

    Params
    ------
    shape : tuple
        HxW defining the number of height and width dimension of the grid

    Returns
    -------
    points: HxWx2 tensor
        Grid coordinates over [0,1] normalized image range.
    """

    h, w = shape[:2]
    c = np.empty((h, w, 2))
    c[..., 0] = np.linspace(0, 1, w, dtype=np.float32)
    c[..., 1] = np.expand_dims(np.linspace(0, 1, h, dtype=np.float32), -1)

    return c


def tps_theta_from_points(c_src, c_dst, reduced=False):
    delta = c_src - c_dst

    cx = np.column_stack((c_dst, delta[:, 0]))
    cy = np.column_stack((c_dst, delta[:, 1]))

    theta_dx = TPS.fit(cx, reduced=reduced)
    theta_dy = TPS.fit(cy, reduced=reduced)

    return np.stack((theta_dx, theta_dy), -1)


def tps_grid(theta, c_dst, dshape):
    ugrid = uniform_grid(dshape)

    reduced = c_dst.shape[0] + 2 == theta.shape[0]

    dx = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 0]).reshape(dshape[:2])
    dy = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 1]).reshape(dshape[:2])
    dgrid = np.stack((dx, dy), -1)

    grid = dgrid + ugrid

    return grid # H'xW'x2 grid[i,j] in range [0..1]


def tps_grid_to_remap(grid, src_shape):
    """Convert a dense grid to OpenCV's remap compatible maps.

    Params
    ------
    grid : HxWx2 array
        Normalized flow field coordinates as computed by compute_densegrid.
    src_shape : tuple
        Height and width of source image in pixels.


    Returns
    -------
    map_x : HxW array
    map_y : HxW array
    """

    mx = (grid[:, :, 0] * src_shape[1]).astype(np.float32)
    my = (grid[:, :, 1] * src_shape[0]).astype(np.float32)

    return mx, my


def get_noise(delta=0.15):
    return delta - np.random.rand() * delta * 2


def warp_image_cv(img, dst_shape=None):
    c_src = np.array([
        [0.0, 0.0],
        [1., 0],
        [1, 1],
        [0, 1],
        [0.25, 0.25],
        [0.75, 0.75],
    ])

    c_dst = np.array([
        [0 + get_noise(), 0 + get_noise()],
        [1. + get_noise(), 0 + get_noise()],
        [1 + get_noise(), 1 + get_noise()],
        [0 + get_noise(), 1 + get_noise()],
        [0.25 + get_noise(), 0.25 + get_noise()],
        [0.75 + get_noise(), 0.75 + get_noise()],
    ])

    dst_shape = dst_shape or img.shape
    theta = tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps_grid(theta, c_dst, dst_shape)
    map_x, map_y = tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC)
