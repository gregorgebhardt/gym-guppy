from numba import njit, prange
from numba import jit

import numpy as np


def normalize(x: np.ndarray):
    return x / np.sqrt((x ** 2).sum())


@njit(fastmath=True)
def rotation(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array(((c, -s), (s, c)))


@njit(fastmath=True)
def row_norm(matrix: np.ndarray):
    return np.sqrt(np.sum(matrix ** 2, axis=1))


@njit(fastmath=True)
def is_point_left(a, b, c):
    """computes if c is left of the line ab.
    :return: True if c is left of the line ab and False otherwise.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0


@njit(fastmath=True)
def get_local_poses(poses, relative_to):
    local_poses = poses - relative_to
    R = rotation(relative_to[2])
    local_poses[:, :2] = local_poses[:, :2].dot(R)

    return local_poses


@njit(fastmath=True)
def transform_sin_cos(radians):
    rads = np.atleast_2d(radians).reshape(-1, 1)
    return np.concatenate((np.sin(rads), np.cos(rads)), axis=-1)


@njit(fastmath=True)
def polar_coordinates(points):
    # compute polar coordinates
    pts = np.atleast_2d(points)
    dist = row_norm(pts)
    phi = np.arctan2(pts[:, 1], pts[:, 0])

    return dist, phi


def ray_casting_walls(fish_pose, world_bounds, ray_orientations, diagonal_length):
    return 1 - np.nanmin(_ray_casting_walls(fish_pose, np.asarray(world_bounds), ray_orientations), axis=1) / \
           diagonal_length


# TODO: Check if Memory Leak still occurs after issue #4093 of Numba was solved
# @jit(nopython=True, parallel=False)
def _ray_casting_walls(fish_pose, world_bounds, ray_orientations):
    # assert len(fish_pose) in [3, 4], 'expecting 3- or 4-dimensional vector for fish_pose'
    fish_position = fish_pose[:2]
    if len(fish_pose) == 3:
        fish_orientation = fish_pose[2]
    else:
        fish_orientation = np.arctan2(fish_pose[3], fish_pose[2])
    ray_orientations = ray_orientations.reshape((-1, 1))
    ray_orientations = ray_orientations + fish_orientation
    # world_bounds = np.asarray(world_bounds)

    ray_sin = np.sin(ray_orientations)
    ray_cos = np.cos(ray_orientations)

    ray_a = ray_sin
    ray_b = -ray_cos
    ray_c = np.zeros_like(ray_orientations)
    ray_lines = np.concatenate((ray_a, ray_b, ray_c), axis=1)

    # compute homogeneous coordinates of walls
    walls_a = np.array([1., .0, 1., .0]).reshape((-1, 1))
    walls_b = np.array([.0, 1., .0, 1.]).reshape((-1, 1))
    walls_c = -(world_bounds - fish_position).reshape((-1, 1))
    wall_lines = np.concatenate((walls_a, walls_b, walls_c), axis=1)

    # intersections of all rays with the walls
    # TODO: allocating 1D array and reshaping afterwards is a workaround for issue #4093
    intersections = np.empty((len(ray_orientations) * 4)).reshape((-1, 4))
    intersections[:] = np.NaN

    indices = [np.asarray(ray_cos < .0).nonzero()[0],
               np.asarray(ray_sin < .0).nonzero()[0],
               np.asarray(ray_cos > .0).nonzero()[0],
               np.asarray(ray_sin > .0).nonzero()[0]]

    # for i, wall, inz in enumerate(zip(wall_lines, indices)):
    for i in prange(4):
        if len(indices[i]) > 0:
            xs = compute_line_line_intersection(ray_lines[indices[i]], wall_lines[i])
            intersections[indices[i], i] = np.sqrt(np.sum(xs**2, axis=1))

    return intersections


@njit(fastmath=True)
def compute_line_line_intersection(line1: np.ndarray, line2: np.ndarray):
    # check that the lines are given as 2d-arrays and convert if necessary
    line1 = np.atleast_2d(line1)
    line2 = np.atleast_2d(line2)

    # check if lines are given as 3d homogeneous coordinates and that we have either 1-n, n-1, n-n (element wise)
    assert line1.shape[1] == 3
    assert line2.shape[1] == 3
    assert line1.shape[0] == line2.shape[0] or line1.shape[0] == 1 or line2.shape[0] == 1

    # compute the last coordinate of the intersections
    c = (line1[:, 0] * line2[:, 1] - line1[:, 1] * line2[:, 0]).reshape(-1, 1)

    # if this coordinate is 0 then there is no intersection
    inz = c.nonzero()[0]
    r = np.empty(shape=(len(c), 2))
    r[:] = np.NaN

    i1 = np.array([0])
    i2 = np.array([0])
    if line1.shape[0] > 1:
        i1 = inz
    if line2.shape[0] > 1:
        i2 = inz

    # compute coordinates of intersections
    a = line1[i1, 1] * line2[i2, 2] - line2[i2, 1] * line1[i1, 2]
    b = line2[i2, 0] * line1[i1, 2] - line1[i1, 0] * line2[i2, 2]
    r[inz, :] = np.concatenate((a.reshape((-1, 1)), b.reshape((-1, 1))), axis=1) / c[inz]
    return r


@jit(nopython=True, parallel=False)
def compute_dist_bins(relative_to, poses, bin_boundaries, max_dist):
    c, s = np.cos(relative_to[2]), np.sin(relative_to[2])
    rot = np.array(((c, -s), (s, c)))
    local_positions = (poses[:, :2] - relative_to[:2]).dot(rot)

    # compute polar coordinates
    dist, phi = polar_coordinates(local_positions)
    dist = np.minimum(dist, max_dist) / max_dist

    dist_array = np.ones(len(bin_boundaries) - 1)
    for i in range(len(poses)):
        for j in range(len(bin_boundaries) - 1):
            if bin_boundaries[j] <= phi[i] < bin_boundaries[j+1]:
                if dist[i] < dist_array[j]:
                    dist_array[j] = dist[i]
                break
    return 1 - dist_array


def sigmoid(x, shrink):
    return 2. / (1 + np.exp(-x * shrink)) - 1.
