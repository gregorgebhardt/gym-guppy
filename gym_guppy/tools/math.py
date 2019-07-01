from numba import njit
from numba import jit

import numpy as np


def normalize(x: np.ndarray):
    return x / np.sqrt((x ** 2).sum())


def rotation(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array(((c, -s), (s, c)))


def is_point_left(a, b, c):
    """computes if c is left of the line ab.
    :return: True if c is left of the line ab and False otherwise.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0


def ray_casting_walls(fish_pose, world_bounds, ray_orientations):
    return np.nanmin(_ray_casting_walls(fish_pose, world_bounds, ray_orientations), axis=1)


@njit
def _ray_casting_walls(fish_pose, world_bounds, ray_orientations):
    fish_position = fish_pose[:2]
    fish_orientation = fish_pose[2]
    ray_orientations = ray_orientations.reshape((-1, 1)) - fish_orientation
    world_bounds = np.asarray(world_bounds)

    # ray_orientations -= fish_orientation
    ray_sin = np.sin(ray_orientations)
    ray_cos = np.cos(ray_orientations)

    sin_zero_idx = np.where(ray_sin == 0)[0]
    ray_a = np.ones_like(ray_orientations)
    ray_b = ray_cos / ray_sin
    ray_b[sin_zero_idx] = 1.
    ray_a[sin_zero_idx] = .0
    ray_c = np.zeros_like(ray_orientations)
    ray_lines = np.concatenate((ray_a, ray_b, ray_c), axis=1)

    # intersections of all rays with the walls
    intersections = np.empty((len(ray_orientations), 4)) * np.nan

    walls_a = np.array([1., .0, 1., .0]).reshape((-1, 1))
    walls_b = np.array([.0, 1., .0, 1.]).reshape((-1, 1))
    walls_c = (-world_bounds - fish_position).reshape((-1, 1))
    wall_lines = np.concatenate((walls_a, walls_b, walls_c), axis=1)

    indices = [np.asarray(ray_cos < .0).nonzero()[0],
               np.asarray(ray_sin < .0).nonzero()[0],
               np.asarray(ray_cos > .0).nonzero()[0],
               np.asarray(ray_sin > .0).nonzero()[0]]

    # for i, wall, inz in enumerate(zip(wall_lines, indices)):
    for i in range(4):
        if len(indices[i]) > 0:
            xs = compute_line_line_intersection(ray_lines[indices[i]], wall_lines[i])
            intersections[indices[i], i] = np.sqrt(np.sum(xs**2, axis=1))

    return intersections


@njit
def compute_line_line_intersection(line1: np.ndarray, line2: np.ndarray):
    # check that the lines are given as 2d-arrays and convert if necessary
    line1 = np.atleast_2d(line1)
    line2 = np.atleast_2d(line2)
    assert line1.ndim == 2
    assert line2.ndim == 2

    # check if lines are given as 3d homogeneous coordinates and that we have either 1-n, n-1, n-n (element wise)
    assert line1.shape[1] == 3
    assert line2.shape[1] == 3
    assert line1.shape[0] == line2.shape[0] or line1.shape[0] == 1 or line2.shape[0] == 1

    # compute the last coordinate of the intersections
    c = (line1[:, 0] * line2[:, 1] - line1[:, 1] * line2[:, 0]).reshape(-1, 1)

    # if this coordinate is 0 then there is no intersection
    inz = c.nonzero()[0]
    r = np.empty(shape=(len(c), 2)) * np.nan

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


@jit
def ray_casting_agents(fish_pose, others_pose, ray_orientations):
    c, s = np.cos(fish_pose[2]), np.sin(fish_pose[2])
    R = np.array(((c, -s), (s, c)))
    local_positions = (others_pose[:,:2] - fish_pose[:2]).dot(R)
    # compute polar coordinates
    dist = np.linalg.norm(local_positions, axis=1)
    phi = np.arctan2(local_positions[:, 1], local_positions[:, 0])
    # filter out fishes outside field of vision
    within_fop = np.abs(phi) <= ray_orientations[-1]
    phi = phi[within_fop]
    dist = dist[within_fop]
    # identify the right ray_orientation for each angle
    idx = np.searchsorted(ray_orientations, phi, side="left")
    idx = idx - (np.abs(phi - ray_orientations[idx-1]) < np.abs(phi - ray_orientations[idx]))
    # Sort in descending order of distances, which ensures that we fill each ray_orientation with 
    # the distance of the nearest agent belonging to the ray_orientation
    reverse_argsorted_dist = np.flip(np.argsort(dist))
    sorted_idx = idx[reverse_argsorted_dist]
    out = np.zeros(len(ray_orientations) - 1)
    out[sorted_idx] = dist[reverse_argsorted_dist]
    return out