import math
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from random import random

from config import DEBUG_BUILDING_MATCHING, BUILDING_ERROR_THRESHOLD, BUILDINGS
from file_utils import insert_object
from math_utils import is_float, normalize, graham_scan, align, p_inside_area, shoelace


def simplify_edges(shape: np.ndarray) -> np.ndarray:
    def angle(pre, curr, succ):
        v1, v2 = normalize(pre - curr), normalize(succ - curr)
        return math.acos(np.dot(v1, v2))

    EPS = 0.01
    result = []
    vertices = np.vstack([shape, shape[0]])
    for v in vertices:
        result.append(v)
        while len(result) >= 2 and np.equal(result[-2], result[-1]).all():
            result.pop(-1)
        while len(result) > 2 and math.pi - EPS < angle(result[-3], result[-2], result[-1]) < math.pi + EPS:
            result.pop(-2)
    result.pop(-1)
    return np.array(result)


class Building:

    def get_layout(self, shape: np.ndarray) -> np.ndarray:
        """
        Extracts the ground plane hull for the given object
        :return: 2D hull of ground plane
        """
        # Build convex hull of ground plane
        xz_values = np.delete(shape, 1, axis=1)
        convex_hull = graham_scan(xz_values)
        # Remove redundant edges
        convex_hull = simplify_edges(convex_hull)
        convex_hull -= np.mean(convex_hull, axis=0)
        if DEBUG_BUILDING_MATCHING:
            plt.title(self.file + " simplified")
            plt.plot(np.append(convex_hull.T[0], convex_hull.T[0][0]),
                     np.append(convex_hull.T[1], convex_hull.T[1][0]),
                     "b+--")
            plt.show()
        return convex_hull

    def __init__(self, file: str):
        self.file = file

        shape = []
        # Parse nodes
        with open(self.file, 'r') as src:
            for line in src.readlines():
                if not line.startswith("v "):
                    continue
                vertex = [float(x) for x in line.split() if is_float(x)]
                shape.append(np.array(vertex))
        self.shape = self.get_layout(shape)

        bbox_min, bbox_max = np.min(shape, axis=0), np.max(shape, axis=0)
        denom = max(bbox_max[2] - bbox_min[2], bbox_max[0] - bbox_min[0])
        self.ground_height_ratio = (bbox_max[1] - bbox_min[1]) / denom

        print(f"Detected building {self.file} with {self.shape.shape[0]} vertices "
              f"and a " + "{:.2f}".format(self.ground_height_ratio) + " height ratio (simplified)")


buildings: list[Building] = [Building(b["path"]) for b in BUILDINGS]


def debug_matching(err: float, building_name: str, nodes: np.ndarray, simplified: np.ndarray, transformed: np.ndarray):
    plt.title(building_name
              + "\nBuilding (red), Simplified (blue), Matched building (green)"
              + "\nerror: " + "{:.2f}".format(err))
    plt.gca().set_aspect(1)

    plt.plot(np.append(nodes.T[0], nodes.T[0][0]),
             np.append(nodes.T[1], nodes.T[1][0]),
             "r+--")

    plt.plot(np.append(simplified.T[0], simplified.T[0][0]),
             np.append(simplified.T[1], simplified.T[1][0]),
             "b+--")

    plt.plot(np.append(transformed.T[0], transformed.T[0][0]),
             np.append(transformed.T[1], transformed.T[1][0]),
             "g+--")
    plt.show()


def fitting_metric(truth: np.ndarray, nodes: np.ndarray, height_diff: float) -> float:
    # Connect each vertex from nodes with the nearest
    # vertex of truth and remove this vertex from truth
    metric = 0
    l_truth = list(truth)
    l_nodes = list(nodes)
    for n in l_nodes:
        t = min(l_truth, key=lambda x: np.linalg.norm(n - x))
        metric += np.linalg.norm(n - t)
        # Weight vertices outside by 2
        if not p_inside_area(n, truth):
            metric += np.linalg.norm(n - t)
        l_truth = [x for x in l_truth if not np.equal(t, x).all()]
        # Refill if l_truth is empty
        if len(l_truth) <= 0:
            l_truth = list(truth)
    return metric / nodes.shape[0] * height_diff


def construct_building(nodes: np.ndarray, levels: float, y_off: float) -> bool:
    target = simplify_edges(nodes)
    t = np.mean(target, axis=0)
    # Move to origin
    target -= t

    candidates = []
    for b in buildings:
        target_nodes = target.shape[0]
        building_nodes = b.shape.shape[0]
        error, scale, rotation, building = math.inf, 0, None, None

        for i in range(target_nodes):
            target_edge = target[(i + 1) % target_nodes] - target[i]
            for j in range(building_nodes):
                building_edge = b.shape[(j + 1) % building_nodes] - b.shape[j]

                s = norm(target_edge) / norm(building_edge)
                if s < 0.014:
                    continue
                R = align(normalize(target_edge), normalize(building_edge))

                transformed = np.asarray([s * R @ v + t for v in b.shape])
                e = fitting_metric(nodes, transformed, abs(levels / 2 - b.ground_height_ratio))
                if e < error:
                    error, scale, rotation, building = e, s, R, b

        if DEBUG_BUILDING_MATCHING and b is not None:
            building_name = b.file.split("/")[-1]
            simplified = np.asarray([v + t for v in target])
            transformed = np.asarray([scale * rotation @ v + t for v in b.shape])
            debug_matching(error, building_name, nodes, simplified, transformed)

        if error < BUILDING_ERROR_THRESHOLD:
            candidates.append((error, scale, rotation, building))

    # Choose building randomly and prefer those with low error
    candidates = sorted(candidates, key=lambda x: x[0])
    if len(candidates) > 0:
        for error, scale, rotation, building in candidates:
            if random() < 0.5:
                insert_object(building.file, t, np.asarray([scale, scale, scale]), rotation, y_off)
                break
    return len(candidates) > 0
