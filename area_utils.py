import numpy as np
import random

import mapbox_earcut as me

from config import FLORA
from file_utils import write_object, insert_object
from math_utils import bbox, halton, p_inside_area, shoelace


def construct_area(way_id: str, nodes: np.ndarray, material: str, y_off: float):
    normals = [[0, 1, 0]]
    indices = me.triangulate_float64(nodes, [len(nodes)])
    faces = []
    for tri in indices.reshape(-1, 3):
        faces.append([[x, 0] for x in tri])
    vertices = [np.asarray([x, y_off, y]) for (x, y) in nodes]
    write_object(f'Area.{way_id}', vertices, [], normals, faces, material)


def place_flora(nodes: np.ndarray, y_offset: float):
    bbox_min, bbox_max = bbox(nodes)
    area = shoelace(nodes)
    tree_count = max(2, int(area / 300))
    i = 0
    for (x, y) in zip(halton(2), halton(3)):
        if i >= tree_count:
            return
        sample = bbox_min + np.array([x, y]) * (bbox_max - bbox_min)
        valid = p_inside_area(sample, nodes)
        if valid:
            rnd = random.randint(0, len(FLORA) - 1)
            flora = FLORA[rnd]
            insert_object(flora["path"], sample, flora["scale"], y_offset=y_offset)
            i += 1
