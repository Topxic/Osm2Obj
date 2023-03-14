import math
import numpy as np
import random

from config import STREET_CONFIG, PROPS, FLORA
from file_utils import write_object, insert_object
from map import osm_map
from math_utils import normalize, find_intersection, angle_clockwise, align


class Crossing:
    def find_streets(self, node_id: int):
        result = []
        for street in osm_map.streets:
            # Find all streets that participate in this crossing
            if street.find(f'.//nd[@ref="{self.center_id}"]') is not None:
                # Check node_id is direct neighbour of center_id to filter correct street
                node_ids = [x.attrib['ref'] for x in street.findall("./nd")]
                indices = [i for i, x in enumerate(node_ids) if x == str(self.center_id)]
                for idx in indices:
                    if idx != 0 and node_ids[idx - 1] == str(node_id):
                        result.append((street, True))
                    elif idx != len(node_ids) - 1 and node_ids[idx + 1] == str(node_id):
                        result.append((street, False))
        # In case we have at least two ways that cover the same path (e.g. stairs)
        # We ignore them by selecting the longest way
        return max(result, key=lambda x: len(x[0].findall("./nd")))

    def find_neighbours(self, neighbours: list[int]):
        """
        Finds the adjacent neighbours for the point origin at
        the crossing at center
        :param neighbours: List of all points connected to center
        :return: Neighbouring ids; Format: left, right
        """
        assert len(neighbours) >= 3
        nc = sum([osm_map.lookup(x) for x in neighbours]) / len(neighbours)
        neighbours = [[x, osm_map.lookup(x)] for x in neighbours if x != self.origin_id]
        angles = [[node_id, angle_clockwise(x - nc, self.origin - nc), x] for (node_id, x) in neighbours]
        left = [x for x in angles if x[1] > 0]
        right = [x for x in angles if x[1] < 0]
        right_id = min(left, key=lambda x: x[1])[0]
        left_id = max(right, key=lambda x: x[1])[0]
        assert right_id is not None and left_id is not None
        return right_id, left_id

    def __init__(self, origin_id: int, center_id: int, origin_config):
        self.origin_id = origin_id
        self.center_id = center_id
        self.center = osm_map.lookup(center_id)
        self.origin = osm_map.lookup(origin_id)

        left_id, right_id = self.find_neighbours(osm_map.crossings[center_id])
        self.left, self.right = osm_map.lookup(left_id), osm_map.lookup(right_id)

        left_street, left_to_center = self.find_streets(left_id)
        right_street, right_to_center = self.find_streets(right_id)

        # Set configs
        self.origin_config = origin_config
        self.left_config = STREET_CONFIG.get(left_street.find('.//tag[@k="highway"]').attrib['v'],
                                             STREET_CONFIG["default"])
        self.right_config = STREET_CONFIG.get(right_street.find('.//tag[@k="highway"]').attrib['v'],
                                              STREET_CONFIG["default"])

        # Check if sidewalks intersect or not
        left_sidewalk = left_street.find('.//tag[@k="sidewalk"]')
        if left_to_center:
            self.left_sidewalk = left_sidewalk is not None and left_sidewalk.attrib['v'] in ["right", "both"]
        else:
            self.left_sidewalk = left_sidewalk is not None and left_sidewalk.attrib['v'] in ["left", "both"]

        right_sidewalk = right_street.find('.//tag[@k="sidewalk"]')
        if right_to_center:
            self.right_sidewalk = right_sidewalk is not None and right_sidewalk.attrib['v'] in ["left", "both"]
        else:
            self.right_sidewalk = right_sidewalk is not None and right_sidewalk.attrib['v'] in ["right", "both"]


def junction_points(crossing: Crossing, sidewalk: bool) -> tuple[np.ndarray, np.ndarray]:
    # Collect half distances
    origin_distance = crossing.origin_config["width"] / 2
    left_distance = crossing.left_config["width"] / 2
    right_distance = crossing.right_config["width"] / 2
    if sidewalk:
        origin_distance += crossing.origin_config["sidewalk_width"]
        if crossing.left_sidewalk:
            left_distance += crossing.left_config["sidewalk_width"]
        if crossing.right_sidewalk:
            right_distance += crossing.right_config["sidewalk_width"]

    main_dir = normalize(crossing.center - crossing.origin)

    # Find intersection with left neighbour
    """ L     R
     P3 X     X
         \   /
       P4 \ /
        P2 X
           |
        P1 X
           O 
    """
    left_dir = np.array([-main_dir[1], main_dir[0]])
    p1 = crossing.origin + origin_distance * left_dir
    p2 = p1 + main_dir

    nl_dir = normalize(crossing.center - crossing.left)
    right_dir = np.array([nl_dir[1], -nl_dir[0]])
    p3 = crossing.left + left_distance * right_dir
    p4 = p3 + nl_dir

    # Lines are parallel
    if np.array_equal(nl_dir, main_dir):
        pl = crossing.center - origin_distance * left_dir
    elif np.array_equal(nl_dir, -main_dir):
        pl = crossing.center + origin_distance * left_dir
    else:
        pl = find_intersection(p1, p2, p3, p4)

    # Find intersection with left neighbour
    """ L     R
        X     X P3
         \   /
          \ / P4
           X P2
           |
           X P1
           O
    """
    right_dir = np.array([main_dir[1], -main_dir[0]])
    p1 = crossing.origin + origin_distance * right_dir
    p2 = p1 + main_dir

    nr_dir = normalize(crossing.center - crossing.right)
    left_dir = np.array([-nr_dir[1], nr_dir[0]])
    p3 = crossing.right + right_distance * left_dir
    p4 = p3 + nr_dir

    # Lines are parallel
    if np.array_equal(nr_dir, main_dir):
        pr = crossing.center - origin_distance * left_dir
    elif np.array_equal(nr_dir, -main_dir):
        pr = crossing.center + origin_distance * left_dir
    else:
        pr = find_intersection(p1, p2, p3, p4)

    return pl, pr


# FIXME This function is a real mess and needs some refactoring
def construct_road(way_id: str, node_ids: list[int], highway_type: str,
                   sidewalk_left: bool, sidewalk_right: bool, y_off: float):
    config = STREET_CONFIG.get(highway_type, STREET_CONFIG["default"])
    street_width = config["width"]
    sidewalk_width = config["sidewalk_width"]
    sidewalk_height = STREET_CONFIG["sidewalk_height"]

    v, f = [], []
    offset = 0

    sln, srn = [[0, 1, 0]], [[0, 1, 0]]
    srv, srf, slv, slf = [], [], [], []
    sl_offset, sr_offset = 0, 0

    path = [node_ids[0], *node_ids, node_ids[len(node_ids) - 1]]
    for (pre_id, curr_id, succ_id) in zip(path, path[1:], path[2:]):

        pre_node = osm_map.lookup(pre_id)
        curr_node = osm_map.lookup(curr_id)
        succ_node = osm_map.lookup(succ_id)

        if curr_id in osm_map.crossings:
            # Not first node of street
            if curr_id != pre_id:
                crossing = Crossing(pre_id, curr_id, config)
                pl, pr = junction_points(crossing, False)

                v.append([pl[0], y_off, pl[1]])
                v.append([pr[0], y_off, pr[1]])
                v.append([curr_node[0], y_off, curr_node[1]])
                f.append([[x + offset, 0] for x in [2, 1, 0]])
                offset += 3

                # Append front face if there is no sidewalk intersection
                if not crossing.left_sidewalk:
                    slf.append([[x + sl_offset, 0] for x in [2, 1, 0]])
                    slf.append([[x + sl_offset, 0] for x in [1, 2, 3]])
                if not crossing.right_sidewalk:
                    srf.append([[x + sr_offset, 0] for x in [2, 1, 0]])
                    srf.append([[x + sr_offset, 0] for x in [1, 2, 3]])

                if sidewalk_left or sidewalk_right:
                    spl, spr = junction_points(crossing, True)
                    if sidewalk_left:
                        # Bottom right
                        slv.append([pl[0], y_off, pl[1]])
                        # Bottom left
                        slv.append([spl[0], y_off, spl[1]])
                        # Top right
                        slv.append([pl[0], sidewalk_height + y_off, pl[1]])
                        # Top left
                        slv.append([spl[0], sidewalk_height + y_off, spl[1]])
                        sl_offset += 4

                    if sidewalk_right:
                        # Bottom right
                        srv.append([spr[0], y_off, spr[1]])
                        # Bottom left
                        srv.append([pr[0], y_off, pr[1]])
                        # Top right
                        srv.append([spr[0], sidewalk_height + y_off, spr[1]])
                        # Top left
                        srv.append([pr[0], sidewalk_height + y_off, pr[1]])
                        sr_offset += 4

                # Append back face if there is no sidewalk intersection
                if not crossing.left_sidewalk and curr_id != succ_id:
                    slf.append([[x + sl_offset, 0] for x in [2, 1, 0]])
                    slf.append([[x + sl_offset, 0] for x in [1, 2, 3]])
                if not crossing.right_sidewalk and curr_id != succ_id:
                    srf.append([[x + sr_offset, 0] for x in [2, 1, 0]])
                    srf.append([[x + sr_offset, 0] for x in [1, 2, 3]])

            # Not last node of street
            if curr_id != succ_id:
                crossing = Crossing(succ_id, curr_id, config)
                pl, pr = junction_points(crossing, False)

                v.append([curr_node[0], y_off, curr_node[1]])
                v.append([pl[0], y_off, pl[1]])
                v.append([pr[0], y_off, pr[1]])
                f.append([[x + offset, 0] for x in [2, 1, 0]])
                f.append([[x + offset, 0] for x in [1, 2, 3]])
                f.append([[x + offset, 0] for x in [1, 3, 4]])
                offset += 3

                # Append front face if there is no sidewalk intersection
                if not crossing.left_sidewalk:
                    slf.append([[x + sl_offset, 0] for x in [2, 1, 0]])
                    slf.append([[x + sl_offset, 0] for x in [1, 2, 3]])
                if not crossing.right_sidewalk:
                    srf.append([[x + sr_offset, 0] for x in [2, 1, 0]])
                    srf.append([[x + sr_offset, 0] for x in [1, 2, 3]])

                if sidewalk_left or sidewalk_right:
                    spl, spr = junction_points(crossing, True)
                    if sidewalk_left:
                        # Bottom left
                        slv.append([pr[0], y_off, pr[1]])
                        # Bottom right
                        slv.append([spr[0], y_off, spr[1]])
                        # Top left
                        slv.append([pr[0], sidewalk_height + y_off, pr[1]])
                        # Top right
                        slv.append([spr[0], sidewalk_height + y_off, spr[1]])
                        if curr_id != succ_id:
                            # Top
                            slf.append([[x + sl_offset, 0] for x in [3, 6, 2]])
                            slf.append([[x + sl_offset, 0] for x in [7, 6, 3]])
                            # Left
                            slf.append([[x + sl_offset, 0] for x in [1, 5, 3]])
                            slf.append([[x + sl_offset, 0] for x in [3, 5, 7]])
                            # Right
                            slf.append([[x + sl_offset, 0] for x in [0, 4, 2]])
                            slf.append([[x + sl_offset, 0] for x in [2, 4, 6]])
                            sl_offset += 4

                    if sidewalk_right:
                        # Bottom left
                        srv.append([spl[0], y_off, spl[1]])
                        # Bottom right
                        srv.append([pl[0], y_off, pl[1]])
                        # Top left
                        srv.append([spl[0], sidewalk_height + y_off, spl[1]])
                        # Top right
                        srv.append([pl[0], sidewalk_height + y_off, pl[1]])
                        if curr_id != succ_id:
                            # Top
                            srf.append([[x + sr_offset, 0] for x in [3, 6, 2]])
                            srf.append([[x + sr_offset, 0] for x in [7, 6, 3]])
                            # Left
                            srf.append([[x + sr_offset, 0] for x in [1, 5, 3]])
                            srf.append([[x + sr_offset, 0] for x in [3, 5, 7]])
                            # Right
                            srf.append([[x + sr_offset, 0] for x in [0, 4, 2]])
                            srf.append([[x + sr_offset, 0] for x in [2, 4, 6]])
                            sr_offset += 4

                # Append back face if there is no sidewalk intersection
                if not crossing.left_sidewalk:
                    slf.append([[x + sl_offset, 0] for x in [2, 1, 0]])
                    slf.append([[x + sl_offset, 0] for x in [1, 2, 3]])
                if not crossing.right_sidewalk:
                    srf.append([[x + sr_offset, 0] for x in [2, 1, 0]])
                    srf.append([[x + sr_offset, 0] for x in [1, 2, 3]])

        # No Crossing -> Append rectangle
        else:
            d = normalize(succ_node - pre_node)
            r = np.array([d[1], -d[0]])

            left = curr_node - street_width / 2 * r
            v.append([left[0], y_off, left[1]])
            right = curr_node + street_width / 2 * r
            v.append([right[0], y_off, right[1]])
            if curr_id != succ_id:
                f.append([[x + offset, 0] for x in [2, 1, 0]])
                f.append([[x + offset, 0] for x in [1, 2, 3]])
                offset += 2

            if sidewalk_left:
                # Bottom right
                slv.append([left[0], y_off, left[1]])
                # Bottom left
                bl = left - sidewalk_width * r
                slv.append([bl[0], y_off, bl[1]])
                # Top right
                slv.append([left[0], sidewalk_height + y_off, left[1]])
                # Top left
                slv.append([bl[0], sidewalk_height + y_off, bl[1]])
                # Append front face of sidewalk
                if curr_id == pre_id:
                    slf.append([[x + sl_offset, 0] for x in [0, 1, 2]])
                    slf.append([[x + sl_offset, 0] for x in [3, 2, 1]])
                if curr_id != succ_id:
                    # Top
                    slf.append([[x + sl_offset, 0] for x in [3, 6, 2]])
                    slf.append([[x + sl_offset, 0] for x in [7, 6, 3]])
                    # Left
                    slf.append([[x + sl_offset, 0] for x in [1, 5, 3]])
                    slf.append([[x + sl_offset, 0] for x in [3, 5, 7]])
                    # Right
                    slf.append([[x + sl_offset, 0] for x in [0, 4, 2]])
                    slf.append([[x + sl_offset, 0] for x in [2, 4, 6]])
                    sl_offset += 4
                # Append back face of sidewalk
                if curr_id == succ_id:
                    slf.append([[x + sl_offset, 0] for x in [0, 1, 2]])
                    slf.append([[x + sl_offset, 0] for x in [3, 2, 1]])

            if sidewalk_right:
                # Bottom right
                br = right + sidewalk_width * r
                srv.append([br[0], y_off, br[1]])
                # Bottom left
                srv.append([right[0], y_off, right[1]])
                # Top right
                srv.append([br[0], sidewalk_height + y_off, br[1]])
                # Top left
                srv.append([right[0], sidewalk_height + y_off, right[1]])
                # Append front face of sidewalk
                if curr_id == pre_id:
                    srf.append([[x + sr_offset, 0] for x in [0, 1, 2]])
                    srf.append([[x + sr_offset, 0] for x in [3, 2, 1]])
                if curr_id != succ_id:
                    # Top
                    srf.append([[x + sr_offset, 0] for x in [3, 6, 2]])
                    srf.append([[x + sr_offset, 0] for x in [7, 6, 3]])
                    # Left
                    srf.append([[x + sr_offset, 0] for x in [1, 5, 3]])
                    srf.append([[x + sr_offset, 0] for x in [3, 5, 7]])
                    # Right
                    srf.append([[x + sr_offset, 0] for x in [0, 4, 2]])
                    srf.append([[x + sr_offset, 0] for x in [2, 4, 6]])
                    sr_offset += 4
                # Append back face of sidewalk
                if curr_id == succ_id:
                    srf.append([[x + sr_offset, 0] for x in [0, 1, 2]])
                    srf.append([[x + sr_offset, 0] for x in [3, 2, 1]])

    write_object(f"{highway_type}.{way_id}", v, [], [[0, 1, 0]], f, "road")
    if len(slv) > 4:
        write_object(f"Sidewalk.Left.{way_id}", slv, [], sln, slf, "sidewalk")
    if len(srv) > 4:
        write_object(f"Sidewalk.Right.{way_id}", srv, [], srn, srf, "sidewalk")


def fill_with_props(node_ids: list[int], highway_type: str, left: bool, sidewalk: bool):
    config = STREET_CONFIG.get(highway_type, STREET_CONFIG["default"])
    if len(config["decorate"]) == 0:
        return

    curr_idx = 0
    p_curr = osm_map.lookup(node_ids[curr_idx])
    p_next = osm_map.lookup(node_ids[curr_idx + 1])
    p = p_curr

    def walk(walk_distance: float):
        nonlocal p, p_curr, p_next, curr_idx
        while walk_distance > 0:
            interval_distance = np.linalg.norm(p_next - p)
            if interval_distance < walk_distance:
                walk_distance -= interval_distance
                curr_idx += 1
                if curr_idx >= len(node_ids) - 2:
                    return
                p_curr = osm_map.lookup(node_ids[curr_idx])
                p_next = osm_map.lookup(node_ids[curr_idx + 1])
                p = p_curr
            else:
                p += walk_distance * normalize(p_next - p_curr)
                walk_distance = 0

    while curr_idx < len(node_ids) - 2:
        prop, spacing = choose_rnd_prop(config["decorate"])

        # Assert stated spacing between props
        walk(spacing)
        if curr_idx >= len(node_ids) - 2:
            return

        # Assert enough space to crossings
        distance_to_crossing = math.inf
        if node_ids[curr_idx] in osm_map.crossings:
            distance_to_crossing = min(np.linalg.norm(p - p_curr), distance_to_crossing)
        if node_ids[curr_idx + 1] in osm_map.crossings:
            distance_to_crossing = min(np.linalg.norm(p - p_next), distance_to_crossing)
        crossing_spacing = 2 * config["width"]
        if distance_to_crossing < crossing_spacing:
            walk(crossing_spacing - distance_to_crossing)

        dir = normalize(p_next - p_curr)
        dir = np.array([dir[1], -dir[0]])
        if left:
            dir = -dir
        insert_prop(prop, config, p, dir, sidewalk)


def choose_rnd_prop(decorates: list):
    # Choose prop based on frequency
    decorates = sorted(decorates, key=lambda x: x["frequency"])
    selected_prop = None
    spacing = None
    rnd = random.random()
    for decorate in decorates:
        if rnd < decorate["frequency"]:
            spacing = decorate["spacing"]
            name = decorate["name"]
            if name == "flora":
                flora_idx = random.randint(0, len(FLORA) - 1)
                selected_prop = FLORA[flora_idx]
            else:
                selected_prop = PROPS[name]
            break
        rnd -= decorate["frequency"]
    assert spacing is not None and selected_prop is not None
    return selected_prop, spacing


def insert_prop(prop, config: dict, p: np.ndarray, dir: np.ndarray, sidewalk: bool):
    # PLace prop
    position = p + 1.2 * config["width"] * dir
    if sidewalk:
        position += config["sidewalk_width"] * dir
    rotation = align(-dir, prop.get("xz_orientation", -dir))
    insert_object(prop["path"], position, prop["scale"], rotation)
