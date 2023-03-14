import math
import numpy as np
from numpy import empty_like, dot


def align(target_vec: np.ndarray, source_vec: np.ndarray) -> np.ndarray:
    # Calculate rotation matrix
    a_dot_b = np.dot(target_vec, source_vec)
    a_cross_b = target_vec[0] * source_vec[1] - target_vec[1] * source_vec[0]
    R = np.asarray([[a_dot_b, a_cross_b],
                    [-a_cross_b, a_dot_b]])

    assert 0.9999999 < np.linalg.det(R) < 1.0000001
    assert np.linalg.inv(R).all() == R.T.all()
    return R


def angle_clockwise(v1: np.ndarray, v2: np.ndarray) -> float:
    det = v1[0] * v2[1] - v1[1] * v2[0]  # determinant
    return math.atan2(det, np.dot(v1, v2))


def is_left(l1, l2, c):
    return (l2[0] - l1[0]) * (c[1] - l1[1]) - (l2[1] - l1[1]) * (c[0] - l1[0])


def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def p_inside_area(p: np.ndarray, nodes: np.ndarray):
    intersections = 0
    for i in range(len(nodes) - 1):
        p1 = nodes[i]
        p2 = nodes[i + 1]
        if intersect(np.array([-1000000, -1000000]), p, p1, p2):
            intersections += 1
    return intersections % 2 == 1


def find_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    da = p2 - p1
    db = p4 - p3
    dp = p1 - p3
    dap = empty_like(da)
    dap[0] = -da[1]
    dap[1] = da[0]
    denom = dot(dap, db)
    assert denom.any()
    num = dot(dap, dp)
    return (num / denom.astype(float)) * db + p3


def intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """
    Checks if lines p1p2 and p3p4 intersect
    :param p1: Start of first line
    :param p2: End of first line
    :param p3: Start of second line
    :param p4: End of second line
    :return: Lines intersect
    """
    p1x1, p1y1 = p1[0], p1[1]
    p1x2, p1y2 = p2[0], p2[1]
    p2x1, p2y1 = p3[0], p3[1]
    p2x2, p2y2 = p4[0], p4[1]
    a1 = p1y2 - p1y1
    b1 = p1x1 - p1x2
    c1 = (p1x2 * p1y1) - (p1x1 * p1y2)
    d1 = (a1 * p2x1) + (b1 * p2y1) + c1
    d2 = (a1 * p2x2) + (b1 * p2y2) + c1
    if d1 > 0 and d2 > 0:
        return False
    if d1 < 0 and d2 < 0:
        return False
    a2 = p2y2 - p2y1
    b2 = p2x1 - p2x2
    c2 = (p2x2 * p2y1) - (p2x1 * p2y2)
    d1 = (a2 * p1x1) + (b2 * p1y1) + c2
    d2 = (a2 * p1x2) + (b2 * p1y2) + c2
    if d1 > 0 and d2 > 0:
        return False
    if d1 < 0 and d2 < 0:
        return False
    if (a1 * b2) - (a2 * b1) == 0.0:
        return False
    return True


def bbox(nodes: np.ndarray):
    """
    Calculates bounding box for given set of points
    """
    x_min, x_max = math.inf, -math.inf
    y_min, y_max = math.inf, -math.inf
    z_min, z_max = math.inf, -math.inf
    match nodes.shape[1]:
        case 3:
            for (x, y, z) in nodes:
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)
                z_min, z_max = min(z_min, z), max(z_max, z)
            return np.asarray([[x_min, y_min, z_min], [x_max, y_max, z_max]])
        case 2:
            for (x, z) in nodes:
                x_min, x_max = min(x_min, x), max(x_max, x)
                z_min, z_max = min(z_min, z), max(z_max, z)
            return np.asarray([[x_min, z_min], [x_max, z_max]])


def halton(b: int):
    """
    Generator function for Halton sequence
    :param b: Halton Base
    """
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d


def normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normalizes a given numpy array
    :param vec: Numpy array to be normalized
    :return: Normalized array
    """
    assert vec.any()
    return vec / np.linalg.norm(vec)


def point_line_distance(l1: np.ndarray, l2: np.ndarray, p: np.ndarray) -> float:
    num = (l2[0] - l1[0]) * (l1[1] - p[1]) \
          - (l1[0] - p[0]) * (l2[1] - l1[1])
    denom = math.sqrt((l2[0] - l1[0]) ** 2 + (l2[1] - l1[1]) ** 2)
    return num / denom


# https://leetcode.com/problems/erect-the-fence/discuss/103300/Detailed-explanation-of-Graham-scan-in-14-lines-(Python)
def graham_scan(shape: np.ndarray) -> np.ndarray:
    # Remove duplicates
    shape = list(np.unique(shape, axis=0))

    def cross(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    # Computes slope of line between p1 and p2
    def slope(p1, p2):
        return 1.0 * (p1[1] - p2[1]) / (p1[0] - p2[0]) if p1[0] != p2[0] else float('inf')

    # Find the smallest left point and remove it from points
    start = min(shape, key=lambda p: (p[0], p[1]))
    for i in range(len(shape)):
        if shape[i][0] == start[0] and shape[i][1] == start[1]:
            shape.pop(i)
            break

    # Sort points so that traversal is from start in a ccw circle.
    shape.sort(key=lambda p: (slope(p, start), -p[1], p[0]))

    # Add each point to the convex hull.
    # If the last 3 points make a cw turn, the second to last point is wrong.
    ans = [start]
    for p in shape:
        ans.append(p)
        while len(ans) > 2 and cross(ans[-3], ans[-2], ans[-1]) <= 0:
            ans.pop(-2)
    return np.asarray(ans)


def shoelace(poly: np.ndarray):
    x = poly[:, 0]
    y = poly[:, 1]
    S1 = np.sum(x * np.roll(y, -1))
    S2 = np.sum(y * np.roll(x, -1))
    return 0.5 * np.absolute(S1 - S2)
