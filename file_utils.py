import numpy as np
import os
import shutil
from fnmatch import fnmatch

from config import WRITE_UV
from map import osm_map
from math_utils import is_float

# As we pack everything in one .obj file we need global indices
face_idx = 1
normal_idx = 1
uv_idx = 1
inserted_obj_id = 0


def write_object(name, vertices, uv, normals, faces, material):
    """
    Adds object to a global waveform object file
    :param name: Name of the object
    :param vertices: List of glm.vec3 points
    :param uv: List of texture coordinate 2-tuples
    :param normals: List of glm.vec3 normals
    :param faces: Face indices in format v1/t1/n1 v2/t2/n2 v3/t3/n3
    :param material: Material name to be used for the object
    """
    global face_idx
    global normal_idx
    global uv_idx

    has_single_n = len(normals) == 1 and [normals[0][0], normals[0][1], normals[0][2]] == [0, 1, 0]

    if not WRITE_UV:
        uv = []

    with open("out.obj", 'a') as out:
        out.write(f'o {name}\n')
        for v in vertices:
            v -= osm_map.min
            out.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for (u, v) in uv:
            out.write("vt %.6f %.6f\n" % (u, v))
        for n in normals:
            if not has_single_n:
                out.write("vn %.6f %.6f %.6f\n" % (n[0], n[1], n[2]))
        out.write(f"usemtl {material}\n")
        out.write("s off\n")
        for f in faces:
            for point in f:
                assert point[0] < len(vertices)
                point[0] += face_idx
                if len(uv) > 0:
                    assert point[1] < len(uv)
                    assert point[2] < len(normals)
                    point[1] += uv_idx
                    point[2] += normal_idx
                else:
                    assert point[1] < len(normals)
                    point[1] += normal_idx

            if len(uv) > 0:
                if [n[0], n[1], n[2]] != [0, 1, 0]:
                    out.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n"
                              % (f[0][0], f[0][1], f[0][2],
                                 f[1][0], f[1][1], f[1][2],
                                 f[2][0], f[2][1], f[2][2]))
                else:
                    out.write("f %d/%d/1 %d/%d/1 %d/%d/1\n"
                              % (f[0][0], f[0][1],
                                 f[1][0], f[1][1],
                                 f[2][0], f[2][1]))
            else:
                if [n[0], n[1], n[2]] != [0, 1, 0]:
                    out.write("f %d//%d %d//%d %d//%d\n"
                              % (f[0][0], f[0][1],
                                 f[1][0], f[1][1],
                                 f[2][0], f[2][1]))
                else:
                    out.write("f %d//1 %d//1 %d//1\n"
                              % (f[0][0], f[1][0], f[2][0]))
    face_idx += len(vertices)
    if not has_single_n:
        normal_idx += len(normals)
    uv_idx += len(uv)


def insert_object(path: str,
                  translation: np.ndarray = np.zeros(shape=(2, 1)),
                  scale: np.ndarray = np.asarray([1, 1, 1]),
                  rotation: np.ndarray = np.identity(2),
                  y_offset: float = 0):
    nodes = []
    with open(path, 'r') as src:
        for line in src.readlines():
            if line.startswith("v "):
                vertex = [float(x) for x in line.split() if is_float(x)]
                nodes.append(vertex)

    obj_min = np.mean(np.asarray(nodes), axis=0)
    obj_min[1] = np.min(np.asarray(nodes), axis=0)[1]
    global face_idx
    global normal_idx
    global uv_idx
    global inserted_obj_id

    face_count = 0
    normal_count = 0
    uv_count = 0
    with open(path, 'r') as src:
        with open("out.obj", 'a') as out:
            for line in src.readlines():
                # Skip comments
                if line.startswith("#"):
                    continue
                # Skip material imports
                if line.startswith("mtllib"):
                    continue
                # Skip polygon groups
                if line.startswith("g"):
                    continue
                # Skip material imports
                if line.startswith("o"):
                    line = line.rstrip() + f".{inserted_obj_id}\n"
                    inserted_obj_id += 1
                # Keep track of global indices
                if line.startswith("vt"):
                    if WRITE_UV:
                        uv_count += 1
                    else:
                        continue
                if line.startswith("vn"):
                    normal_count += 1
                    # Rotate normal
                    normal = [float(x) for x in line.split() if is_float(x)]
                    # Apply transformations
                    xz = np.array([normal[0], normal[2]])
                    xz = rotation @ xz
                    normal[0], normal[2] = xz[0], xz[1]
                    line = "vn %.6f %.6f %.6f\n" % (normal[0], normal[1], normal[2])
                # Vertex needs to be scaled and translated
                if line.startswith("v "):
                    face_count += 1
                    v = np.asarray([float(x) for x in line.split() if is_float(x)])
                    v -= obj_min
                    xz = np.asarray([v[0], v[2]])
                    xz = np.asarray([scale[0], scale[2]]) * rotation @ xz + translation
                    v[0], v[2] = xz[0], xz[1]
                    v[1] *= scale[1]
                    v[1] += y_offset
                    v -= osm_map.min

                    line = "v %.6f %.6f %.6f\n" % (v[0], v[1], v[2])
                # We need to reorganize face indices
                tmp = ""
                if line.startswith("f"):
                    for block in line.replace("f", "").split():
                        face = block.split("/")

                        v_idx = int(face[0]) - 1 + face_idx
                        tex_idx = ""
                        n_idx = ""
                        if face[1] != "" and WRITE_UV:
                            tex_idx = int(face[1]) - 1 + uv_idx
                        if len(face) >= 3 and face[2] != "":
                            n_idx = int(face[2]) - 1 + normal_idx

                        tmp += f"{v_idx}/{tex_idx}/{n_idx} "

                    line = "f " + tmp + '\n'
                out.write(line)
    # Update global indices
    face_idx += face_count
    normal_idx += normal_count
    uv_idx += uv_count


def file_init():
    # Assert that materials folder exists
    if not os.path.exists("materials/"):
        os.mkdir("materials/")
    # Clear .obj and .mtl file
    open("out.obj", 'w').close()
    open("materials/out.mtl", 'w').close()
    # Import materials
    with open("out.obj", 'a') as obj:
        with open("materials/out.mtl", 'a') as mtl:
            for path, subdirs, files in os.walk("./resources"):
                for name in files:
                    if fnmatch(name, "*.mtl"):
                        with open(f"{path}/{name}", 'r') as mtl_file:
                            for line in mtl_file.readlines():
                                if line in ["\n", "\r\n"] or line.startswith("#"):
                                    continue
                                if line.startswith("newmtl"):
                                    mtl.write("\n")
                                mtl.write(line)
        obj.write(f"mtllib materials/out.mtl\n")

    # Copy textures into materials folder
    for path, subdirs, files in os.walk("./resources"):
        for name in files:
            if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".bmp"):
                shutil.copy(f"{path}/{name}", f"materials/{name}")

    # Write upwards normal once as it is used very frequently
    with open("out.obj", 'a') as out:
        out.write("vn %.6f %.6f %.6f\n" % (0, 1, 0))

    # Insert ground plane
    min_node, max_node = osm_map.min, osm_map.max
    length = max_node - min_node
    vertices = [[min_node[0] - length[0], 0, min_node[2] - length[2]],
                [max_node[0] + length[0], 0, min_node[2] - length[2]],
                [min_node[0] - length[0], 0, max_node[2] + length[2]],
                [max_node[0] + length[0], 0, max_node[2] + length[2]]]
    normals = [[0, 1, 0]]
    faces = [[[0, 0], [1, 0], [2, 0]], [[3, 0], [2, 0], [1, 0]]]
    write_object("Ground", vertices, [], normals, faces, "ground_plane")
