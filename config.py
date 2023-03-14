import numpy as np

# IO settings
OSM_IN = "./input/techfak_residential.osm"
WRITE_UV = False

# Debug settings
DEBUG_BUILDING_MATCHING = False
PRINT_PROGRESS = True

# Building settings
BUILDINGS = [
    {"path": "./resources/buildings/modern_house_1.obj"},
    {"path": "./resources/buildings/moder_house_2.obj"},
    {"path": "./resources/buildings/modern_house_3.obj"},
    {"path": "./resources/buildings/residential_building.obj"},
    {"path": "./resources/buildings/high_residential.obj"}
]
BUILDING_ERROR_THRESHOLD = 18

FLORA = [
    {"path": "./resources/flora/_1_tree_Cone.002.obj", "scale": np.asarray([4, 4, 4])},
    {"path": "./resources/flora/_2_tree_Cylinder.003.obj", "scale": np.asarray([2, 2, 2])},
    {"path": "./resources/flora/_3_tree_Sphere.002.obj", "scale": np.asarray([2, 2, 2])},
    {"path": "./resources/flora/_4_tree_Icosphere.002.obj", "scale": np.asarray([2, 2, 2])},
    {"path": "./resources/flora/_5_tree_Sphere.002.obj", "scale": np.asarray([2, 2, 2])},
    {"path": "./resources/flora/_6_tree_Cube.002.obj", "scale": np.asarray([2, 2, 2])},
    {"path": "./resources/flora/_7_tree_Icosphere.002.obj", "scale": np.asarray([2, 2, 2])},
    {"path": "./resources/flora/_8_tree_Cube.002.obj", "scale": np.asarray([2, 2, 2])},
    {"path": "./resources/flora/_9_tree_Cube.002.obj", "scale": np.asarray([6, 6, 6])},
    {"path": "./resources/flora/_10_tree_Cylinder.002.obj", "scale": np.asarray([2, 2, 2])},
    {"path": "./resources/flora/_11_tree_Cylinder.002.obj", "scale": np.asarray([2, 2, 2])},
    {"path": "./resources/flora/Rock_1__Cube.002.obj", "scale": np.asarray([1, 1, 1])},
]

PROPS = {
    "bench": {
        "path": "./resources/props/bench.obj",
        "scale": np.asarray([0.3, 0.3, 0.3]),
        "xz_orientation": np.asarray([0, 1])
    },
    "bus_stop": {
        "path": "./resources/props/bus_stop.obj",
        "scale": np.asarray([0.05, 0.05, 0.05]),
        "xz_orientation": np.asarray([-1, 0])
    },
    "fire_hydrant": {
        "path": "./resources/props/fire_hydrant.obj",
        "scale": np.asarray([1.6, 2, 1]),
        "xz_orientation": np.asarray([1, 0])
    },
    "picnic_table": {
        "path": "./resources/props/picnic_table.obj",
        "scale": np.asarray([0.002, 0.002, 0.002]),
        "xz_orientation": np.asarray([0, 1])
    },
    "stop_sign": {
        "path": "./resources/props/stop_sign.obj",
        "scale": np.asarray([1, 1, 1]),
        "xz_orientation": np.asarray([1, 0])
    },
    "streetlight": {
        "path": "./resources/props/streetlight.obj",
        "scale": np.asarray([1, 1, 1]),
        "xz_orientation": np.asarray([1, 0])
    },
    "waste_basket": {
        "path": "./resources/props/waste_basket.obj",
        "scale": np.asarray([0.6, 0.6, 0.6]),
        "xz_orientation": np.asarray([0, 1])
    },
}

STREET_CONFIG = {
    "trunk": {
        "width": 6.0,
        "sidewalk_width": 2.0,
        "decorate": []
    },
    "footway": {
        "width": 1.0,
        "sidewalk_width": 1.0,
        "decorate": [{"name": "bench", "frequency": 0.1, "spacing": 16},
                     {"name": "flora", "frequency": 0.7, "spacing": 16},
                     {"name": "fire_hydrant", "frequency": 0.2, "spacing": 16}]
    },
    "residential": {
        "width": 4.0,
        "sidewalk_width": 2.0,
        "decorate": [{"name": "streetlight", "frequency": 0.9, "spacing": 16},
                     {"name": "bus_stop", "frequency": 0.1, "spacing": 16}]
    },
    "default": {
        "width": 2.0,
        "sidewalk_width": 1.5,
        "decorate": []
    },
    "sidewalk_height": 0.35,
    # These streets are not modeled at all
    "ignore": ["tertiary", "pedestrian"]
}
