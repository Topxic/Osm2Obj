import random

from area_utils import construct_area, place_flora
from building_utils import construct_building
from config import PRINT_PROGRESS, BUILDING_ERROR_THRESHOLD, FLORA, PROPS
from file_utils import insert_object, file_init
from map import osm_map
from street_utils import construct_road, fill_with_props


def print_progress(msg: str, t: int):
    if PRINT_PROGRESS:
        print_progress.c += 1
        if print_progress.c >= t:
            print_progress.c = 0
            print(f"{msg}: 100%", end='\n', flush=True)
        else:
            print("{}: {:.2f}% ".format(msg, print_progress.c / t * 100), end='\r', flush=True)	


print_progress.c = 0


def model_buildings():
    failed = 0
    for building in osm_map.buildings:
        print_progress("Modelling buildings", len(osm_map.buildings))
        levels = 0.5
        has_levels = building.find('.//tag[@k="building:levels"]')
        if has_levels is not None:
            levels = int(has_levels.attrib['v']) / 2
        nodes = osm_map.parse_nodes(building)
        if not construct_building(nodes, levels, 1.5):
            failed += 1
    print(f"There were {failed} building that could not be matched with"
          f" an error < {BUILDING_ERROR_THRESHOLD}")


def model_streets():
    for street in osm_map.streets:
        print_progress("Modelling streets", len(osm_map.streets))
        way_id = street.attrib['id']
        highway_type = street.find('.//tag[@k="highway"]').attrib['v']
        # Parse nodes
        node_ids = [int(n.attrib['ref']) for n in street.findall('./nd')]
        # Check for possible sidewalks (none/no, left, right, both)
        has_sidewalk = street.find('.//tag[@k="sidewalk"]')
        sidewalk = "none"
        if has_sidewalk is not None:
            sidewalk = has_sidewalk.attrib['v']
        left = sidewalk in ["left", "both"]
        right = sidewalk in ["right", "both"]
        # Construct street
        construct_road(way_id, node_ids, highway_type, left, right, 0.5)
        fill_with_props(node_ids, highway_type, True, left)
        fill_with_props(node_ids, highway_type, False, right)


def model_areas():
    for green in osm_map.green:
        print_progress("Modelling greens", len(osm_map.green))
        way_id = green.attrib['id']
        nodes = osm_map.parse_nodes(green)
        construct_area(way_id, nodes, "grass", 1)
        place_flora(nodes, 1)

    for water in osm_map.water:
        print_progress("Modelling water ponds", len(osm_map.water))
        way_id = water.attrib['id']
        nodes = osm_map.parse_nodes(water)
        construct_area(way_id, nodes, "water", 1)


def place_props():
    for pos in osm_map.fire_hydrants:
        print_progress("Placing fire hydrants", len(osm_map.fire_hydrants))
        fire_hydrant = PROPS["fire_hydrant"]
        insert_object(fire_hydrant["path"], pos, fire_hydrant["scale"])

    for tree in osm_map.trees:
        print_progress("Placing trees", len(osm_map.trees))
        rnd = random.randint(0, len(FLORA) - 1)
        flora = FLORA[rnd]
        insert_object(flora["path"], tree, flora["scale"])

    for pos in osm_map.waste_baskets:
        print_progress("Placing waste bins", len(osm_map.waste_baskets))
        waste_basket = PROPS["waste_basket"]
        insert_object(waste_basket["path"], pos, waste_basket["scale"])

    for pos in osm_map.picnic_tables:
        print_progress("Placing picnic tables", len(osm_map.picnic_tables))
        picnic_table = PROPS["picnic_table"]
        insert_object(picnic_table["path"], pos, picnic_table["scale"])


file_init()
model_streets()
model_buildings()
model_areas()
place_props()

