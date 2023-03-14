import numpy as np
import osmnx as ox
import utm
from networkx import MultiDiGraph
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from config import STREET_CONFIG, OSM_IN, PRINT_PROGRESS


def spherical_to_cartesian(lon, lat):
    x, y, zone, ut = utm.from_latlon(lat, lon)
    return np.asarray([x, y])


class Map:

    def lookup(self, node_id: str | int):
        x = self.g.nodes[node_id]['x']
        y = self.g.nodes[node_id]['y']
        return spherical_to_cartesian(x, y)

    def parse_nodes(self, elem: Element) -> np.ndarray:
        nodes = elem.findall('./nd')
        result = []
        for i in range(len(nodes)):
            node_id = int(nodes[i].attrib['ref'])
            node = self.xml.find(f'.//node[@id="{node_id}"]')
            lon = float(node.attrib['lon'])
            lat = float(node.attrib['lat'])
            coord = spherical_to_cartesian(lon, lat)
            result.append(coord)
        return np.asarray(result)

    def detect_crossings(self, node_count: int):
        """
        Extracts all crossings with node count >= node_count
        :return: A dictionary with the crossing center as key and neighbour list as value
        """
        crossings = {}
        # Iterate over edges
        center = -1
        neighbours = []
        for (x, y, k) in self.g.edges(data='highway'):
            if k is None or k in STREET_CONFIG["ignore"]:
                continue
            if center == -1:
                center = x
            if x != center:
                if len(neighbours) >= node_count:
                    crossings[center] = neighbours[:]
                neighbours.clear()
                center = x
            if y not in neighbours:
                neighbours.append(y)
        return crossings

    def __init__(self, path: str):
        self.xml = ElementTree.parse(path)

        self.streets = [x for x in self.xml.findall('way')
                        if x.find('.//tag[@k="highway"]') is not None]
        self.buildings = [x for x in self.xml.findall('way')
                          if x.find('.//tag[@k="building"]') is not None
                          and x.find('.//tag[@v="roof"]') is None]
        self.green = [x for x in self.xml.findall('way')
                      if x.find('.//tag[@v="green"]') is not None
                      or x.find('.//tag[@v="forest"]') is not None]
        self.water = [x for x in self.xml.findall('way')
                      if x.find('.//tag[@v="water"]') is not None
                      or x.find('.//tag[@v="pond"]') is not None]

        # Props
        self.fire_hydrants = [spherical_to_cartesian(float(x.attrib["lon"]), float(x.attrib["lat"]))
                              for x in self.xml.findall('node')
                              if x.find('.//tag[@v="fire_hydrant"]') is not None]
        self.trees = [spherical_to_cartesian(float(x.attrib["lon"]), float(x.attrib["lat"]))
                      for x in self.xml.findall('node')
                      if x.find('.//tag[@v="tree"]') is not None]
        self.waste_baskets = [spherical_to_cartesian(float(x.attrib["lon"]), float(x.attrib["lat"]))
                              for x in self.xml.findall('node')
                              if x.find('.//tag[@v="waste_basket"]') is not None
                              or x.find('.//tag[@v="waste_disposal"]') is not None]
        self.picnic_tables = [spherical_to_cartesian(float(x.attrib["lon"]), float(x.attrib["lat"]))
                              for x in self.xml.findall('node')
                              if x.find('.//tag[@v="picnic_table"]') is not None]

        # Xml bounds differ from actual bounds
        self.minlon = float(self.xml.find('bounds').attrib['minlon'])
        self.minlat = float(self.xml.find('bounds').attrib['minlat'])
        self.maxlon = float(self.xml.find('bounds').attrib['maxlon'])
        self.maxlat = float(self.xml.find('bounds').attrib['maxlat'])

        min_node = spherical_to_cartesian(self.minlon, self.minlat)
        max_node = spherical_to_cartesian(self.maxlon, self.maxlat)
        self.min = np.asarray([min_node[0], 0, min_node[1]])
        self.max = np.asarray([max_node[0], np.linalg.norm(max_node - min_node), max_node[1]])

        diff = max_node - min_node
        self.ratio = diff[0] / diff[1]

        self.g: MultiDiGraph = ox.graph_from_xml(path, simplify=False, retain_all=True)
        self.crossings = self.detect_crossings(3)

        if PRINT_PROGRESS:
            print(f"Parsed map {path} with ratio {self.ratio} and range "
                  f"lon: [{self.minlon}, {self.maxlon}], "
                  f"lat: [{self.minlat}, {self.maxlat}] with:")
            print(f"\t{len(self.streets)} Streets")
            print(f"\t{len(self.buildings)} Buildings")
            print(f"\t{len(self.green)} Green areas")
            print(f"\t{len(self.water)} Water ponds")
            print(f"\t{len(self.fire_hydrants)} Fire hydrants")
            print(f"\t{len(self.trees)} Trees")
            print(f"\t{len(self.waste_baskets)} Waste baskets")
            print(f"\t{len(self.picnic_tables)} Picnic tables")


osm_map = Map(OSM_IN)
