import osmnx as ox
import geopandas as gpd
import numpy as np

def get_osm_datapoints(latitude, longitude, box_size_km=2, poi_tags=None):
    """
    Get OSM data points within a bounding box around given coordinates.
    
    Args:
        latitude (float): Center latitude
        longitude (float): Center longitude  
        box_size_km (float): Size of bounding box in km
        poi_tags (dict): OSM tags to filter points of interest
        
    Returns:
        geopandas.GeoDataFrame: OSM features within the bounding box
    """
    if poi_tags is None:
        poi_tags = {
            "amenity": True,
            "building": True,
            "historic": True,
            "leisure": True,
            "shop": True,
            "tourism": True
        }
    
    # Convert km to degrees (approx 1° = 111km)
    box_width = box_size_km / 111
    box_height = box_size_km / 111
    
    # Create bounding box
    north = latitude + box_height/2
    south = latitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    bbox = (west, south, east, north)
    
    # Download OSM data
    pois = ox.features_from_bbox(bbox, poi_tags)
    
    return pois
