
import osmnx as ox
import matplotlib.pyplot as plt
import warnings
import math
warnings.filterwarnings("ignore", category=FutureWarning, module='osmnx')
"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

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
       # Convert km to degrees (approx 1° = 111km)
    box_width = box_size_km / 111
    box_height = box_size_km / 111
    
    # Create bounding box
    north = latitude + box_height/2
    south = latitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    bbox = (west, south, east, north)
    
    if poi_tags is None:
        poi_tags = {
            "amenity": True,
            "building": True,
            "historic": True,
            "leisure": True,
            "shop": True,
            "tourism": True,
            "religion": True,
             "memorial": True,
            "aeroway": ["runway", "aerodrome"],
           "natural": True,
           "highway": True,
           "waterway": True,
           
        }
    

    # Download OSM data
    pois = ox.features_from_bbox(bbox, poi_tags)
    
    return pois
from typing import Any, Union
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None
county_relation_ids = {
    "Mombasa":      "R3495554",
    "Kwale":        "R3495548",
    "Kilifi":       "R3495545",
    "Tana River":   "R3495561",
    "Lamu":         "R3495550",
    "Taita-Taveta": "R3495560",
    "Garissa":      "R3495541",
    "Wajir":        "R3495566",
    "Mandera":      "R3495551",
    "Marsabit":     "R3495552",
    "Isiolo":       "R3495542",
    "Meru":         "R3495553",
    "Tharaka-Nithi":"R3495562",
    "Embu":         "R3495540",
    "Kitui":        "R3495547",
    "Machakos":     "r3492707",
    "Makueni":      "r3492708",
    "Nyandarua":    "r3495557",
    "Nyeri":        "r3495558",
    "Kirinyaga":    "r3495546",
    "Muranga":      "r3977580",
    "Kiambu":       "r3495544",
    "Turkana":      "R3495564",
    "West Pokot":   "r11981571",
    "Samburu":      "r3495559",
    "Trans Nzoia":  "r3495563",
    "Uasin Gishu":  "r3495565",
    "Elgeyo-Marakwet":"R11981582",
    "Nandi":        "R3495556",
    "Baringo":      "r3495537",
    "Laikipia":     "r3495549",
    "Nakuru":       "r14649082",
    "Narok":        "r3338145",
    "Kajiado":      "r3486020",
    "Kericho":      "r3486288",
    "Bomet":        "R14649074",
    "Kakamega":     "r3495543",
    "Vihiga":       "r3486322",
    "Bungoma":      "r3495538",
    "Busia":        "r3486321",
    "Siaya":        "r3486291",
    "Kisumu":       "r3486289",
    "Homabay":      "r3486017",
    "Migori":       "r3486018",
    "Kisii":        "r3338140",
    "Nyamira":      "r3486290",
    "Nairobi":      "R3492709"
}

def clean_county_names(counties):
    """
    Clean up county names:
    - Strip whitespace
    - Remove 'City'
    - Remove straight and curly apostrophes
    - Add 'County, Kenya' suffix (except if already present)
    """
    county_list = []
    for c in counties:
        name = str(c).strip()
        # Remove "City" if present
        name = name.replace(" City", "")
        # Remove apostrophes
        name = name.replace("'", "")
        name = name.replace("’", "")  # also handle curly apostrophe
        county_list.append(name)
    return county_list

def get_relation_ids(county_list):
    """
    Given a list of county names and a dictionary mapping
    county names to relation IDs, return a list of relation IDs.
    """
    relation_ids = []
    for county in county_list:
        if county in county_relation_ids:
            relation_ids.append(county_relation_ids[county])
        else:
            print(f"Warning: {county} not found in dictionary")
    return relation_ids





def just_plot_counties(relation_ids, ax=None):
    """
    Plot multiple counties from a list of OSM relation IDs.
    
    Parameters:
        relation_ids (list): List of OSM relation IDs (e.g., ["R3495554", "R3495548"]).
        ax (matplotlib axis, optional): Axis to plot on. Creates new one if None.
        
    Returns:
        gdf (GeoDataFrame): GeoDataFrame with all geometries.
    """
    # Get all counties as a GeoDataFrame
    gdf = ox.geocoder.geocode_to_gdf(relation_ids, by_osmid=True)
    
    # If no axis is passed, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot with different colors
    gdf.plot(ax=ax, alpha=0.5, edgecolor="black", legend=True)
    
    plt.show()
    return gdf


