
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

import pandas as pd
import logging
from typing import Union

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def data(file_path: str = "data.csv") -> Union[pd.DataFrame, None]:
    """
    Load data from a CSV or Stata (.dta) file into a DataFrame with logging and error handling.

    Args:
        file_path (str): Path to the data file. Defaults to 'data.csv'.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if load failed.
    """
    logger.info("Starting data access operation")

    try:
        logger.info(f"Loading data from {file_path}...")

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".dta"):
            df = pd.read_stata(file_path)
        else:
            raise ValueError("Unsupported file type. Please use .csv or .dta")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns")
        return df

    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        print(f"Error: Could not find {file_path}. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        logger.error("File is empty or corrupted")
        print("Error: The file is empty or corrupted")
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
education_mapping = {
    'Never Attended': 0,
    'None': 0,
    'NON-FORMAL EDUCATION (ADULT BASIC EDUCATION)': 0.5, # Basic literacy, not formal school
    'Play Group': 1,
    'PRE-PRIMARY 1': 2,
    'PRE-PRIMARY 2': 3,
    'STANDARD/GRADE 1': 4,
    'STANDARD/GRADE 2': 5,
    'STANDARD/GRADE 3': 6,
    'STANDARD/GRADE 4': 7,
    'STANDARD/GRADE 5': 8,
    'STANDARD/GRADE 6': 9,
    'STANDARD 7': 9.5,        # Last year of old primary system
    'STANDARD 8': 10,        # Last year of old primary system (same level as Std 7 in new system)
    'JUNIOR SCHOOL-GRADE 7': 9.5, # Last year of JSS (equivalent to Std 8)
    'JUNIOR SCHOOL-GRADE 8': 10, # Last year of JSS (CBC)
    'FORM 1': 11,
    'FORM 2': 12,
    'FORM 3': 13,
    'FORM 4': 14,            # End of Secondary
    'FORM 5': 14.5,          # A-Levels (now largely phased out)
    'FORM 6': 14.8,          # A-Levels
    'POST PRIMARY VOCATIONAL TRAINING CERTIFICATE YEAR 1': 11.5,
    'POST PRIMARY VOCATIONAL TRAINING CERTIFICATE YEAR 2': 12,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET CERTIFICATE YEAR 1': 15,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET CERTIFICATE YEAR 2': 15.5,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET DIPLOMA YEAR 1': 15.5,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET DIPLOMA YEAR 2': 16,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET DIPLOMA YEAR 3': 16.5,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 1': 16,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 2': 16.5,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 3': 17,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 4': 17.5,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 5': 18,  # e.g., Medicine
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 6': 18.5,# e.g., Medicine
    'UNIVERSITY POST-GRADUATE DIPLOMA YEAR 1': 18,
    'UNIVERSITY POST-GRADUATE DIPLOMA YEAR 2': 18.5,
    'UNIVERSITY POST-GRADUATE MASTERS DEGREE YEAR 1': 19,
    'UNIVERSITY POST-GRADUATE MASTERS DEGREE YEAR 2': 19.5,
    'UNIVERSITY POST-GRADUATE MASTERS DEGREE YEAR 3': 20,
    'UNIVERSITY POST-GRADUATE MASTERS DEGREE YEAR 4': 20.5,
    'UNIVERSITY POST-GRADUATE DOCTORATES DEGREE YEAR 1': 21,
    'UNIVERSITY POST-GRADUATE DOCTORATES DEGREE YEAR 2': 21.5,
    'UNIVERSITY POST-GRADUATE DOCTORATES DEGREE YEAR 3': 22,
    'UNIVERSITY POST-GRADUATE DOCTORATES DEGREE YEAR 4': 22.5,
    'MADRASSA/DUKSI': 4,     # Mapping religious education to primary level
    'SPECIAL NEEDS EDUCATION': 4 # Mapping to primary level
}
easy_education_mapping = {
    'Never Attended': 0,
    'None': 0,
    'NON-FORMAL EDUCATION (ADULT BASIC EDUCATION)': 0,
    'Play Group': 1,
    'PRE-PRIMARY 1': 1,
    'PRE-PRIMARY 2': 1,
    'STANDARD/GRADE 1': 2,
    'STANDARD/GRADE 2': 2,
    'STANDARD/GRADE 3': 2,
    'STANDARD/GRADE 4': 2,
    'STANDARD/GRADE 5': 2,
    'STANDARD/GRADE 6': 2,
    'STANDARD 7': 2,        # Primary
    'STANDARD 8': 2,        # Primary
    'JUNIOR SCHOOL-GRADE 7': 2, # Primary (CBC)
    'JUNIOR SCHOOL-GRADE 8': 2, # Primary (CBC)
    'FORM 1': 3,
    'FORM 2': 3,
    'FORM 3': 3,
    'FORM 4': 3,            # Secondary
    'FORM 5': 3,            # Secondary (A-Level)
    'FORM 6': 3,            # Secondary (A-Level)
    'POST PRIMARY VOCATIONAL TRAINING CERTIFICATE YEAR 1': 3,
    'POST PRIMARY VOCATIONAL TRAINING CERTIFICATE YEAR 2': 3,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET CERTIFICATE YEAR 1': 4,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET CERTIFICATE YEAR 2': 4,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET DIPLOMA YEAR 1': 4,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET DIPLOMA YEAR 2': 4,
    'MIDDLE LEVEL COLLEGE/POST SECONDARY TVET DIPLOMA YEAR 3': 4,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 1': 5,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 2': 5,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 3': 5,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 4': 5,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 5': 5,
    'UNIVERSITY UNDERGRADUATE DEGREE YEAR 6': 5,
    'UNIVERSITY POST-GRADUATE DIPLOMA YEAR 1': 6,
    'UNIVERSITY POST-GRADUATE DIPLOMA YEAR 2': 6,
    'UNIVERSITY POST-GRADUATE MASTERS DEGREE YEAR 1': 6,
    'UNIVERSITY POST-GRADUATE MASTERS DEGREE YEAR 2': 6,
    'UNIVERSITY POST-GRADUATE MASTERS DEGREE YEAR 3': 6,
    'UNIVERSITY POST-GRADUATE MASTERS DEGREE YEAR 4': 6,
    'UNIVERSITY POST-GRADUATE DOCTORATES DEGREE YEAR 1': 7,
    'UNIVERSITY POST-GRADUATE DOCTORATES DEGREE YEAR 2': 7,
    'UNIVERSITY POST-GRADUATE DOCTORATES DEGREE YEAR 3': 7,
    'UNIVERSITY POST-GRADUATE DOCTORATES DEGREE YEAR 4': 7,
    'MADRASSA/DUKSI': 2,     # Map to Primary level
    'SPECIAL NEEDS EDUCATION': 2 # Map to Primary level
}
category_labels = {
    0: 'No Formal Education',
    1: 'Pre-Primary',
    2: 'Primary',
    3: 'Secondary',
    4: 'TVET/College',
    5: 'Undergraduate',
    6: 'Postgrad Diploma/Masters',
    7: 'Doctorate'
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


def clean_boolean_columns(df, columns=None):
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns.tolist()
    
    mapping = {
        '1': True, 'true': True, 'yes': True, 'y': True,
        '0': False, 'false': False, 'no': False, 'n': False
    }
    
    for col in columns:
        if col in df_clean.columns:
            # Convert only non-null values
            df_clean[col] = (
                df_clean[col]
                .dropna()  # leave NaN untouched
                .astype(str).str.lower()
                .map(mapping)
            ).reindex(df_clean.index)  # restore alignment
    
    return df_clean


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

def plot_counties(relation_ids, prop=None, ax=None):
    gdf = ox.geocoder.geocode_to_gdf(relation_ids, by_osmid=True)
    
    if isinstance(prop, (list, pd.Series)):
        gdf["prop"] = prop
        prop = "prop"
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    if prop is not None and prop in gdf.columns:
        gdf.plot(column=prop, ax=ax, cmap="cividis_r",edgecolor="black", linewidth=0.1 , legend=True)
    else:
        gdf.plot(ax=ax, alpha=0.5, edgecolor="black")
    
    plt.show()
    return
