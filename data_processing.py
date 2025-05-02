import streamlit as st
import pandas as pd
import numpy as np
import json
import traceback
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Local module import (assuming stats.py is in the same directory or PYTHONPATH)
from stats import load_results, compute_gym_stats

#------------------------------------------------------------------------------
# DATACLASS FOR PROCESSED DATA
#------------------------------------------------------------------------------
@dataclass
class ProcessedData:
    """Holds all the dataframes and results from the processing pipeline."""
    raw_data: List[Dict]
    gym_boulder_counts: Dict
    completion_histograms: Dict
    participation_counts: Dict
    climbers_df: pd.DataFrame
    gyms_df: pd.DataFrame
    outlier_warning_message: Optional[str]

#------------------------------------------------------------------------------
# DATA VALIDATION
#------------------------------------------------------------------------------
# Data validation function
def validate_data(data: List[Dict]) -> bool:
    """
    Validate the structure and types of the input data.
    Checks for required fields, type consistency, and gym/boulder naming consistency.
    Returns True if data is valid, False otherwise.
    """
    required_climber_fields = {'climber', 'rank', 'completed', 'gyms'}
    required_gym_fields = {'gym', 'completed', 'total', 'percent', 'completed_climbs'}
    gym_names = set()
    for climber in data:
        if not required_climber_fields.issubset(climber.keys()):
            st.error(f"Validation Error: Climber missing fields {required_climber_fields - set(climber.keys())} in record: {climber.get('climber', 'Unknown')}")
            return False
        if not isinstance(climber['gyms'], list):
            st.error(f"Validation Error: 'gyms' field is not a list for climber: {climber.get('climber', 'Unknown')}")
            return False
        for gym in climber['gyms']:
            if not required_gym_fields.issubset(gym.keys()):
                 st.error(f"Validation Error: Gym missing fields {required_gym_fields - set(gym.keys())} in record for climber: {climber.get('climber', 'Unknown')}, gym: {gym.get('gym', 'Unknown')}")
                 return False
            if not isinstance(gym['completed_climbs'], list):
                 st.error(f"Validation Error: 'completed_climbs' is not a list for climber: {climber.get('climber', 'Unknown')}, gym: {gym.get('gym', 'Unknown')}")
                 return False
            gym_names.add(gym['gym'])
    # Optionally: check for consistent gym naming (case, whitespace)
    normalized_names = {g.strip().lower() for g in gym_names}
    if len(normalized_names) != len(gym_names):
        st.warning("Validation Warning: Potential inconsistencies in gym naming conventions (e.g., capitalization, spacing).")
        # Decide if this is a fatal error or just a warning
        # return False # Uncomment if strict consistency is required
    return True

#------------------------------------------------------------------------------
# DATA PROCESSING
#------------------------------------------------------------------------------
# Data processing function
@st.cache_data(ttl=600)
def process_data() -> ProcessedData:
    """
    Load and process data from results.json, with caching for performance.
    
    Returns:
        ProcessedData: An object containing all processed data structures.
    """
    outlier_warning_message = None # Initialize
    # Create empty default structures for error cases
    empty_data = ProcessedData(
        raw_data=[],
        gym_boulder_counts={},
        completion_histograms={},
        participation_counts={},
        climbers_df=pd.DataFrame(),
        gyms_df=pd.DataFrame(),
        outlier_warning_message="Data validation failed." # Default error message
    )
    try:
        data = load_results()
        if not validate_data(data):
            st.warning("Data validation failed: missing or inconsistent fields detected. Please check your results.json file.")
            # Return the empty structure with validation error message
            return empty_data
        gym_boulder_counts, completion_histograms, participation_counts = compute_gym_stats(data)
        
        # Create a DataFrame for climbers
        climbers_data = []
        all_gym_names = sorted(participation_counts.keys()) # Get all unique gym names

        for climber in data:
            climber_info = {
                'Climber': climber['climber'],
                'Rank': climber['rank'],
                'Completed': climber['completed'], # Official total completed
            }
            
            gyms_active_count = 0
            gym_completions = {} # Store completions per gym for this climber

            for gym in climber['gyms']:
                gym_name = gym['gym']
                completed_count = gym.get('completed', 0) or 0 # Ensure it's an int
                gym_completions[gym_name] = completed_count
                
                if completed_count > 0:
                    gyms_active_count += 1
            
            climber_info['Gyms_Active'] = gyms_active_count # New column: gyms with >0 completions

            # Add gym-specific completion columns dynamically
            for gym_name in all_gym_names:
                # Use a consistent column name format, e.g., 'Comp_GymName'
                # Ensure gym names are valid for column headers (replace spaces, etc. if needed)
                safe_gym_name_col = f"Comp_{gym_name.replace(' ', '_')}" 
                climber_info[safe_gym_name_col] = gym_completions.get(gym_name, 0) # Default to 0 if climber didn't visit/complete in this gym

            climbers_data.append(climber_info)
        
        climbers_df = pd.DataFrame(climbers_data)
        
        # Outlier detection for 'Completed' using IQR
        if not climbers_df.empty:
            Q1 = climbers_df['Completed'].quantile(0.25)
            Q3 = climbers_df['Completed'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (climbers_df['Completed'] < lower_bound) | (climbers_df['Completed'] > upper_bound)
            outliers = climbers_df[outlier_mask]
            if not outliers.empty:
                outlier_list = ', '.join(f"{row['Climber']} ({row['Completed']})" for _, row in outliers.iterrows())
                # Store the warning message instead of printing it directly
                outlier_warning_message = f"Potential outliers detected in 'Completed Boulders': {outlier_list}. (Values outside {lower_bound:.1f} - {upper_bound:.1f})"
            
            # Calculate Avg_Per_Gym based on Gyms_Active AFTER potential outliers identified (but before returning)
            # Avoid division by zero errors
            climbers_df['Avg_Per_Gym_Active'] = (climbers_df['Completed'] / climbers_df['Gyms_Active']).replace([np.inf, -np.inf], 0).fillna(0).round(1)
        
        # Create a DataFrame for gyms
        gym_data = []
        for gym_name, count in participation_counts.items():
            boulder_counts = gym_boulder_counts.get(gym_name, {})
            total_ascents = sum(boulder_counts.values()) if boulder_counts else 0
            
            # Calculate average completed per climber who completed at least one boulder
            completed_counts_filtered = []
            for n_completed, climber_count in completion_histograms.get(gym_name, {}).items():
                if n_completed > 0:  # Only include climbers who completed at least one boulder
                    completed_counts_filtered.extend([n_completed] * climber_count)
            
            avg_completed_active = 0
            if completed_counts_filtered:
                avg_completed_active = round(np.mean(completed_counts_filtered), 2)

            # Calculate average completed per participant (including those with 0 completions)
            avg_completed_all = round(total_ascents / count, 2) if count > 0 else 0
            
            # Find maximum boulder number safely
            max_boulder = 0
            if boulder_counts:
                try:
                    max_boulder = max(int(b) for b in boulder_counts.keys() if b.isdigit())
                except (ValueError, AttributeError) as e:
                    st.warning(f"Warning processing gym {gym_name}: {str(e)}")
                    max_boulder = 0
            
            gym_data.append({
                'Gym': gym_name,
                'Participants': count,
                'Total_Ascents': total_ascents,
                'Avg_Completed_Per_Active_Climber': avg_completed_active,
                'Avg_Completed_All_Participants': avg_completed_all, # Added new metric
                'Boulder_Count': max_boulder
            })
        
        gyms_df = pd.DataFrame(gym_data)
        
        # Return the results packaged in the dataclass
        return ProcessedData(
            raw_data=data,
            gym_boulder_counts=gym_boulder_counts,
            completion_histograms=completion_histograms,
            participation_counts=participation_counts,
            climbers_df=climbers_df,
            gyms_df=gyms_df,
            outlier_warning_message=outlier_warning_message
        )
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.code(traceback.format_exc())
        # Return empty data structure with specific processing error message
        empty_data.outlier_warning_message = f"Error during data processing: {str(e)}"
        return empty_data

#------------------------------------------------------------------------------
# CLIMBER-BOULDER MATRIX (for recommendations)
#------------------------------------------------------------------------------
# Function to create climber-boulder matrix for recommendation engine
@st.cache_data(ttl=600)
def create_climber_boulder_matrix(data: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]]]:
    """
    Create a matrix representation of climbers and their boulder completions for the recommendation system.
    
    This function builds two key data structures:
    1. A DataFrame with climbers as rows and boulders as columns (one-hot encoded)
    2. A nested dictionary mapping climbers to gyms to completed boulder lists
    
    Each boulder is identified by its gym and number (e.g., "Gym1_12" for boulder #12 at Gym1).
    For visited gyms, only gyms where the climber has actually completed boulders are included.
    
    Parameters:
    -----------
    data : List[Dict]
        The raw competition data loaded from results.json
        
    Returns:
    --------
    matrix_df : pd.DataFrame
        One-hot encoded matrix with climbers as rows and boulders as columns
    climber_gym_boulders : Dict[str, Dict[str, List[str]]]
        Nested dictionary mapping climbers to gyms to completed boulder lists
    """
    # Create a mapping of climbers to their completed boulders per gym
    climber_gym_boulders = {}
    for climber_data in data:
        climber_name = climber_data['climber']
        climber_gym_boulders[climber_name] = {}
        
        for gym_data in climber_data['gyms']:
            gym_name = gym_data['gym']
            completed_climbs = gym_data.get('completed_climbs', [])
            # Only add the gym if there are completed climbs
            if completed_climbs:  # This ensures empty gyms aren't counted as "visited"
                climber_gym_boulders[climber_name][gym_name] = completed_climbs
    
    # Create a single dataframe with all climbers and their boulder completions
    rows = []
    for climber_data in data:
        climber_name = climber_data['climber']
        rank = climber_data['rank']
        completed = climber_data['completed']
        
        # Add a row for overall climber stats
        row = {
            'Climber': climber_name,
            'Rank': rank,
            'Completed': completed
        }
        
        # Add completed boulders for each gym
        for gym_data in climber_data['gyms']:
            gym_name = gym_data['gym']
            completed_climbs = gym_data.get('completed_climbs', [])
            
            for boulder in completed_climbs:
                boulder_key = f"{gym_name}_{boulder}"
                row[boulder_key] = 1
        
        rows.append(row)
    
    # Create dataframe
    matrix_df = pd.DataFrame(rows)
    matrix_df = matrix_df.fillna(0)
    
    # Removed the unused top_10_total calculation
    
    return matrix_df, climber_gym_boulders 