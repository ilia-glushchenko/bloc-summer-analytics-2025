import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import traceback
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

# Local module import (assuming stats.py is in the same directory or PYTHONPATH)
from stats import load_results, compute_gym_stats, compute_boulder_grades
import config

# Try to import grading system
try:
    from grading_system import FrenchGradingSystem
    GRADING_SYSTEM_AVAILABLE = True
except ImportError:
    GRADING_SYSTEM_AVAILABLE = False
    st.warning("Grading system module not available. Grade-related features will be disabled.")

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
    grading_system: Optional[Any] = None  # FrenchGradingSystem instance

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
def process_data(gender: str = 'men') -> ProcessedData:
    """
    Load and process data from results.json, with caching for performance.
    
    Args:
        gender: Gender category to process ('men' or 'women'). Defaults to 'men'.
    
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
        outlier_warning_message="Data validation failed.", # Default error message
        grading_system=None
    )
    try:
        data = load_results(gender=gender)
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

            # Special handling for climbers with empty gyms array but completed > 0
            if climber['completed'] > 0 and not climber['gyms']:
                # Handle the special case where gyms array is empty but climber has completions
                # Set a reasonable value for Gyms_Active based on other top climbers
                gyms_active_count = 4  # Assume all 4 gyms were visited as this is likely the case for top climbers
                
                # Distribute completions evenly among gyms for display purposes
                even_distribution = climber['completed'] // 4
                remainder = climber['completed'] % 4
                
                for i, gym_name in enumerate(all_gym_names):
                    completed_count = even_distribution + (1 if i < remainder else 0)
                    gym_completions[gym_name] = completed_count
            else:
                # Normal processing for climbers with proper gym data
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
        
        # Compute French boulder grades if grading system is available
        grading_system = None
        if GRADING_SYSTEM_AVAILABLE:
            try:
                grading_system = compute_boulder_grades(data, gender)
                # Removed notification: if grading_system:
                #     st.success(f"Computed French boulder grades for {len(grading_system.boulder_grades)} gyms")
            except Exception as e:
                st.warning(f"Failed to compute boulder grades: {str(e)}")
        
        # Return the results packaged in the dataclass
        return ProcessedData(
            raw_data=data,
            gym_boulder_counts=gym_boulder_counts,
            completion_histograms=completion_histograms,
            participation_counts=participation_counts,
            climbers_df=climbers_df,
            gyms_df=gyms_df,
            outlier_warning_message=outlier_warning_message,
            grading_system=grading_system
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
        
        # Add gender information if available (for combined division)
        if 'gender' in climber_data:
            row['Gender'] = climber_data['gender']
        
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

#------------------------------------------------------------------------------
# COMBINED DIVISION PROCESSING
#------------------------------------------------------------------------------

@st.cache_data(ttl=600)
def process_combined_division() -> ProcessedData:
    """
    Process combined division data by merging men and women divisions.
    
    Important: This function does NOT use the ranks provided in the JSON file.
    Instead, it recalculates ranks based on total completed boulders across
    both divisions to create a truly combined ranking.
    
    GRADING: Uses men's division data for boulder grade calibration to ensure
    consistent absolute grading across all divisions.
    
    Returns:
        ProcessedData: An object containing all processed data structures for combined division.
    """
    outlier_warning_message = None
    # Create empty default structures for error cases
    empty_data = ProcessedData(
        raw_data=[],
        gym_boulder_counts={},
        completion_histograms={},
        participation_counts={},
        climbers_df=pd.DataFrame(),
        gyms_df=pd.DataFrame(),
        outlier_warning_message="Combined division data processing failed.",
        grading_system=None
    )
    
    try:
        # Load both men and women data
        men_data = load_results(gender='men')
        women_data = load_results(gender='women')
        
        # Combine the data and add gender information
        combined_data = []
        
        # Add men data with gender marker
        for climber in men_data:
            climber_copy = climber.copy()
            climber_copy['gender'] = 'M'
            combined_data.append(climber_copy)
        
        # Add women data with gender marker
        for climber in women_data:
            climber_copy = climber.copy()
            climber_copy['gender'] = 'F'
            combined_data.append(climber_copy)
        
        # Sort by completed boulders (descending) and recalculate ranks
        # This ignores the original JSON ranks and creates new combined ranks
        combined_data.sort(key=lambda x: x['completed'], reverse=True)
        
        # Assign new ranks based on completed boulders
        # Handle ties properly by giving same rank to climbers with same completion count
        current_rank = 1
        prev_completed = None
        climbers_with_same_completion = 0
        
        for i, climber in enumerate(combined_data):
            if prev_completed is not None and climber['completed'] != prev_completed:
                # Different completion count - update rank
                current_rank = i + 1
            
            # Assign the rank (ignoring the original JSON rank)
            climber['combined_rank'] = current_rank
            climber['original_rank'] = climber['rank']  # Keep original for reference
            climber['rank'] = current_rank  # Update rank for consistency with other functions
            
            prev_completed = climber['completed']
        
        # Validate the combined data
        if not validate_data(combined_data):
            st.warning("Combined division data validation failed: missing or inconsistent fields detected.")
            return empty_data
        
        # Compute gym stats for combined data
        gym_boulder_counts, completion_histograms, participation_counts = compute_gym_stats(combined_data)
        
        # Create a DataFrame for climbers (similar to process_data but with gender column)
        climbers_data = []
        all_gym_names = sorted(participation_counts.keys())

        for climber in combined_data:
            climber_info = {
                'Climber': climber['climber'],
                'Gender': climber['gender'],
                'Combined_Rank': climber['combined_rank'],
                'Original_Rank': climber['original_rank'],
                'Completed': climber['completed'],
            }
            
            gyms_active_count = 0
            gym_completions = {}

            # Handle special cases and normal processing (same logic as process_data)
            if climber['completed'] > 0 and not climber['gyms']:
                gyms_active_count = 4
                even_distribution = climber['completed'] // 4
                remainder = climber['completed'] % 4
                
                for i, gym_name in enumerate(all_gym_names):
                    completed_count = even_distribution + (1 if i < remainder else 0)
                    gym_completions[gym_name] = completed_count
            else:
                for gym in climber['gyms']:
                    gym_name = gym['gym']
                    completed_count = gym.get('completed', 0) or 0
                    gym_completions[gym_name] = completed_count
                    
                    if completed_count > 0:
                        gyms_active_count += 1
            
            climber_info['Gyms_Active'] = gyms_active_count

            # Add gym-specific completion columns
            for gym_name in all_gym_names:
                safe_gym_name_col = f"Comp_{gym_name.replace(' ', '_')}"
                climber_info[safe_gym_name_col] = gym_completions.get(gym_name, 0)

            climbers_data.append(climber_info)
        
        climbers_df = pd.DataFrame(climbers_data)
        
        # Outlier detection for combined division
        if not climbers_df.empty:
            Q1 = climbers_df['Completed'].quantile(0.25)
            Q3 = climbers_df['Completed'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (climbers_df['Completed'] < lower_bound) | (climbers_df['Completed'] > upper_bound)
            outliers = climbers_df[outlier_mask]
            if not outliers.empty:
                outlier_list = ', '.join(f"{row['Climber']} ({row['Gender']}, {row['Completed']})" for _, row in outliers.iterrows())
                outlier_warning_message = f"Potential outliers detected in 'Completed Boulders' (Combined Division): {outlier_list}. (Values outside {lower_bound:.1f} - {upper_bound:.1f})"
            
            # Calculate average per gym
            climbers_df['Avg_Per_Gym_Active'] = (climbers_df['Completed'] / climbers_df['Gyms_Active']).replace([np.inf, -np.inf], 0).fillna(0).round(1)
        
        # Create gym DataFrame (same logic as process_data)
        gym_data = []
        for gym_name, count in participation_counts.items():
            boulder_counts = gym_boulder_counts.get(gym_name, {})
            total_ascents = sum(boulder_counts.values()) if boulder_counts else 0
            
            completed_counts_filtered = []
            for n_completed, climber_count in completion_histograms.get(gym_name, {}).items():
                if n_completed > 0:
                    completed_counts_filtered.extend([n_completed] * climber_count)
            
            avg_completed_active = 0
            if completed_counts_filtered:
                avg_completed_active = round(np.mean(completed_counts_filtered), 2)

            avg_completed_all = round(total_ascents / count, 2) if count > 0 else 0
            
            max_boulder = 0
            if boulder_counts:
                try:
                    max_boulder = max(int(b) for b in boulder_counts.keys() if b.isdigit())
                except (ValueError, AttributeError) as e:
                    st.warning(f"Warning processing gym {gym_name} in combined division: {str(e)}")
                    max_boulder = 0
            
            gym_data.append({
                'Gym': gym_name,
                'Participants': count,
                'Total_Ascents': total_ascents,
                'Avg_Completed_Per_Active_Climber': avg_completed_active,
                'Avg_Completed_All_Participants': avg_completed_all,
                'Boulder_Count': max_boulder
            })
        
        gyms_df = pd.DataFrame(gym_data)
        
        # Compute French boulder grades using men's division calibration
        # This ensures consistent grading across all divisions
        grading_system = None
        if GRADING_SYSTEM_AVAILABLE:
            try:
                # Always use 'combined' as gender parameter, but compute_boulder_grades
                # will internally use men's division data for calibration
                grading_system = compute_boulder_grades(combined_data, gender='combined')
                logger.info("Computed French boulder grades for combined division using men's division calibration")
            except Exception as e:
                st.warning(f"Failed to compute boulder grades for combined division: {str(e)}")
        
        return ProcessedData(
            raw_data=combined_data,
            gym_boulder_counts=gym_boulder_counts,
            completion_histograms=completion_histograms,
            participation_counts=participation_counts,
            climbers_df=climbers_df,
            gyms_df=gyms_df,
            outlier_warning_message=outlier_warning_message,
            grading_system=grading_system
        )
        
    except Exception as e:
        st.error(f"Error processing combined division data: {str(e)}")
        st.code(traceback.format_exc())
        empty_data.outlier_warning_message = f"Error during combined division processing: {str(e)}"
        return empty_data

#------------------------------------------------------------------------------
# DATACLASS FOR PROCESSED DATA
#------------------------------------------------------------------------------
