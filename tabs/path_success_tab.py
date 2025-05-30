import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any, Optional, Callable
import html # Import the html module for escaping

# Assuming components are accessible
from ui_components import render_metrics_row, render_section_header
from data_processing import create_climber_boulder_matrix
from recommendations import find_similar_climbers, recommend_boulders
from utils import get_default_selected_climber
import config # Import config

# Try to import grading system
try:
    from grading_system import FrenchGradingSystem
    GRADING_SYSTEM_AVAILABLE = True
except ImportError:
    GRADING_SYSTEM_AVAILABLE = False

# Configuration values moved to config.py
# TARGET_RANK = 10 
# SIMILARITY_N = 5 
DEFAULT_CLIMBER = get_default_selected_climber() or "[Select Climber]" # Get default from env or provide fallback

# --- Helper functions specific to this tab ---

def display_similar_climbers(similar_climbers: List[Tuple[str, float, int, int]]) -> None:
    """
    Display a table of climbers with similar climbing styles using st.dataframe.
    Uses CSS classes for styling.
    """
    if similar_climbers:      
        render_section_header("Similar Climbers", level=4)

        # Create the dataframe without HTML formatting
        similar_df = pd.DataFrame(
            similar_climbers,
            columns=['Climber', 'Similarity', 'Rank', 'Completed']
        )
        similar_df['Similarity'] = (similar_df['Similarity'] * 100)
        
        st.dataframe(
            similar_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Climber": st.column_config.TextColumn(help="Climber name"),
                "Similarity": st.column_config.NumberColumn(
                    "Similarity (%)",
                    help="Percentage similarity to selected climber",
                    format="%.1f%%"
                ),
                "Rank": st.column_config.NumberColumn(format="%d"),
                "Completed": st.column_config.NumberColumn("Boulders Completed", format="%d")
            }
        )

def create_recommendation_dataframe(
    recommendations: Dict[str, List[Tuple[str, float]]],
    unvisited_gyms: Set[str],
    gym_boulder_counts: Dict[str, Dict[str, int]],
    participation_counts: Dict[str, int],
    all_gyms_in_dataset: Set[str]
) -> pd.DataFrame:
    """
    Create a comprehensive DataFrame with all boulder recommendations,
    including flags for new gyms.
    """
    all_recommendations = []
    
    for gym, boulder_scores in recommendations.items():
        is_new_gym = gym in unvisited_gyms
        if not boulder_scores and is_new_gym:
            all_recommendations.append({
                'Gym': gym,
                'Boulder': 'No specific recommendations',
                'Probability': 0,
                'Success Probability': 0,
                'New Gym': is_new_gym # Keep boolean flag
            })
        else:
            for boulder, score in boulder_scores:
                boulder_count = gym_boulder_counts.get(gym, {}).get(boulder, 0)
                total_climbers = participation_counts.get(gym, 0)
                success_prob = boulder_count / total_climbers * 100 if total_climbers > 0 else 0
                
                all_recommendations.append({
                    'Gym': gym,
                    'Boulder': boulder,
                    'Probability': score, # Raw score before formatting
                    'Success Probability': success_prob, # Raw probability before formatting
                    'New Gym': is_new_gym # Keep boolean flag
                })
    
    # Add placeholders for any remaining unvisited gyms not in recommendations
    for gym in all_gyms_in_dataset:
        if gym in unvisited_gyms and gym not in recommendations:
             all_recommendations.append({
                'Gym': gym,
                'Boulder': 'No specific recommendations',
                'Probability': 0,
                'Success Probability': 0,
                'New Gym': True
            })

    if all_recommendations:
        rec_df = pd.DataFrame(all_recommendations)
        rec_df = rec_df.sort_values(by=['Probability', 'Success Probability'], ascending=[False, False])
        
        # Format scores AFTER sorting
        rec_df['Probability'] = (rec_df['Probability'] * 100).round(1)
        rec_df['Success Probability'] = rec_df['Success Probability'].round(1)
        
        return rec_df
    else:
        return pd.DataFrame()

def display_visited_gyms_recommendations(visited_gyms_list: List[str], rec_df: pd.DataFrame, 
                                   enhanced_recommendations: Optional[Dict[str, List[Tuple[str, float, str]]]] = None,
                                   processed_data: Optional[Any] = None) -> None:
    """Display recommendations for gyms the climber has already visited."""
    if visited_gyms_list:
        # Add custom CSS to ensure tables don't get scrollbars and to make the layout responsive
        st.markdown("""
        <style>
        [data-testid="stDataFrame"] {
            width: 100%;
        }
        [data-testid="stDataFrame"] > div {
            max-height: none !important;
            overflow: visible !important;
        }
        [data-testid="stDataFrame"] [data-testid="dataFrameContainer"] {
            max-height: none !important;
            overflow: visible !important;
        }
        /* Responsive grid layout */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 1rem;
        }
        [data-testid="column"] {
            min-width: 250px;
            flex: 1 1 auto !important;
        }
        /* Make sure table cells don't cause horizontal scrolling */
        .dataframe-container {
            width: 100% !important;
            overflow-x: hidden !important;
        }
        thead th {
            word-break: break-word !important;
        }
        tbody td {
            word-break: break-word !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Show explanation about keys and recommendation scores before the gyms
        st.markdown("""
        <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #4CAF50; background-color: #f8f9fa;">
            <b>Understanding Boulder Probabilities:</b><br>
            📊 <b>Probability</b>: Shows the Bayesian probability of completion based on a combination of general success rates and similar climber data.<br>
            📈 <b>Success Rate</b>: Shows the historical completion rate of the boulder across all climbers.
        </div>
        """, unsafe_allow_html=True)
        
        # Get number of gyms
        num_gyms = len(visited_gyms_list)
        
        # Auto-detect screen width and adjust columns - Ultra-wide support
        # Use session state to check if user has manually selected a layout preference
        if 'visited_layout_preference' not in st.session_state:
            # Default layouts based on number of gyms
            if num_gyms <= 3:
                cols_per_row = num_gyms  # 1, 2, or 3 columns
            elif num_gyms == 4:
                cols_per_row = 2  # 2x2 grid for 4 gyms
            else:
                # For 5+ gyms, use layout selection to allow responsive layouts
                layout_options = ["Auto-detect", "2 columns", "3 columns", "4 columns", "5 columns"]
                default_index = 0  # Auto-detect is default
                
                layout_choice = st.radio(
                    "Layout for visited gyms:", 
                    layout_options, 
                    horizontal=True, 
                    key="visited_layout_radio",
                    index=default_index
                )
                
                if layout_choice == "Auto-detect":
                    # Let CSS handle the layout (will be responsive based on screen width)
                    cols_per_row = min(5, max(2, (num_gyms + 1) // 2))  # Dynamic column count, max 5
                elif layout_choice == "2 columns":
                    cols_per_row = 2
                elif layout_choice == "3 columns":
                    cols_per_row = 3
                elif layout_choice == "4 columns":
                    cols_per_row = 4
                else:  # 5 columns
                    cols_per_row = 5
                
                # Store choice in session state
                st.session_state.visited_layout_preference = cols_per_row
        else:
            # Use stored preference
            cols_per_row = st.session_state.visited_layout_preference
            
        # Create columns based on determined layout
        cols = st.columns(cols_per_row)
        
        for i, gym_name in enumerate(visited_gyms_list):
            col_idx = i % cols_per_row
            with cols[col_idx]:
                # Use smaller header for gym name, escaping the name
                st.markdown(f"<h6>🏠 {html.escape(gym_name)}</h6>", unsafe_allow_html=True)
                gym_recs = rec_df[(rec_df['Gym'] == gym_name) & (rec_df['New Gym'] == False)].copy()

                if not gym_recs.empty and not (len(gym_recs) == 1 and gym_recs['Boulder'].iloc[0] == 'No specific recommendations'):
                    # For the new Bayesian model, we don't have enhanced recommendations with insights
                    # Just display the basic recommendation data
                    gym_recs_display = gym_recs[['Boulder', 'Probability', 'Success Probability']].copy()
                    
                    # Add French Grade column if grading system is available
                    if processed_data and hasattr(processed_data, 'grading_system') and processed_data.grading_system:
                        gym_recs_display['French Grade'] = gym_recs_display['Boulder'].apply(
                            lambda boulder: get_boulder_french_grade_for_display(processed_data.grading_system, gym_name, str(boulder))
                        )
                    
                    st.dataframe(
                        gym_recs_display,
                        use_container_width=True,
                        hide_index=True,
                        height=None,  # Explicitly set to None to show all rows
                        column_config={
                            "Boulder": st.column_config.TextColumn("Boulder"),
                            "Probability": st.column_config.NumberColumn(
                                "Probability (%)",  # Restored original label
                                help="Bayesian probability of completion",
                                format="%.1f%%"
                            ),
                            "Success Probability": st.column_config.NumberColumn(
                                "Success Rate (%)",  # Restored original label
                                help="Historical completion rate",
                                format="%.1f%%"
                            ),
                            "French Grade": st.column_config.TextColumn(
                                "Est. Grade",
                                help="Estimated French boulder grade"
                            ) if 'French Grade' in gym_recs_display.columns else None
                        }
                    )
                else:
                    st.info(f"No specific recommendations available for {gym_name} based on your current climbs.")
    else:
        st.info("No recommendations found for gyms you've already visited.")

def display_unvisited_gyms_recommendations(
    unvisited_gyms_list: List[str],
    rec_df: pd.DataFrame,
    similar_climbers: List[Tuple[str, float, int, int]],
    climber_gym_boulders: Dict[str, Dict[str, List[str]]],
    processed_data: Optional[Any] = None
) -> None:
    """Display recommendations for gyms the climber hasn't visited yet."""
    if unvisited_gyms_list:
        # Get number of gyms
        num_gyms = len(unvisited_gyms_list)
        
        # Auto-detect screen width and adjust columns - Ultra-wide support
        # Use session state to check if user has manually selected a layout preference
        if 'unvisited_layout_preference' not in st.session_state:
            # Default layouts based on number of gyms
            if num_gyms <= 3:
                cols_per_row = num_gyms  # 1, 2, or 3 columns
            elif num_gyms == 4:
                cols_per_row = 2  # 2x2 grid for 4 gyms
            else:
                # For 5+ gyms, use layout selection to allow responsive layouts
                layout_options = ["Auto-detect", "2 columns", "3 columns", "4 columns", "5 columns"]
                default_index = 0  # Auto-detect is default
                
                layout_choice = st.radio(
                    "Layout for unvisited gyms:", 
                    layout_options, 
                    horizontal=True, 
                    key="unvisited_layout_radio",
                    index=default_index
                )
                
                if layout_choice == "Auto-detect":
                    # Let CSS handle the layout (will be responsive based on screen width)
                    cols_per_row = min(5, max(2, (num_gyms + 1) // 2))  # Dynamic column count, max 5
                elif layout_choice == "2 columns":
                    cols_per_row = 2
                elif layout_choice == "3 columns":
                    cols_per_row = 3
                elif layout_choice == "4 columns":
                    cols_per_row = 4
                else:  # 5 columns
                    cols_per_row = 5
                
                # Store choice in session state
                st.session_state.unvisited_layout_preference = cols_per_row
        else:
            # Use stored preference
            cols_per_row = st.session_state.unvisited_layout_preference
            
        # Create columns based on determined layout
        columns = st.columns(cols_per_row)
        
        # Get just the climber names from similar_climbers
        similar_climber_names = [c[0] for c in similar_climbers]
        
        for i, gym_name in enumerate(unvisited_gyms_list):
            col_idx = i % cols_per_row
            with columns[col_idx]:
                # Use smaller header for gym name, escaping the name
                st.markdown(f"<h6>🆕 {html.escape(gym_name)}</h6>", unsafe_allow_html=True)
                
                # Get recommendations for this gym
                gym_recs = rec_df[rec_df['Gym'] == gym_name]
                
                if gym_recs.empty or (len(gym_recs) == 1 and gym_recs['Boulder'].iloc[0] == 'No specific recommendations'):
                    st.info(f"No specific recommendations for {gym_name}.")
                    
                    # Find similar climbers who visited this gym
                    gym_visitors = []
                    for climber in similar_climber_names:
                        if climber in climber_gym_boulders and gym_name in climber_gym_boulders[climber]:
                            # Count boulders completed by this climber at this gym
                            boulders_completed = len(climber_gym_boulders[climber][gym_name])
                            gym_visitors.append((climber, boulders_completed))
                    
                    if gym_visitors:
                        st.markdown("**Similar climbers who visited this gym:**")
                        
                        # Create DataFrame for display
                        visitors_df = pd.DataFrame(gym_visitors, columns=['Climber', 'Boulders Completed'])
                        
                        # Display with highlight support
                        st.dataframe(
                            visitors_df, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "Climber": st.column_config.TextColumn("Climber"),
                                "Boulders Completed": st.column_config.NumberColumn(format="%d")
                            }
                        )
                    else:
                        st.write("None of your similar climbers have visited this gym.")
                else:
                    # Display actual recommendations
                    gym_recs_display = gym_recs[['Boulder', 'Probability', 'Success Probability']].copy()
                    
                    # Add French Grade column if grading system is available
                    if processed_data and hasattr(processed_data, 'grading_system') and processed_data.grading_system:
                        gym_recs_display['French Grade'] = gym_recs_display['Boulder'].apply(
                            lambda boulder: get_boulder_french_grade_for_display(processed_data.grading_system, gym_name, str(boulder))
                        )
                    
                    st.dataframe(
                        gym_recs_display,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Boulder": st.column_config.TextColumn("Boulder"),
                            "Probability": st.column_config.NumberColumn(
                                "Probability (%)",  # Restored original label
                                format="%.1f%%",
                                help="Bayesian probability of completion"
                            ),
                            "Success Probability": st.column_config.NumberColumn(
                                "Success Rate (%)",  # Restored original label
                                format="%.1f%%",
                                help="Historical completion rate"
                            ),
                            "French Grade": st.column_config.TextColumn(
                                "Est. Grade",
                                help="Estimated French boulder grade"
                            ) if 'French Grade' in gym_recs_display.columns else None
                        }
                    )

def get_boulder_french_grade_for_display(grading_system, gym_name: str, boulder_id: str) -> str:
    """
    Get the estimated French grade for a boulder for display purposes.
    
    Args:
        grading_system: FrenchGradingSystem instance
        gym_name: Name of the gym
        boulder_id: ID of the boulder
        
    Returns:
        French grade string or "N/A" if not available
    """
    if not grading_system:
        return "N/A"
    
    boulder_grade = grading_system.get_boulder_grade(gym_name, boulder_id)
    if boulder_grade:
        return boulder_grade.french_grade
    return "N/A"

# --- Main function for the tab ---

def display_path_success(data: List[Dict], climbers_df: pd.DataFrame, gym_boulder_counts: Dict, participation_counts: Dict, sync_callback: Optional[Callable] = None, processed_data: Optional[Any] = None) -> None:
    """Displays the Path to Success tab content with French grading when available."""
    st.markdown("### Path to Success Analysis")

    if climbers_df.empty:
        st.warning("No climber data available for analysis.")
        return
    
    climber_list = sorted(climbers_df['Climber'].unique())
    # Add a placeholder if no default is found
    default_climber = get_default_selected_climber() # Read from env via util
    default_index = 0
    if default_climber and default_climber in climber_list:
        default_index = climber_list.index(default_climber)
    elif DEFAULT_CLIMBER != "[Select Climber]" and DEFAULT_CLIMBER in climber_list: # Check module fallback
        default_index = climber_list.index(DEFAULT_CLIMBER)

    # Create columns for climber selection and target rank
    col_climber, col_target = st.columns([3, 1])
    
    with col_climber:
        # Use constant for session key
        selected_climber = st.selectbox(
            "Select your Climber",
            climber_list,
            index=default_index,
            key=config.SESSION_KEY_SELECTED_CLIMBER_PATH_TAB, # Use specific key for this widget
            on_change=sync_callback # Trigger sync when selection changes
        )
    
    # Update the main session state (redundant if callback works, but safe)
    # Use constant for session key
    if selected_climber != st.session_state.get(config.SESSION_KEY_SELECTED_CLIMBER, None):
        st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] = selected_climber
        # No rerun needed here if on_change handles it

    # Ensure selected climber is valid before proceeding
    if not selected_climber or selected_climber == "[Select Climber]":
        st.info("Please select a climber to see their path analysis.")
        return

    # --- Data Preparation ---
    try:
        # Use constant for session key
        climber_data = climbers_df[climbers_df['Climber'] == st.session_state[config.SESSION_KEY_SELECTED_CLIMBER]].iloc[0]
        matrix_df, climber_gym_boulders = create_climber_boulder_matrix(data)
        
        # Check if selected climber exists in the matrix
        if st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] not in matrix_df['Climber'].values:
            st.error(f"Selected climber '{st.session_state[config.SESSION_KEY_SELECTED_CLIMBER]}' not found in the recommendation matrix. Data might be inconsistent.")
            return
            
    except IndexError:
        # Use constant for session key
        st.error(f"Could not find data for selected climber: {st.session_state.get(config.SESSION_KEY_SELECTED_CLIMBER, 'N/A')}")
        return
    except Exception as e:
        st.error(f"Error preparing data for recommendations: {e}")
        return
        
    # Get current climber's rank for validation
    # Handle both regular 'Rank' and combined division 'Combined_Rank' columns
    rank_column = 'Combined_Rank' if 'Combined_Rank' in climber_data.index else 'Rank'
    current_rank = climber_data[rank_column]
    
    # Add input for target rank with validation
    # Initialize session state for target rank if not already set
    if config.SESSION_KEY_TARGET_RANK not in st.session_state:
        st.session_state[config.SESSION_KEY_TARGET_RANK] = config.PATH_TARGET_RANK

    # Ensure the session state target rank doesn't exceed the maximum allowed value
    max_allowed_rank = current_rank - 1 if current_rank > 1 else 1
    if st.session_state[config.SESSION_KEY_TARGET_RANK] > max_allowed_rank:
        st.session_state[config.SESSION_KEY_TARGET_RANK] = max_allowed_rank

    # Add target rank selector with validation in the second column
    with col_target:
        # Create a callback function to update target rank without changing tab
        def update_target_rank():
            st.session_state[config.SESSION_KEY_TARGET_RANK] = st.session_state["temp_target_rank"]
        
        target_rank = st.number_input(
            "Target Rank",
            min_value=config.PATH_TARGET_RANK_MIN,
            max_value=max_allowed_rank,
            value=st.session_state[config.SESSION_KEY_TARGET_RANK],
            step=1,
            help=f"Select your target rank (between {config.PATH_TARGET_RANK_MIN} and your current rank {current_rank})",
            key="temp_target_rank",
            on_change=update_target_rank
        )
    
    # --- Analysis and Display ---
    # Recalculate top_10_target (minimum boulders for target rank) - now using selected target rank
    top_10_target = 0
    if not climbers_df.empty:
        # Use selected target rank instead of fixed config value
        # Use the same rank column that exists in the DataFrame
        target_rank_climbers = climbers_df[climbers_df[rank_column] == target_rank]
        if not target_rank_climbers.empty:
            # If climbers exist exactly at the target rank, use their boulder count
            top_10_target = target_rank_climbers.iloc[0]['Completed']
        else:
            # Otherwise, find the minimum boulders among those ranked better than or equal to the target
            top_rank_climbers = climbers_df[climbers_df[rank_column] <= target_rank]
            if not top_rank_climbers.empty:
                top_10_target = top_rank_climbers['Completed'].min()
            # If still 0 (no climbers at or better than target rank), it remains 0, which is handled later.

    # Use constant for session key
    boulders_completed = climber_data['Completed']
    avg_per_gym = climber_data['Avg_Per_Gym_Active']
    active_gyms = climber_data['Gyms_Active']

    # Get total climbers for rank display
    total_climbers = len(climbers_df)
    
    # Get total boulders in dataset
    # First, find all unique boulders across all gyms
    all_boulders = set()
    for climber, gyms_dict in climber_gym_boulders.items():
        for gym, boulders in gyms_dict.items():
            all_boulders.update(boulders)
    total_boulders = len(all_boulders)

    # One column for metrics
    render_metrics_row({
        "Current Rank": f"{current_rank}/{total_climbers}",
        "Boulders Completed": f"{boulders_completed}/{total_boulders}",
        "Active Gyms Visited": f"{active_gyms}",
        "Avg Boulders/Active Gym": f"{avg_per_gym:.1f}"
    })
    
    # Update success/info messages to use the selected target rank
    if current_rank <= target_rank:
        st.success(f"🎉 Congratulations! You are already ranked {current_rank}, meeting the Top {target_rank} goal!")
    else:
        st.info(f"Currently ranked {current_rank}. Let's find a path to the Top {target_rank}.")

    # Find similar climbers using the configured N
    # Use constant for session key and N
    similar_climbers = find_similar_climbers(
        matrix_df, 
        st.session_state[config.SESSION_KEY_SELECTED_CLIMBER], 
        n_similar=config.PATH_SIMILARITY_N
    )
    display_similar_climbers(similar_climbers)

    # Generate recommendations (call moved here, after similar climbers are displayed)
    try:
        with st.spinner("Generating boulder recommendations..."):
            recommendations, unvisited_gyms = recommend_boulders(
                matrix_df=matrix_df,
                target_climber=st.session_state[config.SESSION_KEY_SELECTED_CLIMBER],
                climber_gym_boulders=climber_gym_boulders,
                similar_climbers=similar_climbers, 
                top_10_target=top_10_target,       
                gym_boulder_counts=gym_boulder_counts,
                participation_counts=participation_counts
            )
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        # Debug exception handling removed for production
        return # Stop if recommendations fail
    
    all_gyms_in_dataset = set(participation_counts.keys())
    rec_df = create_recommendation_dataframe(recommendations, unvisited_gyms, gym_boulder_counts, participation_counts, all_gyms_in_dataset)
    
    # Separate visited and unvisited gyms for display
    visited_gyms = set(climber_gym_boulders.get(st.session_state[config.SESSION_KEY_SELECTED_CLIMBER], {}).keys())
    visited_gyms_with_recs = sorted([gym for gym in visited_gyms if gym in recommendations])
    unvisited_gyms_list = sorted(list(unvisited_gyms))
    
    # Check if there are any actual boulder recommendations to show
    has_actual_recommendations = False
    if not rec_df.empty:
        has_actual_recommendations = not rec_df[rec_df['Boulder'] != 'No specific recommendations'].empty
        
    if has_actual_recommendations:
        # Original UI Structure: Show visited gyms first, then unvisited
        if visited_gyms_with_recs:
            render_section_header("Boulders in Visited Gyms", level=5)
            display_visited_gyms_recommendations(visited_gyms_with_recs, rec_df, None, processed_data)
        
        if unvisited_gyms_list:
            render_section_header("Boulders in Unvisited Gyms", level=5)
            display_unvisited_gyms_recommendations(
                unvisited_gyms_list, 
                rec_df, 
                similar_climbers,
                climber_gym_boulders,
                processed_data
            )
        
        # --- Optimized Path Section --- 
        # Check if the special key exists from the backend for the optimized path
        if "__TOP_10_PATH__" in recommendations and recommendations["__TOP_10_PATH__"]:
            render_section_header(f"Optimized Path to Top {target_rank}", level=4)
            
            st.markdown(f"""
            This is your step-by-step path to reach top {target_rank}, using a Bayesian probability model
            that calculates the true likelihood of completing each boulder. This approach combines
            general success rates with data from similar climbers, weighted by confidence.
            """)
            
            # Add explanation about gym sorting
            st.markdown(f"""
            <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #1E88E5; background-color: #f8f9fa;">
                <b>Optimized Gym Order:</b> Gyms are sorted by potential contribution to your top {target_rank} goal, 
                with those offering the most high-probability boulders listed first. Within each gym, 
                boulders are sorted by completion probability (highest first).
            </div>
            """, unsafe_allow_html=True)

            # Add explanation of the columns
            st.markdown("""
            <div style="margin-bottom: 15px; padding: 10px; border-left: 3px solid #FF9800; background-color: #fff3e0;">
                <b>Understanding the Data:</b>
                <ul style="margin: 5px 0 0 15px; padding: 0;">
                    <li><b>Completion Probability</b>: Your personalized probability of completing the boulder, based on Bayesian analysis of your climbing pattern and similar climbers.</li>
                    <li><b>Historical Success</b>: The percentage of all climbers who have successfully completed this boulder, regardless of climbing style.</li>
                </ul>
                <small style="display: block; margin-top: 5px;">The difference between these values can indicate boulders that match your specific climbing strengths.</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a container for the path display
            path_container = st.container()
            
            with path_container:
                # Add CSS for responsive grid layout for the path display
                st.markdown("""
                <style>
                /* Ensure expanders display properly in the responsive grid */
                .streamlit-expanderContent {
                    width: 100%;
                    overflow-x: hidden !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display the optimized path using the format from __TOP_10_PATH__
                gym_boulders = {}
                gym_visit_recommendations = {}
                
                for item, score in recommendations["__TOP_10_PATH__"]:
                    if "_" in item:
                        gym, boulder = item.split("_", 1)
                        if gym not in gym_boulders:
                            gym_boulders[gym] = []
                        gym_boulders[gym].append((boulder, score))
                    else:
                        gym = item
                        gym_visit_recommendations[gym] = score
                        if gym not in gym_boulders:
                            gym_boulders[gym] = []
                
                if gym_boulders:
                    total_boulders_in_path = 0
                    gym_potential = []
                    for gym, boulders in gym_boulders.items():
                        boulder_count = len(boulders)
                        total_probability = sum(score for _, score in boulders)
                        potential_score = boulder_count * 10 + total_probability * 100
                        gym_potential.append((gym, boulders, potential_score, boulder_count))
                    
                    gym_potential.sort(key=lambda x: x[2], reverse=True)
                    
                    # Create a responsive grid layout for the gym expanders
                    # Calculate number of gyms to display
                    num_gyms = len(gym_potential)
                    
                    # Use the CSS from style.css for responsiveness
                    # We'll display gyms in a responsive grid
                    
                    # Group gyms by potential (high/medium/low) for better organization if many gyms
                    gym_groups = []
                    
                    # If more than 6 gyms, group them
                    if num_gyms > 6:
                        # Use potential score to divide into high/medium/low groups
                        scores = [score for _, _, score, _ in gym_potential]
                        max_score = max(scores) if scores else 0
                        min_score = min(scores) if scores else 0
                        score_range = max_score - min_score if max_score > min_score else 1
                        
                        high_gyms = []
                        medium_gyms = []
                        low_gyms = []
                        
                        for gym, boulders, score, count in gym_potential:
                            normalized_score = (score - min_score) / score_range
                            if normalized_score > 0.66:
                                high_gyms.append((gym, boulders, score, count))
                            elif normalized_score > 0.33:
                                medium_gyms.append((gym, boulders, score, count))
                            else:
                                low_gyms.append((gym, boulders, score, count))
                        
                        if high_gyms:
                            gym_groups.append(("High Potential Gyms", high_gyms))
                        if medium_gyms:
                            gym_groups.append(("Medium Potential Gyms", medium_gyms))
                        if low_gyms:
                            gym_groups.append(("Lower Potential Gyms", low_gyms))
                    else:
                        # Just one group for all gyms
                        gym_groups.append(("Optimized Gym Order", gym_potential))
                    
                    # Process each group
                    for group_name, gyms in gym_groups:
                        if group_name != "Optimized Gym Order":
                            st.markdown(f"##### {group_name}")
                        
                        # For each group, create a responsive grid of gyms
                        num_gyms_in_group = len(gyms)
                        
                        # Auto-detect screen width and adjust columns - Ultra-wide support
                        # Default layouts based on number of gyms
                        if num_gyms_in_group <= 3:
                            cols_per_row = num_gyms_in_group  # 1, 2, or 3 columns depending on gym count
                        elif num_gyms_in_group == 4:
                            cols_per_row = 2  # For 4 gyms, use 2 columns to get 2x2 grid
                        elif num_gyms_in_group >= 5 and group_name != "Optimized Gym Order":
                            # For groups with 5+ gyms, offer layout selection only if this isn't the first run
                            if 'optimized_layout_preference' not in st.session_state:
                                # Let CSS handle the layout (will be responsive based on screen width)
                                cols_per_row = min(5, max(2, (num_gyms_in_group + 1) // 2))  # Dynamic column count, max 5
                                
                                layout_options = ["Auto-detect", "2 columns", "3 columns", "4 columns", "5 columns"]
                                default_index = 0  # Auto-detect is default
                                
                                layout_choice = st.radio(
                                    f"Layout for {group_name}:", 
                                    layout_options, 
                                    horizontal=True, 
                                    key=f"optimized_layout_radio_{group_name.replace(' ', '_')}",
                                    index=default_index
                                )
                                
                                if layout_choice == "Auto-detect":
                                    cols_per_row = min(5, max(2, (num_gyms_in_group + 1) // 2))
                                elif layout_choice == "2 columns":
                                    cols_per_row = 2
                                elif layout_choice == "3 columns":
                                    cols_per_row = 3
                                elif layout_choice == "4 columns":
                                    cols_per_row = 4
                                else:  # 5 columns
                                    cols_per_row = 5
                                    
                                # Store choice in session state for this group
                                if 'optimized_layout_preference' not in st.session_state:
                                    st.session_state.optimized_layout_preference = {}
                                st.session_state.optimized_layout_preference[group_name] = cols_per_row
                            else:
                                # Use stored preference for this group if available
                                if group_name in st.session_state.optimized_layout_preference:
                                    cols_per_row = st.session_state.optimized_layout_preference[group_name]
                                else:
                                    # Default layout if no preference is stored
                                    cols_per_row = min(5, max(2, (num_gyms_in_group + 1) // 2))
                        else:
                            # For first run or single group, use auto-detection
                            # Choose more columns on wider screens
                            cols_per_row = min(5, max(2, (num_gyms_in_group + 1) // 2))
                        
                        cols = st.columns(cols_per_row)
                        
                        for i, (gym, boulders, _, boulder_count) in enumerate(gyms):
                            col_idx = i % cols_per_row
                            with cols[col_idx]:
                                is_new_gym_flag = gym in unvisited_gyms
                                gym_label = f"{gym} 🆕" if is_new_gym_flag else gym
                                
                                with st.expander(f"{gym_label} - {boulder_count} boulders", expanded=True):
                                    if gym in gym_visit_recommendations:
                                        st.info(f"📍 **Visit this gym** - Recommended with {gym_visit_recommendations[gym]*100:.1f}% priority")
                                    
                                    if boulders:
                                        sorted_boulders = sorted(boulders, key=lambda x: x[1], reverse=True)
                                        boulder_rows = []
                                        for boulder, score in sorted_boulders:
                                            boulder_completion_count = gym_boulder_counts.get(gym, {}).get(boulder, 0)
                                            total_climbers_at_gym = participation_counts.get(gym, 0)
                                            success_rate = boulder_completion_count / total_climbers_at_gym * 100 if total_climbers_at_gym > 0 else 0
                                            
                                            boulder_row = {
                                                "Boulder": boulder, 
                                                "Probability": f"{score*100:.1f}%",
                                                "Success Rate": f"{success_rate:.1f}%"
                                            }
                                            
                                            # Add French Grade if grading system is available
                                            if processed_data and hasattr(processed_data, 'grading_system') and processed_data.grading_system:
                                                french_grade = get_boulder_french_grade_for_display(processed_data.grading_system, gym, str(boulder))
                                                boulder_row["French Grade"] = french_grade
                                            
                                            boulder_rows.append(boulder_row)
                                        
                                        boulder_df = pd.DataFrame(boulder_rows)
                                        
                                        # Configure column config based on available columns
                                        column_config = {
                                            "Boulder": st.column_config.TextColumn("Boulder", help="Boulder to complete"),
                                            "Probability": st.column_config.TextColumn(
                                                "Completion Probability", 
                                                help="Personalized probability of completion based on Bayesian analysis"
                                            ),
                                            "Success Rate": st.column_config.TextColumn(
                                                "Historical Success", 
                                                help="Percentage of all climbers who completed this boulder"
                                            )
                                        }
                                        
                                        if "French Grade" in boulder_df.columns:
                                            column_config["French Grade"] = st.column_config.TextColumn(
                                                "Est. Grade",
                                                help="Estimated French boulder grade"
                                            )
                                        
                                        st.dataframe(
                                            boulder_df,
                                            use_container_width=True,
                                            hide_index=True,
                                            column_config=column_config
                                        )
                                    else:
                                        st.write("No specific boulder recommendations for this gym.")
                            
                            total_boulders_in_path += boulder_count
                    
                    # Add summary message based on boulders needed vs provided
                    boulders_needed_for_goal = max(0, top_10_target - boulders_completed)
                    if boulders_needed_for_goal > 0:
                        if total_boulders_in_path >= boulders_needed_for_goal:
                            st.success(f"✅ Follow this path with {total_boulders_in_path} boulders to reach Top {target_rank}!")
                        else:
                             st.warning(f"⚠️ This path shows your best options ({total_boulders_in_path} boulders) but may not be sufficient to reach the Top {target_rank} goal ({boulders_needed_for_goal} more needed).")
                    else:
                         st.success(f"🎉 You have already achieved the Top {target_rank} goal!")
                else:
                    st.info("No specific recommendations found for creating an optimized path.")
        else:
             st.info(f"No optimized path to Top {target_rank} could be generated based on the available data.")
    
    # Fallback message if no actual recommendations were generated initially
    elif not recommendations:
        st.warning("No boulder recommendations available. This may occur if no similar climbers have completed boulders you haven't tried.")
    else: # Handles case where rec_df was empty or only had 'No specific recommendations'
        st.info(f"No specific boulder recommendations could be generated for {st.session_state[config.SESSION_KEY_SELECTED_CLIMBER]} at this time.")

    # --- Debug Info (Optional) --- REMOVED
    # Debug information has been removed from the Plan tab for production use
