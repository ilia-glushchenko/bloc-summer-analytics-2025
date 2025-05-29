"""
BLOC Summer Sessions 2025 Analysis Dashboard - Main Application File

This file sets up the Streamlit application structure, loads data,
and delegates the rendering of each tab to specific modules.

USER ISOLATION DESIGN:
- Each user session gets a unique session ID stored in st.session_state
- All user-specific state is stored in st.session_state with proper session keys
- Cached data (st.cache_data) is shared appropriately:
  * process_data() is cached by gender parameter - safe to share since data is identical for all users within each gender
  * create_climber_boulder_matrix() is cached by data parameter - safe to share since it's deterministic
- Random seeds are session-specific to ensure reproducible but isolated randomness per user
- No global variables are used for user-specific state
- Cache clearing operations are avoided to prevent affecting other users

Usage: streamlit run app.py
"""

#-----------------------------------------------------------------------------
# IMPORTS AND DEPENDENCIES
#-----------------------------------------------------------------------------
import streamlit as st
import traceback
import numpy as np # Keep numpy for seed setting
import random # Keep random for seed setting
import json # Added for loading metadata
import uuid

# Local modules
import config # Import the config file
from utils import load_css, get_debug_mode
from data_processing import process_data, process_combined_division, ProcessedData # Import ProcessedData and new function
from tabs.gym_stats_tab import display_gym_stats
from tabs.climber_stats_tab import display_climber_stats
from tabs.path_success_tab import display_path_success, DEFAULT_CLIMBER

# Silence warnings (especially from scikit-learn) - Keep if needed by submodules
# TODO (Maintainability): Consider identifying and suppressing only specific warnings
# instead of globally ignoring all. This prevents hiding potentially useful new warnings.
# warnings.filterwarnings('ignore') # Removed global suppression

#-----------------------------------------------------------------------------
# APP CONFIGURATION
#-----------------------------------------------------------------------------
# Set page configuration
st.set_page_config(
    page_title="BLOC Summer Sessions 2025 Analysis",
    page_icon="🧗‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
    # TODO (Compliance): Add menu items for Privacy Policy and Terms of Service
    # menu_items={
    #     'About': "# This is an *extremely* cool app!",
    #     'Get Help': None, # Placeholder for help link/info
    #     'Report a Bug': None, # Placeholder for bug reporting
    #     # 'Privacy Policy': 'URL_TO_YOUR_PRIVACY_POLICY', 
    #     # 'Terms of Service': 'URL_TO_YOUR_TERMS_OF_SERVICE'
    # }
)

# Load custom CSS
load_css(config.CSS_FILE)

# TODO (Compliance): Add a footer or section with contact information for GDPR/DSR requests
# Example using st.sidebar or bottom of the main page:
# st.sidebar.markdown("---")
# st.sidebar.markdown("**Contact:** [Your Contact Email/Link]")
# st.sidebar.markdown("[Privacy Policy](URL_TO_YOUR_PRIVACY_POLICY)")
# st.sidebar.markdown("[Terms of Service](URL_TO_YOUR_TERMS_OF_SERVICE)")

#-----------------------------------------------------------------------------
# DEBUGGING AND UTILITIES
#-----------------------------------------------------------------------------
def display_session_debug_info() -> None:
    """
    Display session debugging information. 
    Useful for troubleshooting user isolation issues.
    Only shown when debug mode is enabled.
    """
    from utils import get_debug_mode
    
    if get_debug_mode():
        with st.expander("🔧 Session Debug Info", expanded=False):
            st.write("**Session ID:**", st.session_state.get(config.SESSION_KEY_SESSION_ID, "Not set"))
            st.write("**Selected Gender:**", st.session_state.get(config.SESSION_KEY_SELECTED_GENDER, "Not set"))
            st.write("**Selected Climber:**", st.session_state.get(config.SESSION_KEY_SELECTED_CLIMBER, "Not set"))
            st.write("**Active Tab:**", st.session_state.get(config.SESSION_KEY_ACTIVE_TAB, "Not set"))
            st.write("**All Session State Keys:**", list(st.session_state.keys()))

#-----------------------------------------------------------------------------
# MAIN APPLICATION LOGIC
#-----------------------------------------------------------------------------
def main() -> None:
    """Main function to run the Streamlit application."""
    try:
        # Initialize unique session ID for this user session if not already set
        if config.SESSION_KEY_SESSION_ID not in st.session_state:
            st.session_state[config.SESSION_KEY_SESSION_ID] = str(uuid.uuid4())
            
            # Set session-specific random seed for reproducibility within this session
            # This ensures each user session has its own random sequence
            session_seed = hash(st.session_state[config.SESSION_KEY_SESSION_ID]) % (2**32)
            random.seed(session_seed)
            np.random.seed(session_seed)
        
        # Global Gender Selector - Initialize session state first
        if config.SESSION_KEY_SELECTED_GENDER not in st.session_state:
            st.session_state[config.SESSION_KEY_SELECTED_GENDER] = 'men'
        
        # Get current gender for header (ensure lowercase)
        current_gender_for_header = st.session_state.get(config.SESSION_KEY_SELECTED_GENDER, 'men')
        if current_gender_for_header not in ['men', 'women', 'combined']:
            current_gender_for_header = 'men'
        
        # Create appropriate division display name
        if current_gender_for_header == 'combined':
            gender_display = "Combined Division"
        else:
            gender_display = current_gender_for_header.title() + " Division"
        
        # Header with integrated division name and right-aligned selector
        # Use smaller column ratio to force dropdown constraint
        col1, col2 = st.columns([4, 1], gap="medium")
        
        with col1:
            st.markdown(f"""
            <div class="app-header">
                <h1>BLOC Summer Sessions 2025 Analysis - {gender_display}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Simpler approach - just use the column constraint
            st.markdown("""
            <style>
            .header-dropdown-container {
                padding: 0.5rem 0.75rem !important;
                margin: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: flex-end !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="header-dropdown-container">', unsafe_allow_html=True)
            selected_gender = st.selectbox(
                "Select Category:",
                options=['Men', 'Women', 'Combined'],
                index={'men': 0, 'women': 1, 'combined': 2}.get(st.session_state[config.SESSION_KEY_SELECTED_GENDER], 0),
                key='gender_selector',
                help="Switch between male, female, and combined competition results",
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Update session state when selection changes
            # Convert back to lowercase for internal use
            selected_gender_internal = selected_gender.lower()
            if selected_gender_internal != st.session_state[config.SESSION_KEY_SELECTED_GENDER]:
                st.session_state[config.SESSION_KEY_SELECTED_GENDER] = selected_gender_internal
                # Instead of clearing all cache globally, we let the cached function handle 
                # different gender parameters naturally - this ensures user isolation
                # The cache will automatically use different cache entries for different genders
                st.rerun()
        
        # Use the selected gender for data processing (ensure it's always lowercase)
        current_gender = st.session_state[config.SESSION_KEY_SELECTED_GENDER]
        # Safety check to ensure it's valid
        if current_gender not in ['men', 'women', 'combined']:
            current_gender = 'men'  # fallback to default
        
        with st.spinner("Crunching the numbers... Please wait."):
            # Load and process data using the appropriate function
            if current_gender == 'combined':
                processed_data: ProcessedData = process_combined_division()
            else:
                processed_data: ProcessedData = process_data(gender=current_gender)

        # Initialize active tab in session state if it doesn't exist
        if config.SESSION_KEY_ACTIVE_TAB not in st.session_state:
            st.session_state[config.SESSION_KEY_ACTIVE_TAB] = 0

        # Initialize session state for selected climber (if needed globally)
        if config.SESSION_KEY_SELECTED_CLIMBER not in st.session_state:
            # Use the DEFAULT_CLIMBER from path_success_tab
            st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] = DEFAULT_CLIMBER
        
        # Add callback for handling climber selection changes
        def sync_selected_climber() -> None:
            """Ensures selected climber is synchronized across tabs"""
            if config.SESSION_KEY_SELECTED_CLIMBER_PATH_TAB in st.session_state:
                st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] = st.session_state[config.SESSION_KEY_SELECTED_CLIMBER_PATH_TAB]

        # Create tabs (this needs to be a simple call to set up the UI structure)
        tab1, tab2, tab3 = st.tabs(["🧗‍♀️ Gyms", "🏆 Rankings", "🛣️ Plan"])
        
        # When a user clicks on a tab, update the active tab in session state
        # This doesn't happen automatically, but we can detect which tab to show
        # based on which one has content changes
                
        # Render content in each tab
        with tab1:
            # Update active tab when this tab is viewed
            if st.session_state[config.SESSION_KEY_ACTIVE_TAB] != 0:
                st.session_state[config.SESSION_KEY_ACTIVE_TAB] = 0
            
            # Display content
            display_gym_stats(
                processed_data.raw_data, 
                processed_data.gym_boulder_counts, 
                processed_data.participation_counts,
                processed_data.completion_histograms,
                processed_data.outlier_warning_message
            )

        with tab2:
            # Update active tab when this tab is viewed
            if st.session_state[config.SESSION_KEY_ACTIVE_TAB] != 1:
                st.session_state[config.SESSION_KEY_ACTIVE_TAB] = 1
                
            # Display content
            display_climber_stats(
                processed_data.raw_data, 
                processed_data.climbers_df, 
                processed_data.completion_histograms, 
                processed_data.outlier_warning_message
            )

        with tab3:
            # Update active tab when this tab is viewed
            if st.session_state[config.SESSION_KEY_ACTIVE_TAB] != 2:
                st.session_state[config.SESSION_KEY_ACTIVE_TAB] = 2
                
            # Display content
            display_path_success(
                processed_data.raw_data, 
                processed_data.climbers_df, 
                processed_data.gym_boulder_counts, 
                processed_data.participation_counts, 
                sync_selected_climber
            )

        # Display debug information if debug mode is enabled
        display_session_debug_info()

    except Exception as e:
        # User-friendly error message
        st.error("😔 Apologies, an unexpected error occurred while running the application.")
        st.info("Details have been logged for the development team. Please try refreshing the page or contact support if the issue persists.")
        # Log the full error for debugging (visible in the console where Streamlit runs)
        print("------------------- ERROR LOG -------------------")
        print(f"Error in main application: {str(e)}")
        print(traceback.format_exc())
        print("-----------------------------------------------")

if __name__ == "__main__":
    main()

# Run with: streamlit run app.py 