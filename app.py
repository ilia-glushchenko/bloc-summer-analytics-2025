"""
BLOC Summer Sessions 2025 Analysis Dashboard - Main Application File

This file sets up the Streamlit application structure, loads data,
and delegates the rendering of each tab to specific modules.

Usage: streamlit run app.py
"""

#-----------------------------------------------------------------------------
# IMPORTS AND DEPENDENCIES
#-----------------------------------------------------------------------------
import streamlit as st
import traceback
import numpy as np # Keep numpy for seed setting
import random # Keep random for seed setting

# Local modules
import config # Import the config file
from utils import load_css
from data_processing import process_data, ProcessedData # Import ProcessedData
from tabs.gym_stats_tab import display_gym_stats
from tabs.climber_stats_tab import display_climber_stats
from tabs.path_success_tab import display_path_success, DEFAULT_CLIMBER

# Silence warnings (especially from scikit-learn) - Keep if needed by submodules
# TODO (Maintainability): Consider identifying and suppressing only specific warnings
# instead of globally ignoring all. This prevents hiding potentially useful new warnings.
# warnings.filterwarnings('ignore') # Removed global suppression

# Set global random seed for reproducibility
# TODO (Maintainability): Consider moving magic numbers like this seed to a config section/file.
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

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
# MAIN APPLICATION LOGIC
#-----------------------------------------------------------------------------
def main() -> None:
    """Main function to run the Streamlit application."""
    try:
        # Wrap data loading and processing in a spinner for better UX
        with st.spinner("Crunching the numbers... Please wait."):
            # Load and process data using the dedicated function
            # This now returns a single ProcessedData object
            processed_data: ProcessedData = process_data()

        # Modern compact app header
        st.markdown("""
        <div class="app-header">
            <h1>BLOC Summer Sessions 2025 Analysis</h1>
        </div>
        """, unsafe_allow_html=True)

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🏢 Gym Stats", "📊 Climbers", "🛣️ Path to Success"])

        # Initialize session state for selected climber (if needed globally)
        # It's also handled within path_success_tab, but can be initialized here too.
        if config.SESSION_KEY_SELECTED_CLIMBER not in st.session_state:
            # Use the DEFAULT_CLIMBER from path_success_tab
            st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] = DEFAULT_CLIMBER
            
        # Add callback for handling climber selection changes
        def sync_selected_climber() -> None:
            """Ensures selected climber is synchronized across tabs"""
            if config.SESSION_KEY_SELECTED_CLIMBER_PATH_TAB in st.session_state:
                st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] = st.session_state[config.SESSION_KEY_SELECTED_CLIMBER_PATH_TAB]

        # Delegate rendering to tab-specific functions
        with tab1:
            # Pass only needed data from the ProcessedData object
            display_gym_stats(
                processed_data.raw_data, 
                processed_data.gym_boulder_counts, 
                processed_data.participation_counts
            )

        with tab2:
            # Pass the specific data required by this tab
            display_climber_stats(
                processed_data.raw_data, 
                processed_data.climbers_df, 
                processed_data.completion_histograms, 
                processed_data.outlier_warning_message
            )

        with tab3:
            # Pass the specific data required by the path to success tab
            display_path_success(
                processed_data.raw_data, 
                processed_data.climbers_df, 
                processed_data.gym_boulder_counts, 
                processed_data.participation_counts, 
                sync_selected_climber # Callback remains the same
            )

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