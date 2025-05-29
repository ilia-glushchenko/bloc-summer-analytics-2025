"""
Configuration and Constants for the Boulder Summer Series Analysis App
"""

# --- General App Settings ---
RANDOM_SEED = 42
CSS_FILE = "style.css"

# --- Session State Keys ---
# Used to store user selections and states across reruns
SESSION_KEY_SESSION_ID = 'session_id'  # Unique identifier for each user session
SESSION_KEY_SELECTED_CLIMBER = 'selected_climber'
SESSION_KEY_SELECTED_CLIMBER_PATH_TAB = 'path_climber_selector_tab3' # Specific key for the path tab selector
SESSION_KEY_SELECTED_GYM_STATS_TAB = 'selected_gym_tab1'
SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB = 'current_page' # Used for climber rankings pagination
SESSION_KEY_TARGET_RANK = 'target_rank' # Key for storing the user-selected target rank
SESSION_KEY_ACTIVE_TAB = 'active_tab' # Key for storing the currently active tab
SESSION_KEY_SELECTED_GENDER = 'selected_gender' # Key for storing the selected gender (men/women)

# --- Environment Variable Defaults ---
# Used in utils.py when reading .env
DEFAULT_DEBUG_MODE = 'false'
DEFAULT_SELECTED_CLIMBER = None

# --- Path to Success Tab Configuration ---
PATH_TARGET_RANK = 10 # Default target rank for path analysis
PATH_TARGET_RANK_MIN = 1 # Minimum allowed target rank
PATH_SIMILARITY_N = 5 # Number of similar climbers to find
# DEFAULT_CLIMBER is now set via environment variable, read by utils.get_default_selected_climber()

# --- Data Processing Configuration ---
# Example: If file paths were hardcoded, they could go here
# RESULTS_FILE = "results.json" # Example - currently handled in stats.py

# --- UI Configuration ---
RANKINGS_TABLE_ITEMS_PER_PAGE = 50 