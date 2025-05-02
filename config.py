"""
Configuration and Constants for the Boulder Summer Series Analysis App
"""

# --- General App Settings ---
RANDOM_SEED = 42
CSS_FILE = "style.css"

# --- Session State Keys ---
# Used to store user selections and states across reruns
SESSION_KEY_SELECTED_CLIMBER = 'selected_climber'
SESSION_KEY_SELECTED_CLIMBER_PATH_TAB = 'path_climber_selector_tab3' # Specific key for the path tab selector
SESSION_KEY_SELECTED_GYM_STATS_TAB = 'selected_gym_tab1'
SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB = 'current_page' # Used for climber rankings pagination

# --- Environment Variable Defaults ---
# Used in utils.py when reading .env
DEFAULT_DEBUG_MODE = 'false'
DEFAULT_SELECTED_CLIMBER = None

# --- Path to Success Tab Configuration ---
PATH_TARGET_RANK = 10 # Target rank for path analysis
PATH_SIMILARITY_N = 5 # Number of similar climbers to find
# DEFAULT_CLIMBER is now set via environment variable, read by utils.get_default_selected_climber()

# --- Data Processing Configuration ---
# Example: If file paths were hardcoded, they could go here
# RESULTS_FILE = "results.json" # Example - currently handled in stats.py

# --- UI Configuration ---
RANKINGS_TABLE_ITEMS_PER_PAGE = 50 