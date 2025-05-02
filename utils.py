import streamlit as st
import os
from dotenv import load_dotenv
from typing import Optional

import config # Import config

# Helper function to load CSS
def load_css(file_name: str) -> None:
    """Loads CSS from a file into the Streamlit app."""
    try:
        with open(file_name, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

def get_debug_mode() -> bool:
    """
    Get the debug mode setting from environment variables.
    Returns True if DEBUG_MODE is set to 'true', otherwise False.
    """
    load_dotenv()  # Load environment variables from .env file
    # Use constant for default value
    debug_mode = os.getenv('DEBUG_MODE', config.DEFAULT_DEBUG_MODE).lower() == 'true'
    return debug_mode 

def get_default_selected_climber() -> Optional[str]:
    """
    Get the default selected climber from environment variables.
    Returns the default selected climber, or None if not set.
    """
    load_dotenv()  # Load environment variables from .env file
    # Use constant for default value (which is None)
    return os.getenv('DEFAULT_SELECTED_CLIMBER', config.DEFAULT_SELECTED_CLIMBER)