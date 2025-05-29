#!/usr/bin/env python3
"""
Test script for verifying user isolation in the BLOC Summer Sessions 2025 Analysis Dashboard.

This script can be used to test that user sessions are properly isolated.

Usage:
1. Run the main Streamlit app: streamlit run app.py
2. Open multiple browser tabs/windows to the app
3. Use this script to verify isolation by checking different behaviors
4. Enable DEBUG_MODE=true in .env to see session debug information

Test Cases:
- Different gender selections should not affect other users
- Different climber selections should remain independent
- Pagination state should be per-user
- Target rank selections should be isolated
- Random operations should be reproducible per session but different between sessions
"""

import streamlit as st
import uuid
import time
from typing import Dict, Any

def test_session_isolation():
    """
    Simple test function to verify session isolation.
    This can be run in multiple browser tabs to verify independence.
    """
    st.title("🧪 User Isolation Test")
    
    # Initialize session if not exists
    if 'test_session_id' not in st.session_state:
        st.session_state.test_session_id = str(uuid.uuid4())
        st.session_state.test_counter = 0
        st.session_state.test_selections = {}
    
    # Display session info
    st.write("**Your Session ID:**", st.session_state.test_session_id[:8] + "...")
    st.write("**Session Counter:**", st.session_state.test_counter)
    
    # Test counter
    if st.button("Increment Counter"):
        st.session_state.test_counter += 1
        st.rerun()
    
    # Test selections
    st.subheader("Test Selections")
    
    test_gender = st.selectbox("Test Gender", ["Men", "Women"], key="test_gender")
    test_climber = st.text_input("Test Climber Name", key="test_climber")
    test_number = st.number_input("Test Number", min_value=1, max_value=100, key="test_number")
    
    # Store selections
    st.session_state.test_selections = {
        "gender": test_gender,
        "climber": test_climber,
        "number": test_number,
        "timestamp": time.time()
    }
    
    # Display all session state
    st.subheader("Current Session State")
    st.json(dict(st.session_state))
    
    # Instructions
    st.subheader("Testing Instructions")
    st.markdown("""
    1. **Open this test in multiple browser tabs**
    2. **Verify each tab has a different Session ID**
    3. **Make different selections in each tab**
    4. **Increment counters independently**
    5. **Verify that changes in one tab don't affect others**
    6. **Check that session state remains isolated**
    """)
    
    # Warning about proper testing
    st.warning("""
    **Important:** To properly test user isolation:
    - Use different browser tabs or windows
    - Try different browsers entirely
    - Test with incognito/private browsing modes
    - Verify that each session maintains its own state
    """)

if __name__ == "__main__":
    test_session_isolation() 