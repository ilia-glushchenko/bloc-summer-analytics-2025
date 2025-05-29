# User Isolation in BLOC Summer Sessions 2025 Analysis Dashboard

## Overview

This document explains the user isolation design implemented in the Streamlit application to ensure that each user's session is completely independent and isolated from other users' sessions.

## Key Principles

### 1. Session State Isolation
- **All user-specific state is stored in `st.session_state`**
- Each user session gets a unique session ID (`st.session_state[SESSION_KEY_SESSION_ID]`)
- Session state is automatically isolated by Streamlit per browser session
- No global variables are used for user-specific data

### 2. Proper Caching Strategy
- **Data caching is safe and shared appropriately:**
  - `process_data(gender)` - Cached by gender parameter, safe to share since data is identical for all users within each gender category
  - `create_climber_boulder_matrix(data)` - Cached by data parameter, safe to share since it's deterministic
- **Cache clearing operations are avoided** to prevent affecting other users
- When gender selection changes, we rely on cache parameter differentiation rather than global cache clearing

### 3. Session-Specific Randomness
- Random seeds are set per session using the unique session ID
- Each user gets reproducible but isolated random sequences
- Prevents cross-user interference in any random operations

### 4. Consistent Session Key Management
- All session state keys are defined as constants in `config.py`
- Keys are used consistently across all tabs and components
- Clear naming convention prevents accidental key conflicts

## Session State Keys

The following session state keys are used for user isolation:

```python
SESSION_KEY_SESSION_ID = 'session_id'  # Unique identifier for each user session
SESSION_KEY_SELECTED_GENDER = 'selected_gender'  # User's selected gender category
SESSION_KEY_SELECTED_CLIMBER = 'selected_climber'  # User's selected climber
SESSION_KEY_SELECTED_CLIMBER_PATH_TAB = 'path_climber_selector_tab3'  # Path tab specific climber
SESSION_KEY_SELECTED_GYM_STATS_TAB = 'selected_gym_tab1'  # Gym stats tab selection
SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB = 'current_page'  # Climber rankings pagination
SESSION_KEY_TARGET_RANK = 'target_rank'  # User's target rank selection
SESSION_KEY_ACTIVE_TAB = 'active_tab'  # Currently active tab
```

## Implementation Details

### Session Initialization
```python
# Each user session gets a unique ID
if config.SESSION_KEY_SESSION_ID not in st.session_state:
    st.session_state[config.SESSION_KEY_SESSION_ID] = str(uuid.uuid4())
    
    # Set session-specific random seed
    session_seed = hash(st.session_state[config.SESSION_KEY_SESSION_ID]) % (2**32)
    random.seed(session_seed)
    np.random.seed(session_seed)
```

### Cache Management
```python
# Safe caching - different cache entries for different parameters
@st.cache_data(ttl=600)
def process_data(gender: str = 'men') -> ProcessedData:
    # This creates separate cache entries for 'men' and 'women'
    # Safe to share since data is identical for all users within each gender
```

### State Synchronization
```python
# Proper state synchronization between tabs
def sync_selected_climber() -> None:
    if config.SESSION_KEY_SELECTED_CLIMBER_PATH_TAB in st.session_state:
        st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] = st.session_state[config.SESSION_KEY_SELECTED_CLIMBER_PATH_TAB]
```

## Debugging User Isolation

A debug function is available to help troubleshoot user isolation issues:

```python
def display_session_debug_info() -> None:
    """Display session debugging information when debug mode is enabled."""
```

To enable debug mode, set the environment variable:
```bash
DEBUG_MODE=true
```

## Testing User Isolation

To test user isolation:

1. Open the application in multiple browser tabs or different browsers
2. Select different options in each tab (gender, climber, target rank, etc.)
3. Verify that changes in one tab don't affect the other tabs
4. Enable debug mode to see session IDs and verify they are different
5. Check that pagination, selections, and other state remain independent

## Common Pitfalls Avoided

1. **Global cache clearing** - We avoid `st.cache_data.clear()` which would affect all users
2. **Module-level variables** - No user-specific data is stored in module-level variables
3. **Shared mutable objects** - All cached objects are either immutable or safely copied
4. **Global random state** - Random seeds are session-specific, not global

## Compliance with Streamlit Best Practices

This implementation follows Streamlit's recommended practices for multi-user applications:

- Session state is used for user-specific data
- Caching is used appropriately for shared, deterministic data
- No global state is used for user-specific information
- Each session maintains its own isolated state

## Monitoring and Maintenance

- Session state keys are centrally managed in `config.py`
- Debug information is available for troubleshooting
- Clear documentation of isolation principles
- Regular testing procedures for user isolation verification 