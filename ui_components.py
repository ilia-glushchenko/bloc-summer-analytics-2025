import streamlit as st
from typing import Dict, Any, Optional

#-----------------------------------------------------------------------------
# UI HELPER FUNCTIONS
#-----------------------------------------------------------------------------
def render_metrics_row(metrics: Dict[str, Any]):
    """Renders a consistent row of metrics using st.columns and st.metric."""
    cols = st.columns(len(metrics))
    i = 0
    for label, value in metrics.items():
        with cols[i]:
            st.metric(label=label, value=str(value))
        i += 1

def render_section_header(title: str, level: int = 4, css_class: Optional[str] = None):
    """Renders a standardized section header with an optional CSS class."""
    class_attr = f" class='{css_class}'" if css_class else ""
    st.markdown(f"<h{level}{class_attr}>{title}</h{level}>", unsafe_allow_html=True) 