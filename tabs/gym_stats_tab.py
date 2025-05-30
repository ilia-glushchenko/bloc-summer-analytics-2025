import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json
import html
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import norm

import config
from data_processing import create_climber_boulder_matrix
from ui_components import render_metrics_row, render_section_header

# Try to import grading system
try:
    from grading_system import FrenchGradingSystem
    GRADING_SYSTEM_AVAILABLE = True
except ImportError:
    GRADING_SYSTEM_AVAILABLE = False

#------------------------------------------------------------------------------
# HELPER FUNCTIONS FOR GRADING
#------------------------------------------------------------------------------

def get_boulder_french_grade(grading_system, gym_name: str, boulder_id: str) -> str:
    """
    Get the estimated French grade for a boulder.
    
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

#------------------------------------------------------------------------------
# HELPER FUNCTIONS FOR CHART GENERATION
#------------------------------------------------------------------------------

def create_radar_chart(
    values: List[float], 
    labels: List[str], 
    completed_values: Optional[List[float]] = None, 
    climber_name: Optional[str] = None
) -> go.Figure:
    """
    Creates a radar chart for boulder characteristics.
    
    Args:
        values: List of total counts for each category
        labels: List of category labels
        completed_values: List of completed counts for the selected climber
        climber_name: Name of the selected climber
        
    Returns:
        A plotly figure object with the radar chart
    """
    # Make the radar chart close by repeating the first point
    if labels:
        labels_closed = labels + [labels[0]]
        values_closed = values + [values[0]]
        completed_values_closed = completed_values + [completed_values[0]] if completed_values else None
    else:
        labels_closed = labels
        values_closed = values
        completed_values_closed = completed_values
    
    # Add radar chart
    fig = go.Figure()
    
    # Add total boulders trace
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill='toself',
        name='Total Boulders',
        fillcolor='rgba(65, 105, 225, 0.2)',
        line=dict(color='royalblue')
    ))
    
    # Add climber's completed boulders trace
    if completed_values_closed and any(completed_values) and climber_name:
        fig.add_trace(go.Scatterpolar(
            r=completed_values_closed,
            theta=labels_closed,
            fill='toself',
            name=f'Completed by {climber_name}',
            fillcolor='rgba(255, 124, 36, 0.2)',
            line=dict(color='#ff7c24')
        ))
    
    # Configure chart layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) + 1]
            )),
        showlegend=False,
        height=450,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

def create_data_table(
    df: pd.DataFrame, 
    col_name: str, 
    count_label: str = "Total Boulders"
) -> None:
    """
    Creates and displays a data table for boulder characteristics.
    
    Args:
        df: DataFrame containing the data
        col_name: Name of the primary column (Category, Wall Angle, etc.)
        count_label: Label for the count column
    """
    column_config = {
        col_name: st.column_config.TextColumn(col_name),
        "Total Boulders": st.column_config.NumberColumn(count_label, format="%d"),
    }
    
    # Add completion columns if present
    if "Completed" in df.columns:
        column_config["Completed"] = st.column_config.NumberColumn("Completed", format="%d")
    
    if "Completion Rate (%)" in df.columns:
        column_config["Completion Rate (%)"] = st.column_config.NumberColumn(
            "Completion Rate", format="%.1f%%"
        )
    
    st.dataframe(df, hide_index=True, column_config=column_config)

def extract_boulder_data(
    gym_boulders: List[Dict], 
    selected_climber_boulders_ids: List[int], 
    tag_key: str
) -> Dict[str, Dict[str, int]]:
    """
    Extracts data for a specific boulder characteristic.
    
    Args:
        gym_boulders: List of boulder data for the selected gym
        selected_climber_boulders_ids: List of boulder IDs completed by the selected climber
        tag_key: The tag key to extract (category, angle, holds, techniques)
        
    Returns:
        Dictionary mapping each tag value to its total and completed counts
    """
    result = {}
    
    # Count total occurrences for each tag value
    for boulder in gym_boulders:
        tag_value = boulder["tags"][tag_key]
        
        # Handle array tags (holds, techniques) differently from scalar tags (category, angle)
        if isinstance(tag_value, list):
            for value in tag_value:
                if value:  # Only count non-empty values
                    if value not in result:
                        result[value] = {"total": 0, "completed": 0}
                    result[value]["total"] += 1
        elif tag_value:  # Handle scalar tags and only count non-empty values
            if tag_value not in result:
                result[tag_value] = {"total": 0, "completed": 0}
            result[tag_value]["total"] += 1
    
    # Count completed occurrences for the selected climber
    for boulder in gym_boulders:
        if boulder["id"] in selected_climber_boulders_ids:
            tag_value = boulder["tags"][tag_key]
            
            if isinstance(tag_value, list):
                for value in tag_value:
                    if value and value in result:
                        result[value]["completed"] += 1
            elif tag_value and tag_value in result:
                result[tag_value]["completed"] += 1
    
    return result

def prepare_characteristic_data(
    data_dict: Dict[str, Dict[str, int]], 
    sort_by_count: bool = False
) -> Tuple[List[str], List[int], List[int]]:
    """
    Prepares data from a characteristic dictionary for visualization.
    
    Args:
        data_dict: Dictionary mapping characteristic values to count data
        sort_by_count: Whether to sort the data by count (highest to lowest)
        
    Returns:
        Tuple of (names, total_counts, completed_counts)
    """
    names = list(data_dict.keys())
    total_counts = [data_dict[name]["total"] for name in names]
    completed_counts = [data_dict[name]["completed"] for name in names]
    
    # Sort by count if requested
    if sort_by_count and names:
        sorted_indices = sorted(range(len(total_counts)), key=lambda i: total_counts[i], reverse=True)
        names = [names[i] for i in sorted_indices]
        total_counts = [total_counts[i] for i in sorted_indices]
        completed_counts = [completed_counts[i] for i in sorted_indices]
    
    return names, total_counts, completed_counts

def generate_characteristic_tab(
    tab: Any, 
    data_dict: Dict[str, Dict[str, int]], 
    col_name: str, 
    climber_name: Optional[str] = None,
    sort_by_count: bool = False,
    no_data_message: str = "No data available"
) -> None:
    """
    Generates a tab with a radar chart and data table for a boulder characteristic.
    
    Args:
        tab: Streamlit tab object to render the content in
        data_dict: Dictionary mapping characteristic values to count data
        col_name: Name of the primary column (Category, Wall Angle, etc.)
        climber_name: Name of the selected climber
        sort_by_count: Whether to sort the data by count (highest to lowest)
        no_data_message: Message to display when no data is available
    """
    if not data_dict:
        tab.info(no_data_message)
        return
        
    # Prepare data
    names, total_counts, completed_counts = prepare_characteristic_data(data_dict, sort_by_count)
    
    # Create columns for layout
    col1, col2 = st.columns([0.4, 0.6])
    
    # Create DataFrame for the table
    df = pd.DataFrame({
        col_name: names,
        'Total Boulders': total_counts
    })
    
    if climber_name:
        df['Completed'] = completed_counts
        df['Completion Rate (%)'] = [(c/t*100 if t > 0 else 0) for c, t in zip(completed_counts, total_counts)]
    
    # Place table in left column
    with col1:
        st.write("""<div style="display: flex; justify-content: center; width: 100%;">""", unsafe_allow_html=True)
        create_data_table(df, col_name)
        st.write("</div>", unsafe_allow_html=True)
    
    # Place chart in right column
    with col2:
        fig = create_radar_chart(total_counts, names, completed_counts, climber_name)
        st.plotly_chart(fig, use_container_width=True)

#------------------------------------------------------------------------------
# MAIN DISPLAY FUNCTION
#------------------------------------------------------------------------------

def display_gym_stats(
    data: List[Dict], 
    gym_boulder_counts: Dict, 
    participation_counts: Dict, 
    completion_histograms: Dict = None, 
    outlier_warning_message: Optional[str] = None,
    processed_data: Optional[Any] = None
) -> None:
    """
    Display comprehensive gym statistics including gym overview, individual boulder 
    analysis, and completion distributions. Now includes French grading when available.
    
    Args:
        data: List of climber dictionaries from the competition
        gym_boulder_counts: Dictionary mapping gym names to boulder completion counts
        participation_counts: Dictionary mapping gym names to total participants
        completion_histograms: Dictionary mapping gym names to completion distribution data
        outlier_warning_message: Optional warning message about outliers in the data
        processed_data: Optional ProcessedData object containing grading system
    """
    st.markdown("### Climbing Gym Statistics")

    # Render top metrics
    if data and participation_counts and gym_boulder_counts:
        total_ascents_overall = sum(sum(counts.values()) for counts in gym_boulder_counts.values())
        render_metrics_row({
            "Total Climbers": len(data),
            "Total Gyms": len(participation_counts),
            "Total Ascents": total_ascents_overall
        })
    else:
        st.warning("Insufficient data to display overall metrics.")

    # Gym selection using button-like toggles
    if gym_boulder_counts:
        gym_names = sorted(gym_boulder_counts.keys())
        if not gym_names:
            st.warning("No gym data available to select.")
        else:
            st.markdown("""
            <style>
            div.gym-button-row {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 20px;
            }
            div.gym-button {
                padding: 8px 16px;
                border-radius: 4px;
                background-color: #f7fafc;
                border: 1px solid #e2e8f0;
                cursor: pointer;
                text-align: center;
                transition: all 0.2s;
            }
            div.gym-button.active {
                background-color: #5a67d8;
                color: white;
                border-color: #4c51bf;
                font-weight: 500;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Initialize session state for selected gym if not present
            if config.SESSION_KEY_SELECTED_GYM_STATS_TAB not in st.session_state:
                st.session_state[config.SESSION_KEY_SELECTED_GYM_STATS_TAB] = gym_names[0]  # Default to first gym
            
            st.markdown("<div class='gym-button-row'>", unsafe_allow_html=True)
            
            # Calculate how many gyms to show per row (max 4)
            gyms_per_row = min(4, len(gym_names))
            cols = st.columns(gyms_per_row)
            
            for i, gym_name in enumerate(gym_names):
                col_idx = i % gyms_per_row
                with cols[col_idx]:
                    is_active = st.session_state[config.SESSION_KEY_SELECTED_GYM_STATS_TAB] == gym_name
                    button_class = "gym-button active" if is_active else "gym-button"
                    
                    # Define callback for gym button
                    def select_gym(gym=gym_name):
                        st.session_state[config.SESSION_KEY_SELECTED_GYM_STATS_TAB] = gym
                    
                    st.button(
                        gym_name, 
                        key=f"gym_btn_{gym_name}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                        on_click=select_gym
                    )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            selected_gym = st.session_state[config.SESSION_KEY_SELECTED_GYM_STATS_TAB]

            #------------------------------------------------------------------
            # SELECTED GYM ANALYSIS
            #------------------------------------------------------------------
            if selected_gym:
                # Get data for the selected gym first
                boulder_counts = gym_boulder_counts.get(selected_gym, {})
                total_climbers_at_gym = participation_counts.get(selected_gym, 0)

                #--------------------------------------------------------------
                # BOULDER ANALYSIS SECTION
                #--------------------------------------------------------------
                if boulder_counts:
                    boulder_df = pd.DataFrame() # Initialize empty dataframe
                    try:
                        # Attempt to find numeric boulders first
                        numeric_boulders = [int(b) for b in boulder_counts.keys() if b.isdigit()]
                        if numeric_boulders:
                            max_boulder_num = max(numeric_boulders)
                        else:
                            # Fallback if no purely numeric boulders exist (handle potential strings)
                            max_boulder_num = 0
                    except ValueError:
                        st.warning(f"Could not determine max boulder number for {selected_gym}. Check boulder naming.")
                        max_boulder_num = 0

                    # Create boulder list including potential non-numeric ones
                    all_possible_boulders_in_gym = sorted(boulder_counts.keys()) # Use keys present
                    if max_boulder_num > 0:
                        # If numeric max found, generate range and combine (avoids missing boulders)
                         numeric_range = [str(i) for i in range(1, max_boulder_num + 1)]
                         combined_boulders = sorted(list(set(numeric_range + all_possible_boulders_in_gym)))
                    else:
                        # Use only the keys found if no numeric max
                        combined_boulders = all_possible_boulders_in_gym

                    # Create boulder data objects
                    complete_boulder_data = []
                    for boulder_num in combined_boulders:
                        complete_boulder_data.append({
                            'Boulder': boulder_num,
                            'Ascents': boulder_counts.get(boulder_num, 0)
                        })
                        
                    # Create DataFrame from boulder data
                    if complete_boulder_data: 
                        boulder_df = pd.DataFrame(complete_boulder_data)
                        # Apply sorting and calculations only if df is not empty
                        if not boulder_df.empty:
                            boulder_df = boulder_df.sort_values('Ascents', ascending=False)
                            total_ascents = boulder_df['Ascents'].sum()
                            boulder_df['Success Rate (%)'] = boulder_df['Ascents'].apply(
                                lambda x: round((x / total_climbers_at_gym * 100), 1) if total_climbers_at_gym > 0 else 0
                            )
                            if boulder_df['Success Rate (%)'].nunique() > 1:
                                boulder_df['Success_Percentile'] = boulder_df['Success Rate (%)'].rank(pct=True) * 100
                                boulder_df['Difficulty (1-10)'] = boulder_df['Success_Percentile'].apply(
                                    lambda p: 10 - int(p // 10) if p < 100 else 1
                                )
                            else:
                                boulder_df['Difficulty (1-10)'] = 5 # Default

                    # Display boulder analysis if we have data
                    if not boulder_df.empty:
                        boulder_df_display = boulder_df.copy()
                        render_section_header(f"Boulder Analysis: {selected_gym}", level=4)

                        # Get selected climber data
                        selected_climber_boulders = []
                        if config.SESSION_KEY_SELECTED_CLIMBER in st.session_state:
                            selected_climber = st.session_state[config.SESSION_KEY_SELECTED_CLIMBER]
                            _, climber_gym_boulders = create_climber_boulder_matrix(data)
                            
                            if selected_climber in climber_gym_boulders and selected_gym in climber_gym_boulders[selected_climber]:
                                selected_climber_boulders = climber_gym_boulders[selected_climber][selected_gym]

                        # Summary metrics info bar 
                        total_boulders_in_gym = len(boulder_df)
                        untouched_count = len(boulder_df[boulder_df['Ascents'] == 0])
                        total_ascents_in_gym = boulder_df['Ascents'].sum()

                        st.markdown(f"""
                        <div class=\"info-bar\">
                            <b>Total Boulders:</b> {html.escape(str(total_boulders_in_gym))} | <b>Total Ascents:</b> {html.escape(str(total_ascents_in_gym))} |
                            <b>Participants:</b> {html.escape(str(total_climbers_at_gym))} | <b>Untouched:</b> {html.escape(str(untouched_count))}
                        </div>
                        """, unsafe_allow_html=True)

                        # Boulder Popularity Chart section
                        render_section_header("Boulder Popularity", level=5)
                        
                        # Sort boulders for the chart
                        sorted_boulders_for_chart = boulder_df_display['Boulder'].tolist()
                        ascents_list = boulder_df_display['Ascents'].tolist()
                        climber_percentages = [round((ascents / total_climbers_at_gym * 100), 1) if total_climbers_at_gym > 0 else 0
                                              for ascents in ascents_list]
                        
                        # Get French grades for chart display
                        french_grades_for_chart = []
                        if processed_data and hasattr(processed_data, 'grading_system') and processed_data.grading_system:
                            for boulder in sorted_boulders_for_chart:
                                grade = get_boulder_french_grade(processed_data.grading_system, selected_gym, str(boulder))
                                french_grades_for_chart.append(grade)
                        else:
                            french_grades_for_chart = ["N/A"] * len(sorted_boulders_for_chart)
                        
                        # Enhanced text values with French grades
                        if any(grade != "N/A" for grade in french_grades_for_chart):
                            text_values = [f"{a}<br>{p:.1f}%<br>{grade}" for a, p, grade in zip(ascents_list, climber_percentages, french_grades_for_chart)]
                            hover_template = 'Boulder %{x}<br>Ascents: %{y}<br>%{customdata:.1f}% of climbers<br>French Grade: %{text}<extra></extra>'
                            custom_data = list(zip(climber_percentages, french_grades_for_chart))
                            hover_text = french_grades_for_chart
                        else:
                            # Fallback to original format if no grades available
                            text_values = [f"{a}<br>{p:.1f}%" for a, p in zip(ascents_list, climber_percentages)]
                            hover_template = 'Boulder %{x}<br>Ascents: %{y}<br>%{customdata:.1f}% of climbers<extra></extra>'
                            custom_data = climber_percentages
                            hover_text = None
                        
                        # Create popularity chart
                        fig_pop = go.Figure()
                        
                        # Prepare hover template and custom data for proper display
                        if hover_text:
                            # Enhanced hover with French grades
                            hovertemplate_final = 'Boulder %{x}<br>Ascents: %{y}<br>%{customdata[0]:.1f}% of climbers<br>French Grade: %{customdata[1]}<extra></extra>'
                            customdata_final = custom_data
                        else:
                            # Standard hover without French grades
                            hovertemplate_final = 'Boulder %{x}<br>Ascents: %{y}<br>%{customdata:.1f}% of climbers<extra></extra>'
                            customdata_final = custom_data
                        
                        fig_pop.add_trace(go.Bar(
                            x=sorted_boulders_for_chart,
                            y=ascents_list,
                            marker=dict(
                                color=ascents_list,
                                colorscale='Viridis',
                                line=dict(
                                    width=[2 if boulder in selected_climber_boulders else 0 for boulder in sorted_boulders_for_chart],
                                    color=['#ff7c24' if boulder in selected_climber_boulders else 'rgba(0,0,0,0)' for boulder in sorted_boulders_for_chart]
                                )
                            ),
                            hovertemplate=hovertemplate_final,
                            customdata=customdata_final,
                            text=text_values,
                            textposition='outside',
                            cliponaxis=False,
                            showlegend=False
                        ))
                        
                        # Text settings - show the enhanced text values on the chart
                        fig_pop.update_traces(
                            textangle=0, 
                            texttemplate='%{text}',
                            textposition='outside'
                        )
                        
                        # Chart layout
                        fig_pop.update_layout(
                            xaxis_title='Boulder Number', 
                            yaxis_title='Ascents', 
                            height=450,
                            xaxis=dict(type='category'),
                            margin=dict(l=25, r=15, t=15, b=25), 
                            uniformtext_minsize=4,
                            uniformtext_mode='hide',
                            plot_bgcolor='rgba(0,0,0,0.02)', 
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_pop, use_container_width=True)

                        # Boulder Stats Table
                        render_section_header("Boulder Stats", level=5)
                        display_df = boulder_df_display[['Boulder', 'Ascents', 'Success Rate (%)', 'Difficulty (1-10)']].copy()
                        display_df.columns = ['Boulder', 'Ascents', 'Success Rate', 'Difficulty']
                        
                        # Add climber completion indicator
                        if config.SESSION_KEY_SELECTED_CLIMBER in st.session_state and selected_climber_boulders:
                            display_df['Completed by You'] = display_df['Boulder'].apply(
                                lambda boulder: "✓" if boulder in selected_climber_boulders else ""
                            )
                        
                        # Add French Grade column if grading system is available
                        if processed_data and hasattr(processed_data, 'grading_system') and processed_data.grading_system:
                            display_df['French Grade'] = display_df['Boulder'].apply(
                                lambda boulder: get_boulder_french_grade(processed_data.grading_system, selected_gym, str(boulder))
                            )
                        
                        # Display table
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=350,
                            hide_index=True,
                            column_config={
                                "Boulder": st.column_config.TextColumn("Boulder"),
                                "Ascents": st.column_config.NumberColumn("Ascents", format="%d"),
                                "Success Rate": st.column_config.NumberColumn("Success Rate (%)", format="%.1f%%"),
                                "Difficulty": st.column_config.NumberColumn("Difficulty (1-10)", format="%d", help="1=easiest, 10=hardest"),
                                "Completed by You": st.column_config.TextColumn("Completed", help="Boulders you have topped") if 'Completed by You' in display_df.columns else None,
                                "French Grade": st.column_config.TextColumn("Est. French Grade", help="Estimated French boulder grade") if 'French Grade' in display_df.columns else None
                            }
                        )

                    else:
                        st.warning(f"No displayable boulder data found for {selected_gym}.")
                else:
                    st.warning(f"No boulder data available for {selected_gym}")
                
                #--------------------------------------------------------------
                # CLIMBER DISTRIBUTION ANALYSIS SECTION
                #--------------------------------------------------------------
                if completion_histograms and selected_gym in completion_histograms:
                    render_section_header("Climber Distribution Analysis", level=4)
                    
                    hist = completion_histograms.get(selected_gym, {})
                    if hist:
                        # Prepare data for distribution visualization
                        hist_data = []
                        for n_completed, count in hist.items():
                            if n_completed > 0:  # Exclude 0 completions
                                hist_data.extend([n_completed] * count)

                        # Only show visualization if we have valid data
                        if hist_data:
                            hist_array = np.array(hist_data)
                            mean = np.mean(hist_array)
                            median = np.median(hist_array)
                            std_dev = np.std(hist_array)

                            # Summary metrics in info bar
                            st.markdown(f"""
                            <div class="info-bar">
                                <b>Climbers:</b> {len(hist_array)} | 
                                <b>Mean:</b> {mean:.2f} | <b>Median:</b> {median:.2f} | <b>Std Dev:</b> {std_dev:.2f}
                            </div>
                            """, unsafe_allow_html=True)

                            # Distribution Plot
                            render_section_header("Distribution of Completed Boulders", level=5)
                            st.caption("Note: Distribution analysis excludes climbers who completed 0 boulders at this gym.")

                            # Generate x values for normal distribution
                            x = np.linspace(0, 41, 400)
                            normal_dist = None
                            plot_normal = (len(np.unique(hist_array)) >= 5 and std_dev > 0.1)
                            if plot_normal:
                                normal_dist = norm.pdf(x, mean, std_dev)

                            # Create histogram
                            unique, counts = np.unique(hist_array, return_counts=True)
                            hist_df = pd.DataFrame({'Boulders_Completed': unique, 'Climber_Count': counts}).sort_values('Boulders_Completed')

                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Bar(
                                x=hist_df['Boulders_Completed'], 
                                y=hist_df['Climber_Count'], 
                                name='Actual Data', 
                                marker_color='royalblue', 
                                opacity=0.7
                            ))

                            # Scale normal distribution
                            total_climbers = np.sum(counts)
                            bin_width = 1
                            y_scale_factor = total_climbers * bin_width
                            
                            # Add normal distribution curve if appropriate
                            if plot_normal and normal_dist is not None:
                                fig_dist.add_trace(go.Scatter(
                                    x=x, 
                                    y=normal_dist * y_scale_factor, 
                                    mode='lines', 
                                    name='Normal Dist.', 
                                    line=dict(color='red', width=2)
                                ))

                            # Add mean line
                            fig_dist.add_vline(
                                x=mean, 
                                line_dash="dash", 
                                line_color="red", 
                                annotation_text=f"Mean: {mean:.2f}", 
                                annotation_position="top right"
                            )

                            # Set x-axis ticks
                            min_boulders = 1
                            max_boulders = 40
                            all_possible_x_values = list(range(min_boulders, max_boulders + 1))
                            
                            # Histogram layout
                            fig_dist.update_layout(
                                xaxis_title="Number of Boulders Completed", 
                                yaxis_title="Number of Climbers", 
                                barmode='overlay', 
                                height=400,
                                margin=dict(l=40, r=40, t=20, b=40),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                xaxis=dict(
                                    tickmode='array', 
                                    tickvals=all_possible_x_values, 
                                    ticktext=[str(val) for val in all_possible_x_values],
                                    range=[0.5, 40.5]
                                ),
                                plot_bgcolor='rgba(0,0,0,0.02)', 
                                paper_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)

                            # Distribution Data Table
                            render_section_header("Distribution Data", level=5)
                            hist_df.columns = ['Boulders Completed', 'Climber Count']
                            hist_df['Percentage'] = (hist_df['Climber Count'] / hist_df['Climber Count'].sum() * 100)
                            with st.expander("View Full Distribution Data", expanded=False):
                                st.dataframe(
                                    hist_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Boulders Completed": st.column_config.NumberColumn(format="%d"),
                                        "Climber Count": st.column_config.NumberColumn(format="%d"),
                                        "Percentage": st.column_config.NumberColumn(format="%.1f%%")
                                    }
                                )
                        else:
                            st.info(f"No climbers completed > 0 boulders at {selected_gym} to analyze distribution.")
                    else:
                        st.warning(f"No completion histogram data available for {selected_gym}")
                    
                    # Display outlier warning if present
                    if outlier_warning_message and "outlier" in outlier_warning_message.lower():
                        st.markdown("---")
                        st.info(f"📊 Note: {outlier_warning_message}")
                
                #--------------------------------------------------------------
                # BOULDER CHARACTERISTICS ANALYSIS SECTION
                #--------------------------------------------------------------
                render_section_header("Boulder Characteristics Analysis", level=4)
                
                # Load boulder data
                try:
                    with open("boulders.json", "r") as f:
                        boulders_data = json.load(f)
                    
                    # Process gym boulder data
                    if selected_gym in boulders_data:
                        gym_boulders = boulders_data[selected_gym]
                        
                        # Get selected climber's completed boulders
                        selected_climber_boulders_ids = []
                        selected_climber_name = None
                        if config.SESSION_KEY_SELECTED_CLIMBER in st.session_state:
                            selected_climber = st.session_state[config.SESSION_KEY_SELECTED_CLIMBER]
                            selected_climber_name = selected_climber
                            _, climber_gym_boulders = create_climber_boulder_matrix(data)
                            
                            if selected_climber in climber_gym_boulders and selected_gym in climber_gym_boulders[selected_climber]:
                                selected_climber_boulders_ids = [int(b) for b in climber_gym_boulders[selected_climber][selected_gym] if b.isdigit()]
                        
                        # Extract category data
                        categories = extract_boulder_data(gym_boulders, selected_climber_boulders_ids, "category")
                        # Extract wall angle data
                        angles = extract_boulder_data(gym_boulders, selected_climber_boulders_ids, "angle")
                        # Extract hold type data
                        holds = extract_boulder_data(gym_boulders, selected_climber_boulders_ids, "holds")
                        # Extract techniques data
                        techniques = extract_boulder_data(gym_boulders, selected_climber_boulders_ids, "techniques")
                        
                        # Create tabs for the different boulder characteristics
                        if any([categories, angles, holds, techniques]):
                            tab1, tab2, tab3, tab4 = st.tabs(["Categories", "Wall Angles", "Hold Types", "Techniques"])
                            
                            # Generate content for each tab
                            with tab1:
                                generate_characteristic_tab(
                                    tab1, 
                                    categories, 
                                    "Category", 
                                    selected_climber_name,
                                    sort_by_count=False,
                                    no_data_message=f"No category data available for boulders in {selected_gym}."
                                )
                                
                            with tab2:
                                generate_characteristic_tab(
                                    tab2, 
                                    angles, 
                                    "Wall Angle", 
                                    selected_climber_name,
                                    sort_by_count=False,
                                    no_data_message=f"No wall angle data available for boulders in {selected_gym}."
                                )
                                
                            with tab3:
                                generate_characteristic_tab(
                                    tab3, 
                                    holds, 
                                    "Hold Type", 
                                    selected_climber_name,
                                    sort_by_count=True,
                                    no_data_message=f"No hold type data available for boulders in {selected_gym}."
                                )
                                
                            with tab4:
                                generate_characteristic_tab(
                                    tab4, 
                                    techniques, 
                                    "Technique", 
                                    selected_climber_name,
                                    sort_by_count=True,
                                    no_data_message=f"No technique data available for boulders in {selected_gym}."
                                )
                        else:
                            st.info(f"No characteristic data available for boulders in {selected_gym}.")
                    else:
                        st.info(f"No boulder data found for {selected_gym} in boulders.json.")
                except Exception as e:
                    st.error(f"Error loading boulder characteristic data: {str(e)}")

    else:
        st.warning("No gym data loaded.") 
