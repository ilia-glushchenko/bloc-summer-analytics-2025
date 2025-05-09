import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional
import html # Import the html module for escaping
import numpy as np
from scipy.stats import norm

import config # Import config
from data_processing import create_climber_boulder_matrix # Add this import to get climber-boulder data

# Assuming ui_components.py is in the parent directory or accessible
from ui_components import render_metrics_row, render_section_header

def display_gym_stats(data: List[Dict], gym_boulder_counts: Dict, participation_counts: Dict, completion_histograms: Dict = None, outlier_warning_message: Optional[str] = None) -> None:
    """Displays the Gym Statistics tab content."""
    st.markdown("### Climbing Gym Statistics")

    # Use helper to render top metrics
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
            # Use constant for session key
            if config.SESSION_KEY_SELECTED_GYM_STATS_TAB not in st.session_state:
                st.session_state[config.SESSION_KEY_SELECTED_GYM_STATS_TAB] = gym_names[0]  # Default to first gym
            
            st.markdown("<div class='gym-button-row'>", unsafe_allow_html=True)
            
            # Calculate how many gyms to show per row (max 4)
            gyms_per_row = min(4, len(gym_names))
            cols = st.columns(gyms_per_row)
            
            for i, gym_name in enumerate(gym_names):
                col_idx = i % gyms_per_row
                with cols[col_idx]:
                    # Use constant for session key
                    is_active = st.session_state[config.SESSION_KEY_SELECTED_GYM_STATS_TAB] == gym_name
                    button_class = "gym-button active" if is_active else "gym-button"
                    
                    # Define callback for gym button
                    def select_gym(gym=gym_name):
                        # Use constant for session key
                        st.session_state[config.SESSION_KEY_SELECTED_GYM_STATS_TAB] = gym
                    
                    st.button(
                        gym_name, 
                        key=f"gym_btn_{gym_name}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                        on_click=select_gym
                    )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Use constant for session key
            selected_gym = st.session_state[config.SESSION_KEY_SELECTED_GYM_STATS_TAB]

            if selected_gym:
                # Get data for the selected gym first
                boulder_counts = gym_boulder_counts.get(selected_gym, {})
                total_climbers_at_gym = participation_counts.get(selected_gym, 0)

                # Proceed only if there IS boulder data
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
                            # Optional: Log or warn about non-numeric boulder names if needed
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

                    complete_boulder_data = []
                    for boulder_num in combined_boulders:
                        complete_boulder_data.append({
                            'Boulder': boulder_num,
                            'Ascents': boulder_counts.get(boulder_num, 0)
                        })
                        
                    if complete_boulder_data: # Ensure list is not empty before creating df
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

                    # --- Now, check if we have a valid, non-empty df to display ---
                    if not boulder_df.empty:
                        # Filter DataFrame based on checkbox - REMOVED Logic, always filter
                        boulder_df_display = boulder_df.copy() # Show all boulders

                        # Content previously inside the inner 'if' now runs directly
                        render_section_header(f"Boulder Analysis: {selected_gym}", level=4)

                        # Summary metrics info bar (use original df counts for accuracy here)
                        total_boulders_in_gym = len(boulder_df) # Count before filtering
                        untouched_count = len(boulder_df[boulder_df['Ascents'] == 0]) # Count before filtering
                        total_ascents_in_gym = boulder_df['Ascents'].sum() # Sum before filtering

                        st.markdown(f"""
                        <div class=\"info-bar\">
                            <b>Total Boulders:</b> {html.escape(str(total_boulders_in_gym))} | <b>Total Ascents:</b> {html.escape(str(total_ascents_in_gym))} |
                            <b>Participants:</b> {html.escape(str(total_climbers_at_gym))} | <b>Untouched:</b> {html.escape(str(untouched_count))}
                        </div>
                        """, unsafe_allow_html=True)

                        # Get the selected climber's completed boulders for this gym (if any)
                        selected_climber_boulders = []
                        if config.SESSION_KEY_SELECTED_CLIMBER in st.session_state:
                            selected_climber = st.session_state[config.SESSION_KEY_SELECTED_CLIMBER]
                            # Get climber-boulder matrix to check which boulders they've completed
                            _, climber_gym_boulders = create_climber_boulder_matrix(data)
                            
                            # Check if the selected climber has data for this gym
                            if selected_climber in climber_gym_boulders and selected_gym in climber_gym_boulders[selected_climber]:
                                selected_climber_boulders = climber_gym_boulders[selected_climber][selected_gym]

                        # Boulder Popularity Chart (use filtered data)
                        render_section_header("Boulder Popularity", level=5)
                        # Sort boulders for the chart based on the filtered dataframe's order
                        sorted_boulders_for_chart = boulder_df_display['Boulder'].tolist()
                        ascents_list = boulder_df_display['Ascents'].tolist()
                        climber_percentages = [round((ascents / total_climbers_at_gym * 100), 1) if total_climbers_at_gym > 0 else 0
                                              for ascents in ascents_list]
                        
                        # Prepare combined text with the percentage below the number
                        text_values = [f"{a}<br>{p:.1f}%" for a, p in zip(ascents_list, climber_percentages)]
                        
                        fig_pop = go.Figure()
                        
                        # Add main bar trace with combined text (number + percentage)
                        fig_pop.add_trace(go.Bar(
                            x=sorted_boulders_for_chart,
                            y=ascents_list,
                            marker=dict(
                                color=ascents_list,  # Use completion values for the color scale
                                colorscale='Viridis',  # Use the same Viridis colorscale as in boulder popularity
                                line=dict(
                                    width=[2 if boulder in selected_climber_boulders else 0 for boulder in sorted_boulders_for_chart],
                                    color=['#ff7c24' if boulder in selected_climber_boulders else 'rgba(0,0,0,0)' for boulder in sorted_boulders_for_chart]
                                )
                            ),
                            hovertemplate='Boulder %{x}<br>Ascents: %{y}<br>%{customdata:.1f}% of climbers<extra></extra>',
                            customdata=climber_percentages,
                            text=text_values,
                            textposition='outside',
                            cliponaxis=False,
                            showlegend=False
                        ))
                        
                        # Add text offset to prevent overlapping with bars
                        fig_pop.update_traces(textangle=0, texttemplate='%{text}', textposition='outside')
                        
                        fig_pop.update_layout(
                            xaxis_title='Boulder Number', 
                            yaxis_title='Ascents', 
                            height=450,
                            xaxis=dict(type='category'),
                            margin=dict(l=25, r=15, t=15, b=25), 
                            uniformtext_minsize=4,  # Lower min size ensures text starts hiding when needed
                            uniformtext_mode='hide',  # Text hides when too small
                            plot_bgcolor='rgba(0,0,0,0.02)', 
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_pop, use_container_width=True)

                        # Boulder Stats Table (use filtered data)
                        render_section_header("Boulder Stats", level=5)
                        # Use the filtered DataFrame for display
                        display_df = boulder_df_display[['Boulder', 'Ascents', 'Success Rate (%)', 'Difficulty (1-10)']].copy()
                        display_df.columns = ['Boulder', 'Ascents', 'Success Rate', 'Difficulty'] # Shorten names
                        # Add a column to indicate if the boulder was completed by the selected climber
                        if config.SESSION_KEY_SELECTED_CLIMBER in st.session_state and selected_climber_boulders:
                            display_df['Completed by You'] = display_df['Boulder'].apply(
                                lambda boulder: "✓" if boulder in selected_climber_boulders else ""
                            )
                        
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=350,
                            column_config={
                                "Boulder": st.column_config.TextColumn("Boulder"),
                                "Ascents": st.column_config.NumberColumn("Ascents", format="%d"),
                                "Success Rate": st.column_config.NumberColumn("Success Rate (%)", format="%.1f%%"),
                                "Difficulty": st.column_config.NumberColumn("Difficulty (1-10)", format="%d", help="1=easiest, 10=hardest"),
                                "Completed by You": st.column_config.TextColumn("Completed", help="Boulders you have topped") 
                                 if 'Completed by You' in display_df.columns else None
                            }
                        )

                    else:
                        # Display warning if no valid boulder_df could be created initially
                        st.warning(f"No displayable boulder data found for {selected_gym}.")
                else:
                    # Display warning OUTSIDE the container if no initial boulder_counts
                    st.warning(f"No boulder data available for {selected_gym}")
                
                # Add distribution analysis to gym stats tab - MOVED OUTSIDE the boulder_df checks
                if completion_histograms and selected_gym in completion_histograms:
                    render_section_header("Climber Distribution Analysis", level=4)
                    
                    hist = completion_histograms.get(selected_gym, {})
                    if hist:
                        # Prepare data for distribution visualization
                        hist_data = []
                        for n_completed, count in hist.items():
                            if n_completed > 0:  # Exclude 0 completions
                                hist_data.extend([n_completed] * count)

                        # Now, only if we have valid hist_data, draw the content
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

                            # Generate x values across the full range 1-40 for the normal distribution
                            x = np.linspace(0, 41, 400)  # Increased range and resolution
                            normal_dist = None
                            plot_normal = (len(np.unique(hist_array)) >= 5 and std_dev > 0.1)
                            if plot_normal:
                                normal_dist = norm.pdf(x, mean, std_dev)

                            unique, counts = np.unique(hist_array, return_counts=True)
                            hist_df = pd.DataFrame({'Boulders_Completed': unique, 'Climber_Count': counts}).sort_values('Boulders_Completed')

                            fig_dist = go.Figure()
                            fig_dist.add_trace(go.Bar(x=hist_df['Boulders_Completed'], y=hist_df['Climber_Count'], name='Actual Data', marker_color='royalblue', opacity=0.7))

                            # Calculate a more accurate scaling factor for the normal distribution
                            # Scale to match the total area under the histogram
                            total_climbers = np.sum(counts)
                            bin_width = 1  # Each boulder count is 1 unit wide
                            # Scale the PDF to match the histogram area
                            y_scale_factor = total_climbers * bin_width
                            
                            # Only plot normal distribution if data is suitable
                            if plot_normal and normal_dist is not None:
                                fig_dist.add_trace(go.Scatter(x=x, y=normal_dist * y_scale_factor, mode='lines', name='Normal Dist.', line=dict(color='red', width=2)))

                            fig_dist.add_vline(x=mean, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean:.2f}", annotation_position="top right")

                            # Generate all possible boulder values from 1 to 40 (the maximum per gym)
                            min_boulders = 1
                            max_boulders = 40
                            all_possible_x_values = list(range(min_boulders, max_boulders + 1))
                            
                            fig_dist.update_layout(
                                xaxis_title="Number of Boulders Completed", yaxis_title="Number of Climbers", barmode='overlay', height=400,
                                margin=dict(l=40, r=40, t=20, b=40),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                                xaxis=dict(
                                    tickmode='array', 
                                    tickvals=all_possible_x_values, 
                                    ticktext=[str(val) for val in all_possible_x_values],
                                    range=[0.5, 40.5]  # Set fixed range from 0.5 to 40.5 to ensure bars display properly
                                ),
                                plot_bgcolor='rgba(0,0,0,0.02)', paper_bgcolor='rgba(0,0,0,0)'
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
                                    column_config={
                                        "Boulders Completed": st.column_config.NumberColumn(format="%d"),
                                        "Climber Count": st.column_config.NumberColumn(format="%d"),
                                        "Percentage": st.column_config.NumberColumn(format="%.1f%%")
                                    }
                                )
                        else:
                            # Info message if hist exists but hist_data is empty (no >0 completions)
                            st.info(f"No climbers completed > 0 boulders at {selected_gym} to analyze distribution.")
                    else:
                        # Warning if hist is empty for the selected gym
                        st.warning(f"No completion histogram data available for {selected_gym}")
                    
                    # Display the outlier warning at the bottom of the distribution analysis
                    if outlier_warning_message and "outlier" in outlier_warning_message.lower():
                        st.markdown("---")
                        st.info(f"📊 Note: {outlier_warning_message}")

    else:
        st.warning("No gym data loaded.") 