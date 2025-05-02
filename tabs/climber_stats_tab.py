import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, gaussian_kde
from typing import Dict, List, Optional

import config # Import config

# Assuming ui_components.py is in the parent directory or accessible
from ui_components import render_metrics_row, render_section_header

def display_climber_stats(data: List[Dict], climbers_df: pd.DataFrame, completion_histograms: Dict, outlier_warning_message: Optional[str]) -> None:
    """Displays the Climber Statistics tab content."""
    st.markdown("### Climber Distribution Analysis")

    # Use helper for metrics row
    if not climbers_df.empty:
        avg_boulders = climbers_df['Completed'].mean()
        # Use the new 'Gyms_Active' column for the average
        avg_gyms_active = climbers_df['Gyms_Active'].mean()
        render_metrics_row({
            "Total Climbers": len(data),
            "Avg Boulders/Climber": f"{avg_boulders:.1f}", # Format value here
            "Avg Active Gyms Visited": f"{avg_gyms_active:.1f}" # Updated label and value
        })
    else:
         st.warning("No climber data to display metrics.")

    # Create subtabs like in Gym Stats
    if not climbers_df.empty:
        climber_tabs = st.tabs(["🏆 Rankings", "📊 Distribution Analysis"])

        with climber_tabs[0]:
            # Apply the specific class to reduce margin below this header
            render_section_header("Climber Rankings", level=4, css_class="rankings-main-header")

            # Sort climbers by rank
            sorted_climbers = climbers_df.sort_values('Rank')

            # Top climbers visualization
            render_section_header("Top Climbers by Boulder Completion", level=5)
            
            # Fixed to always show top 50 climbers
            top_n = 50
            top_climbers = sorted_climbers.head(top_n)
            # Add check if top_climbers is empty after selection (edge case)
            if top_climbers.empty:
                st.info("No climbers to display with the current settings.")
            else:
                # Create the figure with gradient
                fig_top = go.Figure()
                
                # We need to maintain the original order of climbers, including the selected climber
                # Instead of removing the selected climber, we'll create a graph with all blue bars first
                fig_top.add_trace(go.Bar(
                    x=top_climbers['Climber'],
                    y=top_climbers['Completed'],
                    marker=dict(
                        color=top_climbers['Completed'],  # Use completion values for the color scale
                        colorscale='Viridis'  # Use the same Viridis colorscale as in boulder popularity
                    ),
                    hovertemplate='Climber: %{x}<br>Completed: %{y}<br>Rank: %{customdata}<extra></extra>',
                    customdata=top_climbers['Rank'],
                    text=top_climbers['Completed'], 
                    textposition='outside',  # Show all values outside the bars
                    cliponaxis=False,
                    showlegend=False,  # Hide legend entry
                    width=0.8,  # Set bar width to 80% of available space
                    name='All Climbers'
                ))
                
                # Now we add the selected climber's orange bar on top
                # Use constant for session key
                if config.SESSION_KEY_SELECTED_CLIMBER in st.session_state and st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] in top_climbers['Climber'].values:
                    # Create a dataframe with zeros for all climbers
                    selected_overlay = pd.DataFrame({
                        'Climber': top_climbers['Climber'],
                        'Value': [0] * len(top_climbers)
                    })
                    
                    # Set the selected climber's value to their actual completion count
                    selected_climber_name = st.session_state[config.SESSION_KEY_SELECTED_CLIMBER]
                    selected_climber_row = top_climbers[top_climbers['Climber'] == selected_climber_name].iloc[0]
                    selected_idx = selected_climber_row.name # Use DataFrame index
                    selected_value = selected_climber_row['Completed']
                    selected_rank = selected_climber_row['Rank']
                    selected_overlay.loc[selected_idx, 'Value'] = selected_value
                    
                    # Calculate max possible boulders for percentage calculation
                    gym_comp_cols = [col for col in sorted_climbers.columns if col.startswith('Comp_')]
                    num_gyms = len(gym_comp_cols)
                    max_possible_boulders = num_gyms * 40  # Assuming 40 boulders per gym
                    
                    # Calculate percentages of total possible boulders
                    completion_percentages = [(val / max_possible_boulders * 100) if max_possible_boulders > 0 else 0 
                                              for val in top_climbers['Completed']]
                    
                    # Add transparent bar trace for all positions, with only the selected climber having a value
                    fig_top.add_trace(go.Bar(
                        x=selected_overlay['Climber'],
                        y=selected_overlay['Value'],
                        marker=dict(
                            color=['rgba(0,0,0,0)' if c != st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] else '#ff9d45' for c in selected_overlay['Climber']],
                            line=dict(
                                width=[0 if c != st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] else 2 for c in selected_overlay['Climber']],
                                color=['rgba(0,0,0,0)' if c != st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] else '#ff7c24' for c in selected_overlay['Climber']]
                            )
                        ),
                        hovertemplate='Climber: %{x}<br>Completed: %{y}<br>Rank: %{customdata}<extra></extra>',
                        customdata=top_climbers['Rank'],  # Use the same rank data
                        text=['' if c != st.session_state[config.SESSION_KEY_SELECTED_CLIMBER] else f"{selected_value}" for c in selected_overlay['Climber']],
                        textposition='outside',
                        cliponaxis=False,
                        showlegend=False,  # Hide legend entry
                        width=0.8,  # Set bar width to 80% of available space
                        name='Selected Climber'
                    ))
                
                # Add vertical line after 10th climber
                if len(top_climbers) > 10:
                    fig_top.add_vline(x=9.5, line_dash="dash", line_color="red", 
                                     annotation_text="Top 10", annotation_position="top")
                
                # Calculate maximum possible boulders (40 per gym * number of gyms)
                # Count the number of gym completion columns to determine number of gyms
                gym_comp_cols = [col for col in sorted_climbers.columns if col.startswith('Comp_')]
                num_gyms = len(gym_comp_cols)
                max_possible_boulders = num_gyms * 40  # Assuming 40 boulders per gym
                
                fig_top.update_layout(
                    xaxis_title='Climber', 
                    yaxis_title='Boulders Completed', 
                    height=400,
                    margin=dict(l=25, r=15, t=5, b=100), 
                    xaxis=dict(tickangle=45, type='category'),
                    uniformtext_minsize=8, 
                    uniformtext_mode='show',  # Changed from 'none' to 'show' to properly display labels
                    plot_bgcolor='rgba(0,0,0,0.02)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(range=[0, max_possible_boulders]),  # Set y-axis range to show all possible climbs
                    showlegend=False,  # Ensure no legend is shown
                    bargap=0.15,  # Control the gap between bars (smaller value = wider bars)
                    barmode='overlay'  # Ensure bars are perfectly overlaid
                )
                st.plotly_chart(fig_top, use_container_width=True)

            # Climbers data table
            render_section_header("Climber Stats", level=5)
            # The calculation is now done in process_data, just retrieve it
            # sorted_climbers['Avg_Per_Gym_Active'] = (sorted_climbers['Completed'] / sorted_climbers['Gyms_Active']).replace([np.inf, -np.inf], 0).fillna(0).round(1)

            # Add search field for filtering climbers
            search_query = st.text_input("🔍 Search for climber:", placeholder="Enter name to filter...")
            
            # Dynamically identify gym completion columns
            gym_comp_cols = [col for col in sorted_climbers.columns if col.startswith('Comp_')]
            # Create cleaner display names for gym columns
            gym_display_names = {col: col.replace('Comp_', '').replace('_', ' ') for col in gym_comp_cols}
            
            # Define base columns and add the dynamic gym columns
            base_cols = ['Rank', 'Climber', 'Completed', 'Gyms_Active', 'Avg_Per_Gym_Active']
            all_display_cols = base_cols + gym_comp_cols
            
            # Keep the original sorted_climbers df which has Comp_GymName columns
            display_climbers_df = sorted_climbers.copy()
            
            # Filter by search query if provided
            if search_query:
                # Case-insensitive partial matching
                display_climbers_df = display_climbers_df[
                    display_climbers_df['Climber'].str.lower().str.contains(search_query.lower())
                ]
                if display_climbers_df.empty:
                    st.info(f"No climbers found matching '{search_query}'")
            
            # Add pagination
            total_climbers = len(display_climbers_df)
            # Use constant for items per page
            items_per_page = config.RANKINGS_TABLE_ITEMS_PER_PAGE
            
            # Calculate number of pages
            max_pages = (total_climbers + items_per_page - 1) // items_per_page  # Ceiling division
            
            # Create columns for pagination controls and page info
            # Use weighted columns: smaller equal weight for buttons, larger for center
            col1, col2, col3 = st.columns([1, 4, 1])
            
            # Initialize current_page in session state if not present
            # Use constant for session key
            if config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB not in st.session_state:
                st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] = 1
                
            # Previous/Next buttons
            with col1:
                # Use constant for session key and make button fill column
                if st.button("← Previous", use_container_width=True, disabled=(st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] <= 1)):
                    st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] -= 1
                    st.rerun()
                    
            with col2:
                # Use markdown to explicitly center the page indicator text
                if max_pages > 0:
                    page_text = f"Page {st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB]} of {max_pages}"
                    st.markdown(f"<div style='text-align: center;'>{page_text}</div>", unsafe_allow_html=True)
                else:
                    st.write("Page 1 of 1") # Handle case with 0 or 1 page
            
            with col3: # Right column for Next button
                # Remove nested columns, button directly in col3
                if st.button("Next →", use_container_width=True, disabled=(st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] >= max_pages)):
                    st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] += 1
                    st.rerun()
            
            # Calculate start and end index for slicing
            # Use constant for session key
            start_idx = (st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] - 1) * items_per_page
            end_idx = start_idx + items_per_page
            
            # Get the subset of data for the current page
            if not display_climbers_df.empty:
                paged_df = display_climbers_df.iloc[start_idx:end_idx]
            else:
                paged_df = display_climbers_df  # Empty DataFrame
                
            # --- Add CSS block to override Streamlit defaults for this table ---
            # Using border-collapse: separate + border on table + selective cell borders
            st.markdown(""" 
            <style>
            #climber-rankings-table {
                border-collapse: separate; 
                border-spacing: 0;
                overflow: hidden; 
                border-radius: 8px; 
                width: 100%; 
                font-size: 0.9em;
                /* Explicit outer border on the table */
                border: 1px solid #ddd; 
            }
            #climber-rankings-table th, 
            #climber-rankings-table td {
                padding: 6px; 
                border: none; /* Removed !important flag */
                /* Apply only bottom and right borders */
                border-bottom: 1px solid #ddd !important;
                border-right: 1px solid #ddd !important;
                position: relative; /* Keep for potential future use/consistency */
            }

            /* Remove redundant borders */
            #climber-rankings-table tbody tr:last-child td {
                border-bottom: none !important; /* Updated with !important */
            }
            #climber-rankings-table th:last-child,
            #climber-rankings-table td:last-child {
                border-right: none !important; /* Updated with !important */
            }

            /* Header specific style */
            #climber-rankings-table th {
                background-color: #f0f2f6;
                text-align: center;
                 /* Need header bottom border explicitly */
                 border-bottom: 1px solid #ddd;
            }
            
            /* Corner radius still useful for clipping background colors */
            #climber-rankings-table th:first-child { border-top-left-radius: 8px; }
            #climber-rankings-table th:last-child { border-top-right-radius: 8px; }
            #climber-rankings-table tbody tr:last-child td:first-child { border-bottom-left-radius: 8px; }
            #climber-rankings-table tbody tr:last-child td:last-child { border-bottom-right-radius: 8px; }

            </style>
            """, unsafe_allow_html=True)

            # --- Generate Custom HTML Table (4 Columns) with ID ---
            # Minimal inline styles, relies on CSS block
            if not paged_df.empty:
                # Add ID, remove inline styles handled by CSS block
                html_table = "<table id='climber-rankings-table'>"
                
                # Table Headers - Minimal inline styles
                html_table += "<thead><tr>" 
                html_table += f"<th style='text-align: center;'>Rank</th>"
                html_table += f"<th style='text-align: left;'>Climber</th>"
                html_table += f"<th style='text-align: center;'>Stats</th>"
                html_table += f"<th style='text-align: left;'>Gym Completions (Max 40)</th>"
                html_table += "</tr></thead>"
                
                # Table Body - Minimal inline styles
                html_table += "<tbody>"
                for index, (_, row) in enumerate(paged_df.iterrows()):
                    is_selected = config.SESSION_KEY_SELECTED_CLIMBER in st.session_state and row['Climber'] == st.session_state[config.SESSION_KEY_SELECTED_CLIMBER]
                    row_style = "background-color: #dbeafe;" if is_selected else ""
                    
                    html_table += f"<tr style='{row_style}'>"
                    
                    # Columns - minimal inline styles
                    html_table += f"<td style='text-align: center;'>{int(row['Rank'])}</td>"
                    climber_name = row['Climber']
                    font_weight = "bold" if is_selected else "normal"
                    html_table += f"<td style='text-align: left; font-weight: {font_weight};'>{climber_name}</td>"
                    stats_html = f"<div>Toped: <strong>{int(row['Completed'])}</strong></div>" \
                                 f"<div>Gyms: <strong>{int(row['Gyms_Active'])}</strong></div>" \
                                 f"<div>Avg/Gym: <strong>{row['Avg_Per_Gym_Active']:.1f}</strong></div>"
                    html_table += f"<td style='text-align: center;'>{stats_html}</td>"
                    
                    gym_progress_html = ""
                    sorted_gym_display_names = sorted(gym_display_names.values())
                    col_name_lookup = {v: k for k, v in gym_display_names.items()}
                    for gym_name_display in sorted_gym_display_names:
                        gym_col = col_name_lookup[gym_name_display]
                        completed_count = row[gym_col]
                        progress_pct = min(100, (completed_count / 40) * 100)
                        tooltip_text = f'{gym_name_display}: {int(completed_count)} / 40 ({progress_pct:.0f}%)'
                        gym_progress_html += f"<div style='margin-bottom: 3px;' title='{tooltip_text}'>"
                        gym_progress_html += f"<div style='background-color: #e9ecef; border-radius: 5px; height: 18px; width: 95%; position: relative; text-align: left;'>"
                        gym_progress_html += f"<div style='background-color: #63b3ed; width: {progress_pct}%; height: 100%; border-radius: 5px; position: absolute; top: 0; left: 0;'></div>"
                        text_content = f"{gym_name_display} - {int(completed_count)}"
                        gym_progress_html += f"<div style='position: absolute; top: 0; left: 5px; height: 100%; line-height: 18px; color: #212529; font-size: 0.75em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 90%;'>"
                        gym_progress_html += text_content
                        gym_progress_html += f"</div></div></div>"
                    
                    html_table += f"<td style='vertical-align: top; text-align: left;'>{gym_progress_html}</td>"
                    html_table += "</tr>"
                    
                html_table += "</tbody></table>"
                st.markdown(html_table, unsafe_allow_html=True)
                
            elif search_query: 
                pass 
            else: 
                st.warning("No climber data to display in the table.")

        with climber_tabs[1]:
            # Keep the section container here AS LONG AS there is data
            # Check for completion histograms first
            if completion_histograms:
                gym_options = sorted(completion_histograms.keys())
                if gym_options:
                    # Replace selectbox with button toggles
                    
                    # Initialize session state for selected gym if not present
                    if 'selected_gym_dist_tab2' not in st.session_state:
                        st.session_state.selected_gym_dist_tab2 = gym_options[0]  # Default to first gym
                    
                    # Calculate how many gyms to show per row (max 4)
                    gyms_per_row = min(4, len(gym_options))
                    cols = st.columns(gyms_per_row)
                    
                    for i, gym_name in enumerate(gym_options):
                        col_idx = i % gyms_per_row
                        with cols[col_idx]:
                            is_active = st.session_state.selected_gym_dist_tab2 == gym_name
                            
                            if st.button(
                                gym_name, 
                                key=f"dist_gym_btn_{gym_name}",
                                use_container_width=True,
                                type="primary" if is_active else "secondary"
                            ):
                                st.session_state.selected_gym_dist_tab2 = gym_name
                                st.rerun()
                    
                    # Add visual spacing after button row
                    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
                    
                    selected_gym_dist = st.session_state.selected_gym_dist_tab2

                    hist = completion_histograms.get(selected_gym_dist, {})
                    if hist:
                        # Prepare data for distribution visualization
                        hist_data = []
                        for n_completed, count in hist.items():
                            if n_completed > 0:  # Exclude 0 completions
                                hist_data.extend([n_completed] * count)

                        # Now, only if we have valid hist_data, draw the container and content
                        if hist_data:
                            render_section_header("Climber Distribution Analysis", level=4)
                            
                            hist_array = np.array(hist_data)
                            mean = np.mean(hist_array)
                            median = np.median(hist_array)
                            std_dev = np.std(hist_array)

                            # Summary metrics in info bar
                            st.markdown(f"""
                            <div class="info-bar">
                                <b>Gym:</b> {selected_gym_dist} | <b>Active Participants (>0 boulders):</b> {len(hist_array)} | 
                                <b>Mean Boulders:</b> {mean:.2f} | <b>Median:</b> {median:.2f} | <b>Std Dev:</b> {std_dev:.2f}
                            </div>
                            """, unsafe_allow_html=True)

                            # Distribution Plot
                            render_section_header("Distribution of Completed Boulders", level=5)
                            st.caption("Note: Distribution analysis excludes climbers who completed 0 boulders at this gym.") # Added clarification

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
                            st.info(f"No climbers completed > 0 boulders at {selected_gym_dist} to analyze distribution.")
                    else:
                        # Warning if hist is empty for the selected gym
                        st.warning(f"No completion histogram data available for {selected_gym_dist}")
                else:
                    # Warning if gym_options is empty
                    st.warning("No gyms available for distribution analysis.")
            else:
                # Warning if completion_histograms itself is empty
                st.warning("No completion histogram data available")
                
            # ADD - Display the outlier warning at the bottom of the distribution analysis tab
            if outlier_warning_message and "outlier" in outlier_warning_message.lower():
                st.markdown("---")
                st.info(f"📊 Note: {outlier_warning_message}")
    else:
        st.warning("No climber data available to display tabs.") 