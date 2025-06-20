import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional

import config # Import config

# Assuming ui_components.py is in the parent directory or accessible
from ui_components import render_metrics_row, render_section_header

def display_climber_stats(data: List[Dict], climbers_df: pd.DataFrame, completion_histograms: Dict, outlier_warning_message: Optional[str]) -> None:
    """Displays the Climber Statistics tab content."""
    st.markdown("### Climber Distribution Analysis")

    # Check if this is combined division by looking for Gender column
    is_combined_division = 'Gender' in climbers_df.columns

    # Use helper for metrics row
    if not climbers_df.empty:
        avg_boulders = climbers_df['Completed'].mean()
        # Use the new 'Gyms_Active' column for the average
        avg_gyms_active = climbers_df['Gyms_Active'].mean()
        
        # Create metrics based on division type
        if is_combined_division:
            men_count = len(climbers_df[climbers_df['Gender'] == 'M'])
            women_count = len(climbers_df[climbers_df['Gender'] == 'F'])
            render_metrics_row({
                "Total Climbers": len(data),
                "Men": men_count,
                "Women": women_count,
                "Avg Boulders/Climber": f"{avg_boulders:.1f}",
                "Avg Active Gyms Visited": f"{avg_gyms_active:.1f}"
            })
        else:
            render_metrics_row({
                "Total Climbers": len(data),
                "Avg Boulders/Climber": f"{avg_boulders:.1f}", # Format value here
                "Avg Active Gyms Visited": f"{avg_gyms_active:.1f}" # Updated label and value
            })
    else:
         st.warning("No climber data to display metrics.")

    # Display rankings directly without subtabs
    if not climbers_df.empty:
        # Apply the specific class to reduce margin below this header
        header_text = "Combined Division Rankings" if is_combined_division else "Climber Rankings"
        render_section_header(header_text, level=4, css_class="rankings-main-header")

        # Sort climbers by appropriate rank column
        rank_column = 'Combined_Rank' if is_combined_division else 'Rank'
        sorted_climbers = climbers_df.sort_values(rank_column)

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
            
            # Create hover template based on division type
            if is_combined_division:
                hover_template = 'Climber: %{x}<br>Gender: %{customdata[1]}<br>Completed: %{y}<br>Rank: %{customdata[0]}<extra></extra>'
                custom_data = list(zip(top_climbers[rank_column], top_climbers['Gender']))
            else:
                hover_template = 'Climber: %{x}<br>Completed: %{y}<br>Rank: %{customdata}<extra></extra>'
                custom_data = top_climbers[rank_column]
            
            fig_top.add_trace(go.Bar(
                x=top_climbers['Climber'],
                y=top_climbers['Completed'],
                marker=dict(
                    color=top_climbers['Completed'],  # Use completion values for the color scale
                    colorscale='Viridis'  # Use the same Viridis colorscale as in boulder popularity
                ),
                hovertemplate=hover_template,
                customdata=custom_data,
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
                selected_rank = selected_climber_row[rank_column]
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
                    hovertemplate=hover_template,
                    customdata=custom_data,  # Use the same custom data as the main trace
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
                height=450,  # Increased from 400 to 450 to provide more space for labels
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
        
        # Define base columns based on division type
        if is_combined_division:
            base_cols = ['Combined_Rank', 'Climber', 'Gender', 'Completed', 'Gyms_Active', 'Avg_Per_Gym_Active']
        else:
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
            # Define callback for previous button
            def go_to_previous_page():
                st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] -= 1
            
            # Use constant for session key and make button fill column
            st.button(
                "← Previous", 
                use_container_width=True, 
                disabled=(st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] <= 1),
                on_click=go_to_previous_page
            )
                
        with col2:
            # Use markdown to explicitly center the page indicator text
            if max_pages > 0:
                page_text = f"Page {st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB]} of {max_pages}"
                st.markdown(f"<div style='text-align: center;'>{page_text}</div>", unsafe_allow_html=True)
            else:
                st.write("Page 1 of 1") # Handle case with 0 or 1 page
        
        with col3: # Right column for Next button
            # Define callback for next button
            def go_to_next_page():
                st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] += 1
            
            # Remove nested columns, button directly in col3
            st.button(
                "Next →", 
                use_container_width=True, 
                disabled=(st.session_state[config.SESSION_KEY_CURRENT_PAGE_CLIMBER_TAB] >= max_pages),
                on_click=go_to_next_page
            )
            
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
            if is_combined_division:
                html_table += f"<th style='text-align: center;'>Rank</th>"
                html_table += f"<th style='text-align: left;'>Climber</th>"
                html_table += f"<th style='text-align: center;'>Gender</th>"
                html_table += f"<th style='text-align: center;'>Stats</th>"
                html_table += f"<th style='text-align: left;'>Gym Completions (Max 40)</th>"
            else:
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
                
                # Rank column
                rank_value = int(row[rank_column])
                html_table += f"<td style='text-align: center;'>{rank_value}</td>"
                
                # Climber name column
                climber_name = row['Climber']
                font_weight = "bold" if is_selected else "normal"
                html_table += f"<td style='text-align: left; font-weight: {font_weight};'>{climber_name}</td>"
                
                # Gender column (only for combined division)
                if is_combined_division:
                    gender_display = "♂" if row['Gender'] == 'M' else "♀"
                    gender_color = "#4169E1" if row['Gender'] == 'M' else "#DC143C"
                    html_table += f"<td style='text-align: center; color: {gender_color}; font-size: 1.2em;'>{gender_display}</td>"
                
                # Stats column
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
    else:
        st.warning("No climber data available to display tabs.") 
