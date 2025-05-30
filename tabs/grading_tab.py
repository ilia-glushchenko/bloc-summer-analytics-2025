"""
French Boulder Grading Tab for BLOC Summer Sessions 2025 Analysis

This module provides the UI for displaying French boulder grading analysis,
including grade distributions, difficulty factors, and individual boulder grades.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import numpy as np

# Try to import grading system
try:
    from grading_system import FrenchGradingSystem, BoulderGrade
    GRADING_SYSTEM_AVAILABLE = True
except ImportError:
    GRADING_SYSTEM_AVAILABLE = False


def display_grading_analysis(processed_data: Any) -> None:
    """
    Display French boulder grading analysis tab.
    
    Args:
        processed_data: ProcessedData object containing grading system and other data
    """
    # st.markdown("## 🎯 French Boulder Grading Analysis")
    
    if not GRADING_SYSTEM_AVAILABLE:
        st.error("❌ Grading system module not available. Please check your installation.")
        return
    
    grading_system = processed_data.grading_system
    
    if grading_system is None:
        st.warning("⚠️ No grading data available. The grading system could not be initialized.")
        st.info("This might happen if there's insufficient data or if Boulder 31 at Boulderbar Wienerberg is not found.")
        return
    
    # Gym difficulty factors
    # st.markdown("### 🏋️ Gym Difficulty Factors")
    
    if grading_system.gym_difficulty_factors:
        # Create a DataFrame for better display
        difficulty_data = []
        for gym, factor in grading_system.gym_difficulty_factors.items():
            difficulty_level = "Same" if factor == 1.0 else ("Harder" if factor > 1.0 else "Easier")
            difficulty_data.append({
                'Gym': gym,
                'Difficulty Factor': f"{factor:.2f}",
                'Relative Difficulty': difficulty_level,
                'Factor Value': factor
            })
        
        difficulty_df = pd.DataFrame(difficulty_data)
        
        # Display as a styled table
        st.dataframe(
            difficulty_df[['Gym', 'Difficulty Factor', 'Relative Difficulty']],
            use_container_width=True,
            hide_index=True
        )
        
        # Create a bar chart for difficulty factors
        fig_difficulty = px.bar(
            difficulty_df,
            x='Gym',
            y='Factor Value',
            title='Gym Difficulty Factors (Relative to Boulderbar Wienerberg)',
            color='Factor Value',
            color_continuous_scale='RdYlBu_r',
            text='Difficulty Factor'
        )
        fig_difficulty.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                               annotation_text="Reference (Boulderbar Wienerberg)")
        fig_difficulty.update_layout(height=400)
        fig_difficulty.update_traces(textposition='outside')
        st.plotly_chart(fig_difficulty, use_container_width=True)
    
    # Grade distributions
    st.markdown("### 📊 Grade Distributions by Gym")
    
    # Create tabs for different views
    dist_tab1, dist_tab2, dist_tab3 = st.tabs(["📈 Distribution Charts", "📋 Grade Tables", "🔍 Boulder Details"])
    
    with dist_tab1:
        # Plot grade distributions for each gym
        gyms = sorted(grading_system.boulder_grades.keys())
        
        if len(gyms) > 0:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=gyms,
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # French grade order for consistent x-axis
            grade_order = sorted(grading_system.FRENCH_GRADES.keys(), 
                               key=lambda g: grading_system.FRENCH_GRADES[g])
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for idx, gym in enumerate(gyms):
                row = (idx // 2) + 1
                col = (idx % 2) + 1
                
                distribution = grading_system.get_gym_grade_distribution(gym)
                
                # Prepare data for this gym
                grades_present = []
                counts = []
                
                for grade in grade_order:
                    if grade in distribution:
                        grades_present.append(grade)
                        counts.append(distribution[grade])
                
                if grades_present:
                    fig.add_trace(
                        go.Bar(
                            x=grades_present,
                            y=counts,
                            name=gym,
                            marker_color=colors[idx % len(colors)],
                            text=counts,
                            textposition='outside',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title_text="Grade Distribution by Gym",
                height=1000,
                showlegend=False
            )
            
            # Update x and y axis labels
            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(title_text="French Grade", row=i, col=j)
                    fig.update_yaxes(title_text="Number of Boulders", row=i, col=j)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Overall grade distribution comparison
        st.markdown("#### 🔄 Comparative Grade Distribution")
        
        # Prepare data for comparative chart
        all_grade_data = []
        for gym in gyms:
            distribution = grading_system.get_gym_grade_distribution(gym)
            total_boulders = sum(distribution.values())
            
            for grade, count in distribution.items():
                percentage = (count / total_boulders * 100) if total_boulders > 0 else 0
                all_grade_data.append({
                    'Gym': gym,
                    'Grade': grade,
                    'Count': count,
                    'Percentage': percentage,
                    'Grade_Numeric': grading_system.FRENCH_GRADES.get(grade, 0)
                })
        
        if all_grade_data:
            grade_df = pd.DataFrame(all_grade_data)
            
            # Create stacked bar chart
            fig_stacked = px.bar(
                grade_df,
                x='Grade',
                y='Percentage',
                color='Gym',
                title='Grade Distribution Comparison (Percentage)',
                category_orders={'Grade': grade_order}
            )
            fig_stacked.update_layout(height=600)
            st.plotly_chart(fig_stacked, use_container_width=True)
    
    with dist_tab2:
        # Display grade distribution tables
        for gym in gyms:
            st.markdown(f"#### 🏢 {gym}")
            
            distribution = grading_system.get_gym_grade_distribution(gym)
            total_boulders = sum(distribution.values())
            
            if distribution:
                # Create table data
                table_data = []
                for grade in grade_order:
                    if grade in distribution:
                        count = distribution[grade]
                        percentage = (count / total_boulders * 100) if total_boulders > 0 else 0
                        table_data.append({
                            'French Grade': grade,
                            'Number of Boulders': count,
                            'Percentage': f"{percentage:.1f}%"
                        })
                
                if table_data:
                    table_df = pd.DataFrame(table_data)
                    st.dataframe(table_df, use_container_width=True, hide_index=True)
                    
                    # Add summary statistics
                    difficulty_factor = grading_system.gym_difficulty_factors.get(gym, 1.0)
                    st.caption(f"📊 Total: {total_boulders} boulders | "
                             f"Difficulty Factor: {difficulty_factor:.2f}")
            else:
                st.info("No grade data available for this gym.")
            
            st.markdown("---")
    
    with dist_tab3:
        # Boulder details with search and filtering
        st.markdown("#### 🔍 Individual Boulder Grades")
        
        # Gym selector for boulder details
        selected_gym = st.selectbox(
            "Select Gym for Boulder Details:",
            options=gyms,
            key="boulder_details_gym"
        )
        
        if selected_gym and selected_gym in grading_system.boulder_grades:
            boulders = grading_system.boulder_grades[selected_gym]
            
            # Create boulder details table
            boulder_data = []
            for boulder_id, boulder_grade in boulders.items():
                boulder_data.append({
                    'Boulder ID': boulder_id,
                    'French Grade': boulder_grade.french_grade,
                    'Completion Rate': f"{boulder_grade.completion_rate:.1%}",
                    'Completed Count': boulder_grade.completed_count,
                    'Total Climbers': boulder_grade.total_climbers,
                    'Confidence': f"{boulder_grade.confidence:.2f}",
                    'Numeric Grade': f"{boulder_grade.grade_numeric:.1f}"
                })
            
            if boulder_data:
                boulder_df = pd.DataFrame(boulder_data)
                
                # Add filtering options
                col1, col2 = st.columns(2)
                
                with col1:
                    grade_filter = st.multiselect(
                        "Filter by Grade:",
                        options=sorted(set(boulder_df['French Grade']), 
                                     key=lambda g: grading_system.FRENCH_GRADES.get(g, 0)),
                        key="grade_filter"
                    )
                
                with col2:
                    min_completion = st.slider(
                        "Minimum Completion Rate:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        format="%.0f%%",
                        key="min_completion_filter"
                    )
                
                # Apply filters
                filtered_df = boulder_df.copy()
                
                if grade_filter:
                    filtered_df = filtered_df[filtered_df['French Grade'].isin(grade_filter)]
                
                # Convert completion rate back to float for filtering
                filtered_df['Completion Rate Numeric'] = filtered_df['Completion Rate'].str.rstrip('%').astype(float) / 100
                filtered_df = filtered_df[filtered_df['Completion Rate Numeric'] >= min_completion]
                
                # Display filtered results
                display_df = filtered_df.drop('Completion Rate Numeric', axis=1)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                st.caption(f"Showing {len(display_df)} of {len(boulder_df)} boulders")
                
                # Add scatter plot of completion rate vs grade
                if len(filtered_df) > 0:
                    fig_scatter = px.scatter(
                        filtered_df,
                        x='Numeric Grade',
                        y='Completion Rate Numeric',
                        hover_data=['Boulder ID', 'French Grade'],
                        title=f'Completion Rate vs Grade - {selected_gym}',
                        labels={
                            'Numeric Grade': 'Numeric Grade Value',
                            'Completion Rate Numeric': 'Completion Rate'
                        }
                    )
                    fig_scatter.update_traces(marker=dict(size=8))
                    fig_scatter.update_layout(height=600)
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No boulder data available for this gym.")
    
    # Removed validation metrics section
    # st.markdown("### 📈 Grading System Validation")
    # 
    # # Import stats functions for correlation analysis
    # try:
    #     from stats import analyze_grade_correlations
    #     correlations = analyze_grade_correlations(grading_system)
    #     
    #     if correlations:
    #         col1, col2, col3 = st.columns(3)
    #         
    #         with col1:
    #             st.metric(
    #                 "Completion Rate vs Grade Correlation",
    #                 f"{correlations['completion_rate_vs_grade_correlation']:.3f}",
    #                 help="Correlation between completion rates and assigned grades. Higher absolute values indicate better grading consistency."
    #             )
    #         
    #         with col2:
    #             st.metric(
    #                 "R-squared",
    #                 f"{correlations['r_squared']:.3f}",
    #                 help="Proportion of variance in grades explained by completion rates. Higher values indicate better model fit."
    #             )
    #         
    #         with col3:
    #             st.metric(
    #                 "Sample Size",
    #                 f"{correlations['sample_size']} boulders",
    #                 help="Total number of boulders analyzed across all gyms."
    #             )
    #         
    #         # Interpretation
    #         r_squared = correlations['r_squared']
    #         if r_squared > 0.8:
    #             st.success("🎯 **Excellent correlation!** The grading system shows strong consistency with completion rates.")
    #         elif r_squared > 0.6:
    #             st.info("✅ **Good correlation.** The grading system is reasonably consistent with completion rates.")
    #         elif r_squared > 0.4:
    #             st.warning("⚠️ **Moderate correlation.** The grading system shows some consistency but could be improved.")
    #         else:
    #             st.error("❌ **Poor correlation.** The grading system may need adjustment or more calibration points.")
    #     
    # except ImportError:
    #     st.warning("Validation metrics not available.")
    
    # Removed export functionality section
    # st.markdown("### 💾 Export Grading Data")
    # 
    # col1, col2 = st.columns(2)
    # 
    # with col1:
    #     if st.button("📄 Generate Grading Report", key="generate_report"):
    #         report = grading_system.generate_grading_report()
    #         st.text_area(
    #             "Grading Report:",
    #             value=report,
    #             height=300,
    #             key="grading_report_text"
    #         )
    # 
    # with col2:
    #     if st.button("💾 Export to JSON", key="export_json"):
    #         try:
    #             # Get current gender for filename
    #             current_gender = st.session_state.get('selected_gender', 'unknown')
    #             filename = f"boulder_grades_{current_gender}.json"
    #             grading_system.export_grades_to_json(filename)
    #             st.success(f"✅ Grades exported to {filename}")
    #         except Exception as e:
    #             st.error(f"❌ Export failed: {str(e)}") 
