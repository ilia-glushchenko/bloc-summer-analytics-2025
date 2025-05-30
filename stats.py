"""
Statistical analysis module for Boulder Summer Series competition data.

This module provides functions for loading, processing, and analyzing climbing
competition data from the Boulder Summer Series. It includes functions for
computing statistics per gym, analyzing boulder popularity, and generating
visualization data.
"""

import json
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde, trim_mean
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, DefaultDict, Any, Optional, Set, Union

# Set up logging
logger = logging.getLogger(__name__)

# Import the new grading system
try:
    from grading_system import FrenchGradingSystem, initialize_grading_system_with_known_data, BoulderGrade
    GRADING_SYSTEM_AVAILABLE = True
except ImportError:
    GRADING_SYSTEM_AVAILABLE = False
    logger.warning("Grading system module not available. Grade-related functions will be disabled.")


def load_results(filename: str = 'results.json', gender: str = 'men') -> List[Dict[str, Any]]:
    """
    Load climbing results from a JSON file.
    
    Args:
        filename: Path to the JSON file containing competition results.
                 Defaults to 'results.json'.
        gender: Gender category to load ('men' or 'women'). 
                Defaults to 'men'.
    
    Returns:
        List of dictionaries representing climber data for the specified gender.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle new format with separate men/women sections
    if isinstance(data, dict) and gender in data:
        return data[gender]
    # Fallback for old format (list of all climbers)
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Invalid data format or gender '{gender}' not found in results file.")


# Helper function to replace lambda in defaultdict
def default_dict_int() -> DefaultDict[str, int]:
    """Return a new defaultdict(int) for use in nested defaultdicts."""
    return defaultdict(int)


def compute_gym_stats(data: List[Dict[str, Any]]) -> Tuple[
    DefaultDict[str, DefaultDict[str, int]],
    DefaultDict[str, Counter],
    DefaultDict[str, int]
]:
    """
    Compute statistics per gym from climber data.
    
    Args:
        data: List of climber dictionaries from the competition.
    
    Returns:
        Tuple containing:
        - gym_boulder_counts: Counts of ascents per boulder per gym
        - completion_histograms: Distribution of number of completed boulders per gym
        - participation_counts: Number of climbers who attempted at least one boulder per gym
    """
    gym_boulder_counts = defaultdict(default_dict_int)
    completion_histograms = defaultdict(Counter)
    participation_counts = defaultdict(int)
    
    for climber in data:
        for gym in climber['gyms']:
            gym_name = gym['gym']
            completed_climbs = gym.get('completed_climbs', [])
            
            # Count boulder ascents
            for boulder in completed_climbs:
                gym_boulder_counts[gym_name][boulder] += 1
            
            # Record completion count in histogram
            n_completed = len(completed_climbs)
            completion_histograms[gym_name][n_completed] += 1
            
            # Count participation (climbers who completed at least one boulder)
            if n_completed > 0:
                participation_counts[gym_name] += 1
                
    return gym_boulder_counts, completion_histograms, participation_counts


def print_gym_stats(
    gym_boulder_counts: DefaultDict[str, DefaultDict[str, int]],
    completion_histograms: DefaultDict[str, Counter],
    participation_counts: DefaultDict[str, int]
) -> None:
    """
    Print statistics per gym to the console.
    
    Args:
        gym_boulder_counts: Counts of ascents per boulder per gym
        completion_histograms: Distribution of number of completed boulders per gym
        participation_counts: Number of climbers who attempted at least one boulder per gym
    """
    print("\n---Boulder Popularity---")
    for gym, boulder_counts in gym_boulder_counts.items():
        print(f"\nGym: {gym}")
        sorted_boulders = sorted(boulder_counts.items(), key=lambda x: x[1])
        for boulder, count in sorted_boulders:
            print(f"  Boulder {boulder}: {count} ascents")

    print("\n---Completion Distribution per Gym ---")
    for gym, hist in completion_histograms.items():
        print(f"\nGym: {gym}")
        for n_completed in sorted(hist):
            print(f"  {n_completed} boulders: {hist[n_completed]} climbers")

    print("\n---Participation per Gym ---")
    for gym, count in participation_counts.items():
        print(f"Gym: {gym} - {count} climbers attempted at least one boulder")


def plot_boulder_popularity(
    gym_boulder_counts: DefaultDict[str, DefaultDict[str, int]], 
    completion_histograms: DefaultDict[str, Counter]
) -> None:
    """
    Plot boulder popularity for each gym with visualization of popularity peaks.
    
    Args:
        gym_boulder_counts: Counts of ascents per boulder per gym
        completion_histograms: Distribution of number of completed boulders per gym
    """
    gyms = sorted(list(gym_boulder_counts.keys()))  # Sort gyms alphabetically
    n_gyms = len(gyms)
    ncols = 2
    nrows = (n_gyms + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flatten()
    
    # Calculate normal peaks for each gym using multiple methods
    gym_peaks = {}
    for gym, hist in completion_histograms.items():
        data = []
        for n_completed, count in hist.items():
            if n_completed > 0:
                data.extend([n_completed] * count)
        if data:
            data = np.array(data)
            # 1. Traditional normal fit
            mu_norm, _ = norm.fit(data)
            
            # 2. Trimmed mean (robust to outliers)
            mu_trimmed = trim_mean(data, 0.1)  # Trim 10% from both ends
            
            # 3. Median (even more robust)
            mu_median = np.median(data)
            
            # 4. KDE peak (best for multi-modal data)
            mu_kde = mu_norm  # Default fallback
            if len(data) > 3:  # Need enough points for KDE
                try:
                    kde = gaussian_kde(data)
                    # Find the peak of KDE on a fine grid
                    x_grid = np.linspace(min(data), max(data), 1000)
                    kde_values = kde(x_grid)
                    mu_kde = x_grid[np.argmax(kde_values)]
                except Exception:
                    pass  # Keep fallback value if KDE fails
            
            # Store all computed peaks
            gym_peaks[gym] = {
                'norm': mu_norm,
                'trimmed': mu_trimmed,
                'median': mu_median,
                'kde': mu_kde
            }
            
            # Use KDE peak as default if it worked, otherwise use trimmed mean
            gym_peaks[gym]['best'] = mu_kde if not np.isnan(mu_kde) else mu_trimmed
    
    for idx, gym in enumerate(gyms):
        boulder_counts = gym_boulder_counts[gym]
        all_boulders = set(str(i+1) for i in range(max(int(b) for b in boulder_counts.keys()) if boulder_counts else 0))
        all_boulders.update(boulder_counts.keys())
        all_boulders = sorted(all_boulders, key=lambda x: int(x) if x.isdigit() else x)
        counts_full = [boulder_counts.get(b, 0) for b in all_boulders]
        boulders_counts_sorted = sorted(zip(all_boulders, counts_full), key=lambda x: x[1], reverse=True)  # Reversed order
        boulders_sorted = [b for b, c in boulders_counts_sorted]
        counts_sorted = [c for b, c in boulders_counts_sorted]
        ax = axes[idx]
        ax.bar(boulders_sorted, counts_sorted)
        ax.set_title(f"Boulder Popularity in {gym}")
        ax.set_xlabel("Boulder Number (sorted by ascents, descending)")
        ax.set_ylabel("Number of Ascents")
        ax.set_xticks(boulders_sorted)
        ax.set_xticklabels(boulders_sorted, rotation=90)
        ax.set_ylim(0, max(counts_sorted) + 1)
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        
        # Add a red dotted line at the position corresponding to normal peak
        if gym in gym_peaks:
            # Get the best peak estimate (KDE or trimmed mean)
            normal_peak = gym_peaks[gym]['best']
            peak_method = 'KDE' if normal_peak == gym_peaks[gym]['kde'] else 'Trimmed mean'
            
            # Find the boulder index closest to the normal peak value
            if len(boulders_counts_sorted) > 0:
                # Convert normal peak to an integer index
                peak_index = min(int(normal_peak) - 1, len(boulders_counts_sorted) - 1)
                peak_index = max(0, peak_index)  # Ensure it's not negative
                
                # Check if there are multiple boulders with the same completion count
                peak_value = counts_sorted[peak_index]
                matching_indices = [i for i, count in enumerate(counts_sorted) if count == peak_value]
                
                # If multiple matches exist, use the last (rightmost) one
                if matching_indices:
                    peak_index = matching_indices[-1]
                
                # Draw the red dotted line at the corresponding boulder position
                if peak_index < len(boulders_sorted):
                    ax.axvline(x=boulders_sorted[peak_index], color='red', linestyle=':', linewidth=2,
                              label=f'{peak_method} peak ({normal_peak:.1f})')
                    ax.legend()
    
    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_gym_completion_distributions(completion_histograms: DefaultDict[str, Counter], gender: str = 'men') -> None:
    """
    Plot histogram for number of boulders topped per climber for each gym.
    
    Creates distribution plots showing how many climbers completed different numbers of boulders,
    excluding those with 0 climbs. X axis is 1..total boulders for the gym.
    
    Args:
        completion_histograms: Distribution of number of completed boulders per gym
        gender: Gender category to load ('men' or 'women'). Defaults to 'men'.
    """
    # Load results to get total boulders per gym
    data = load_results(gender=gender)
    gym_to_total: Dict[str, int] = {}
    for climber in data:
        for gym in climber['gyms']:
            gym_name = gym['gym']
            total = gym.get('total', None)
            if total is not None:
                gym_to_total[gym_name] = total
    
    # Sort gyms alphabetically for consistency with the other graph
    gyms = sorted(list(completion_histograms.keys()))
    
    n_gyms = len(gyms)
    ncols = 2
    nrows = (n_gyms + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flatten()
    
    for idx, gym in enumerate(gyms):
        hist = completion_histograms[gym]
        total_boulders = gym_to_total.get(gym, max(hist.keys()) if hist else 1)
        # Exclude 0 climbs
        data = []
        for n_completed, count in hist.items():
            if n_completed > 0:
                data.extend([n_completed] * count)
        if not data:
            continue
        
        # Ensure all possible x values are present (for uniformity)
        min_x = 1
        max_x = total_boulders
        all_x = np.arange(min_x, max_x + 1)
        unique, counts = np.unique(data, return_counts=True)
        freq_dict = dict(zip(unique, counts))
        counts_full = [freq_dict.get(x, 0) for x in all_x]
        ax = axes[idx]
        
        # Set up x-tick positions based on integers
        x_positions = np.arange(1, max_x + 1)
        
        # Plot the bar chart using integer positions
        bars = ax.bar(x_positions, counts_full, alpha=0.6, color='g', width=0.8, label='Data (asc)')
        
        # Add empty brackets for bars with 0 frequency
        for bar, count in zip(bars, counts_full):
            if count == 0:
                ax.text(bar.get_x() + bar.get_width()/2, 0.1, '[]', ha='center', va='bottom', color='gray', fontsize=12)
        
        # Overlay normal distribution fit (only for nonzero bars)
        data = np.array(data)
        
        # Use multiple methods for more robust peak detection
        mu, std = norm.fit(data)  # Traditional normal fit
        mu_trimmed = trim_mean(data, 0.1)  # Trimmed mean (10% from each end)
        mu_median = np.median(data)  # Median (most robust to outliers)
        
        # KDE for better peak detection with multimodal data
        if len(data) > 3:
            try:
                kde = gaussian_kde(data)
                x = np.linspace(min_x - 0.5, max_x + 0.5, 300)
                kde_values = kde(x)
                mu_kde = x[np.argmax(kde_values)]
                
                # Plot the KDE curve
                kde_scaled = kde(x) * len(data) * (x_positions[1] - x_positions[0])
                ax.plot(x, kde_scaled, 'b--', linewidth=1.5, 
                       label=f'KDE fit\npeak={mu_kde:.1f}')
                
                # Draw vertical line at KDE peak
                ax.axvline(mu_kde, color='blue', linestyle='-.', linewidth=1.5, 
                          label='KDE peak')
            except Exception:
                pass
        
        # Use continuous x values for the normal curve
        x = np.linspace(min_x - 0.5, max_x + 0.5, 300)
        
        # Calculate the PDF and scale it to match the histogram height
        p = norm.pdf(x, mu, std) * len(data) * (x_positions[1] - x_positions[0])
        
        # Plot the normal curve
        ax.plot(x, p, 'k', linewidth=2, label=f'Normal fit\nμ={mu:.1f}, σ={std:.1f}')
        
        # Draw vertical line at mu (the mean)
        ax.axvline(mu, color='red', linestyle=':', linewidth=2, label='Mean')
        
        # Also draw the trimmed mean
        ax.axvline(mu_trimmed, color='green', linestyle='--', linewidth=1.5, 
                  label=f'Trimmed mean ({mu_trimmed:.1f})')
        
        # Set the x labels as integers to match bar positions
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(int(i)) for i in x_positions])
        
        # Set proper x-axis limits to show the full normal curve
        ax.set_xlim(min_x - 0.5, max_x + 0.5)
        
        ax.set_title(f"{gym}: Distribution of Topped Boulders (Excl. 0)")
        ax.set_xlabel("Number of Boulders Topped")
        ax.set_ylabel("Frequency")
        
        # Use a compact legend to avoid overlapping
        ax.legend(fontsize='small', loc='best')
        ax.grid(True, linestyle='--', linewidth=0.5)
    
    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def main(gender: str = 'men') -> None:
    """
    Main entry point for running statistical analysis from command line.
    
    Args:
        gender: Gender category to analyze ('men' or 'women'). Defaults to 'men'.
    """
    data = load_results(gender=gender)
    gym_boulder_counts, completion_histograms, participation_counts = compute_gym_stats(data)
    print_gym_stats(gym_boulder_counts, completion_histograms, participation_counts)
    plot_boulder_popularity(gym_boulder_counts, completion_histograms)
    plot_gym_completion_distributions(completion_histograms, gender=gender)


if __name__ == "__main__":
    main()


# ==============================================================================
# FRENCH GRADING SYSTEM INTEGRATION
# ==============================================================================

def compute_boulder_grades(data: List[Dict[str, Any]], gender: str = 'men') -> Optional[FrenchGradingSystem]:
    """
    Compute French boulder grades for all gyms using the completion rate method.
    
    UPDATED: Always uses men's division data for grading calibration to ensure consistent
    grading across all divisions (men's, women's, and combined). This ensures that
    a 6c boulder is graded as 6c regardless of which division is being analyzed.
    
    Args:
        data: List of climber dictionaries from the competition
        gender: Gender category being analyzed (used for display/logging only)
    
    Returns:
        FrenchGradingSystem instance with calculated grades, or None if grading system unavailable
    """
    if not GRADING_SYSTEM_AVAILABLE:
        logger.warning("Grading system not available. Cannot compute boulder grades.")
        return None
    
    # UPDATED: Always use men's division data for grading calibration
    # This ensures consistent absolute grading across all divisions
    try:
        # Load men's division data for grading calibration
        mens_data = load_results(gender='men')
        logger.info(f"Using men's division data for grading calibration (analyzing {gender} division)")
        
        # Compute gym statistics from men's division data
        gym_boulder_counts, completion_histograms, participation_counts = compute_gym_stats(mens_data)
        
    except Exception as e:
        logger.warning(f"Could not load men's division data for grading: {e}")
        logger.info(f"Falling back to {gender} division data for grading")
        # Fallback to provided data if men's data unavailable
        gym_boulder_counts, completion_histograms, participation_counts = compute_gym_stats(data)
    
    # Initialize and configure the grading system with calibration data
    grading_system = initialize_grading_system_with_known_data(gym_boulder_counts, participation_counts)
    
    logger.info(f"Computed French boulder grades for {len(grading_system.boulder_grades)} gyms (calibrated on men's division)")
    return grading_system


def get_gym_grade_summary(grading_system: FrenchGradingSystem, gym: str) -> Dict[str, Any]:
    """
    Get a summary of boulder grades for a specific gym.
    
    Args:
        grading_system: Configured FrenchGradingSystem instance
        gym: Name of the gym
    
    Returns:
        Dictionary containing grade distribution and statistics
    """
    if gym not in grading_system.boulder_grades:
        return {}
    
    boulders = grading_system.boulder_grades[gym]
    grade_distribution = grading_system.get_gym_grade_distribution(gym)
    
    # Calculate statistics
    grades_numeric = [b.grade_numeric for b in boulders.values()]
    completion_rates = [b.completion_rate for b in boulders.values()]
    
    summary = {
        'gym': gym,
        'total_boulders': len(boulders),
        'grade_distribution': grade_distribution,
        'difficulty_factor': grading_system.gym_difficulty_factors.get(gym, 1.0),
        'average_grade_numeric': np.mean(grades_numeric) if grades_numeric else 0,
        'hardest_boulder': max(boulders.values(), key=lambda b: b.grade_numeric) if boulders else None,
        'easiest_boulder': min(boulders.values(), key=lambda b: b.grade_numeric) if boulders else None,
        'average_completion_rate': np.mean(completion_rates) if completion_rates else 0,
        'grade_range': {
            'min': min(grade_distribution.keys(), key=lambda g: grading_system.FRENCH_GRADES.get(g, 0)) if grade_distribution else None,
            'max': max(grade_distribution.keys(), key=lambda g: grading_system.FRENCH_GRADES.get(g, 0)) if grade_distribution else None
        }
    }
    
    return summary


def print_grading_analysis(grading_system: FrenchGradingSystem) -> None:
    """
    Print comprehensive grading analysis to console.
    
    Args:
        grading_system: Configured FrenchGradingSystem instance
    """
    print("\n" + "="*80)
    print("FRENCH BOULDER GRADING ANALYSIS")
    print("="*80)
    
    # Print the full report
    report = grading_system.generate_grading_report()
    print(report)
    
    print("\n" + "-"*60)
    print("DETAILED GYM SUMMARIES")
    print("-"*60)
    
    for gym in sorted(grading_system.boulder_grades.keys()):
        summary = get_gym_grade_summary(grading_system, gym)
        
        print(f"\n📍 {gym}")
        print(f"   Total boulders: {summary['total_boulders']}")
        print(f"   Difficulty factor: {summary['difficulty_factor']:.2f}")
        print(f"   Average grade: {summary['average_grade_numeric']:.1f}")
        print(f"   Grade range: {summary['grade_range']['min']} - {summary['grade_range']['max']}")
        print(f"   Average completion rate: {summary['average_completion_rate']:.1%}")
        
        if summary['hardest_boulder']:
            hardest = summary['hardest_boulder']
            print(f"   Hardest boulder: #{hardest.boulder_id} ({hardest.french_grade}, {hardest.completion_rate:.1%} completion)")
        
        if summary['easiest_boulder']:
            easiest = summary['easiest_boulder']
            print(f"   Easiest boulder: #{easiest.boulder_id} ({easiest.french_grade}, {easiest.completion_rate:.1%} completion)")


def plot_grade_distributions(grading_system: FrenchGradingSystem, gyms: Optional[List[str]] = None) -> None:
    """
    Plot grade distributions for gyms with matplotlib.
    
    Args:
        grading_system: Configured FrenchGradingSystem instance
        gyms: List of gym names to plot. If None, plots all gyms.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.error("Matplotlib not available for plotting grade distributions")
        return
    
    if gyms is None:
        gyms = sorted(grading_system.boulder_grades.keys())
    
    # Prepare data
    all_grades = list(grading_system.FRENCH_GRADES.keys())
    grade_order = sorted(all_grades, key=lambda g: grading_system.FRENCH_GRADES[g])
    
    n_gyms = len(gyms)
    ncols = 2
    nrows = (n_gyms + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows))
    if n_gyms == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, gym in enumerate(gyms):
        distribution = grading_system.get_gym_grade_distribution(gym)
        
        # Prepare data for plotting
        grades_present = []
        counts = []
        
        for grade in grade_order:
            if grade in distribution:
                grades_present.append(grade)
                counts.append(distribution[grade])
        
        if not grades_present:
            continue
        
        ax = axes[idx]
        bars = ax.bar(grades_present, counts, alpha=0.7, color='steelblue')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f"Grade Distribution - {gym}\n"
                    f"(Difficulty Factor: {grading_system.gym_difficulty_factors.get(gym, 1.0):.2f})")
        ax.set_xlabel("French Grade")
        ax.set_ylabel("Number of Boulders")
        ax.grid(True, axis='y', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    # Remove unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def analyze_grade_correlations(grading_system: FrenchGradingSystem) -> Dict[str, float]:
    """
    Analyze correlations between completion rates and assigned grades.
    
    Args:
        grading_system: Configured FrenchGradingSystem instance
    
    Returns:
        Dictionary containing correlation statistics
    """
    all_completion_rates = []
    all_grade_numerics = []
    
    for gym, boulders in grading_system.boulder_grades.items():
        for boulder in boulders.values():
            all_completion_rates.append(boulder.completion_rate)
            all_grade_numerics.append(boulder.grade_numeric)
    
    if len(all_completion_rates) < 2:
        return {}
    
    # Calculate correlations
    correlation_completion_grade = np.corrcoef(all_completion_rates, all_grade_numerics)[0, 1]
    
    # Calculate R-squared
    r_squared = correlation_completion_grade ** 2
    
    return {
        'completion_rate_vs_grade_correlation': correlation_completion_grade,
        'r_squared': r_squared,
        'sample_size': len(all_completion_rates),
        'grade_range': (min(all_grade_numerics), max(all_grade_numerics)),
        'completion_rate_range': (min(all_completion_rates), max(all_completion_rates))
    }


def main_with_grading(gender: str = 'men') -> None:
    """
    Enhanced main function that includes grading analysis.
    
    Args:
        gender: Gender category to analyze ('men' or 'women'). Defaults to 'men'.
    """
    print(f"Loading {gender} competition data...")
    data = load_results(gender=gender)
    
    # Standard statistical analysis
    gym_boulder_counts, completion_histograms, participation_counts = compute_gym_stats(data)
    print_gym_stats(gym_boulder_counts, completion_histograms, participation_counts)
    
    # French grading analysis
    grading_system = compute_boulder_grades(data, gender)
    
    if grading_system:
        print_grading_analysis(grading_system)
        
        # Export grades
        grading_system.export_grades_to_json(f"boulder_grades_{gender}.json")
        
        # Analyze correlations
        correlations = analyze_grade_correlations(grading_system)
        if correlations:
            print(f"\n📊 GRADING SYSTEM VALIDATION")
            print(f"   Completion rate vs grade correlation: {correlations['completion_rate_vs_grade_correlation']:.3f}")
            print(f"   R-squared: {correlations['r_squared']:.3f}")
            print(f"   Sample size: {correlations['sample_size']} boulders")
        
        # Generate plots
        plot_boulder_popularity(gym_boulder_counts, completion_histograms)
        plot_gym_completion_distributions(completion_histograms, gender)
        plot_grade_distributions(grading_system)
    
    else:
        # Fallback to standard plots if grading system unavailable
        plot_boulder_popularity(gym_boulder_counts, completion_histograms)
        plot_gym_completion_distributions(completion_histograms, gender)
