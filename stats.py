"""
Statistical analysis module for Boulder Summer Series competition data.

This module provides functions for loading, processing, and analyzing climbing
competition data from the Boulder Summer Series. It includes functions for
computing statistics per gym, analyzing boulder popularity, and generating
visualization data.
"""

import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde, trim_mean
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, DefaultDict, Any, Optional, Set, Union


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
