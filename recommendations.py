import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any

# Function to find similar climbers using gym-normalized cosine similarity
def find_similar_climbers(matrix_df: pd.DataFrame, target_climber: str, n_similar: int = 5) -> List[Tuple[str, float, int, int]]:
    """
    Find climbers with similar climbing patterns using a gym-normalized approach.
    
    This function calculates similarity between climbers on a per-gym basis,
    then combines these scores with equal weighting. This prevents bias toward
    climbers who simply visited the same gyms as the target climber.
    
    Parameters:
    -----------
    matrix_df : pd.DataFrame
        DataFrame with climbers as rows and boulders as columns (from create_climber_boulder_matrix)
    target_climber : str
        Name of the climber to find similar climbers for
    n_similar : int, default=5
        Number of similar climbers to return
        
    Returns:
    --------
    List[Tuple[str, float, int, int]]
        List of tuples with (climber_name, similarity_score, rank, completed_count)
        sorted by similarity (highest first)
    """
    # Get all boulder columns
    boulder_cols = [col for col in matrix_df.columns if '_' in col]
    
    if not boulder_cols or target_climber not in matrix_df['Climber'].values:
        return []
    
    # Group boulder columns by gym
    gym_to_boulders = defaultdict(list)
    for col in boulder_cols:
        if '_' in col:
            gym_name, _ = col.split('_', 1)
            gym_to_boulders[gym_name].append(col)
    
    # Get target climber's row for easy access
    target_row = matrix_df[matrix_df['Climber'] == target_climber].iloc[0]
    
    # Calculate similarity for each climber, ensuring balanced representation across gyms
    per_climber_similarities = {}
    
    for idx, row in matrix_df.iterrows():
        climber_name = row['Climber']
        if climber_name == target_climber:
            continue
            
        # Track gyms where both climbers have data
        common_gym_count = 0
        total_similarity = 0
        
        # Calculate similarity for each gym where both climbers have attempted boulders
        for gym, gym_boulders in gym_to_boulders.items():
            # Extract boulder data for this gym only
            target_gym_data = target_row[gym_boulders].values
            other_gym_data = row[gym_boulders].values
            
            # If either climber has no data for this gym, skip it
            if np.sum(target_gym_data) == 0 or np.sum(other_gym_data) == 0:
                continue
                
            # Calculate cosine similarity for this gym
            gym_similarity = cosine_similarity([target_gym_data], [other_gym_data])[0][0]
            total_similarity += gym_similarity
            common_gym_count += 1
        
        # Calculate the final similarity score as an average across all common gyms
        if common_gym_count > 0:
            # Scale by number of common gyms to give higher weight to climbers
            # who overlap with the target climber in multiple gyms
            final_similarity = total_similarity / common_gym_count
            scaling_factor = min(1.0, 0.5 + (common_gym_count / len(gym_to_boulders)) * 0.5)
            per_climber_similarities[climber_name] = (final_similarity * scaling_factor, row['Rank'], row['Completed'])
        else:
            # No common gyms, but still include with zero similarity for completeness
            per_climber_similarities[climber_name] = (0.0, row['Rank'], row['Completed'])
    
    # Convert to list of tuples and sort by similarity score
    similarities = [(name, sim, rank, completed) 
                   for name, (sim, rank, completed) in per_climber_similarities.items()]
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:n_similar]

# Function to calculate pure boulder completion probabilities using Bayesian approach
def calculate_bayesian_probabilities(
    target_climber: str,
    similar_climbers: List[Tuple[str, float, int, int]],
    climber_gym_boulders: Dict[str, Dict[str, List[str]]],
    gym_boulder_counts: Dict[str, Dict[str, int]],
    participation_counts: Dict[str, int]
) -> Dict[str, Dict[str, Tuple[float, float, str]]]:
    """
    Calculate completion probability for each boulder using a Bayesian approach.
    
    Parameters:
    -----------
    target_climber : str
        Name of the climber to generate probabilities for
    similar_climbers : List[Tuple[str, float, int, int]]
        List of similar climbers with their similarity scores
    climber_gym_boulders : Dict[str, Dict[str, List[str]]]
        Nested dictionary mapping climbers to gyms to completed boulder lists
    gym_boulder_counts : Dict[str, Dict[str, int]]
        Mapping of gym names to boulder numbers to completion counts
    participation_counts : Dict[str, int]
        Mapping of gym names to number of participants
        
    Returns:
    --------
    Dict[str, Dict[str, Tuple[float, float, str]]]
        Dictionary mapping gym names to boulder IDs to (probability, confidence, insight)
    """
    # Get target climber's completed boulders
    target_completed = climber_gym_boulders.get(target_climber, {})
    target_visited_gyms = set(target_completed.keys())
    
    # Dictionary to store completion probabilities and confidence for each boulder
    completion_probs = defaultdict(dict)

    # Collect all gym names from various sources to ensure we don't miss any
    all_gyms = set(gym_boulder_counts.keys()) | set(participation_counts.keys())
    for climber_data in climber_gym_boulders.values():
        all_gyms.update(climber_data.keys())
    
    # Create a mapping of which similar climbers visited which gyms
    gym_visitors = defaultdict(list)
    for climber_name, similarity, rank, _ in similar_climbers:
        climber_gyms = set(climber_gym_boulders.get(climber_name, {}).keys())
        for gym in climber_gyms:
            gym_visitors[gym].append((climber_name, similarity, rank))
    
    # Calculate global statistics to replace hardcoded constants
    # 1. Calculate global success distribution to set confidence levels
    all_success_rates = []
    for gym, boulders in gym_boulder_counts.items():
        participants = participation_counts.get(gym, 0)
        if participants > 0:
            for boulder, count in boulders.items():
                success_rate = count / participants
                all_success_rates.append(success_rate)
    
    # Calculate useful percentiles from the data
    success_quartiles = np.percentile(all_success_rates, [25, 50, 75]) if all_success_rates else [0.25, 0.5, 0.75]
    low_success, median_success, high_success = success_quartiles
    
    # 2. Calculate typical number of similar climbers who visited gyms
    gym_visitor_counts = [len(visitors) for gym, visitors in gym_visitors.items()]
    median_visitors = np.median(gym_visitor_counts) if gym_visitor_counts else 2
    max_visitors = max(gym_visitor_counts) if gym_visitor_counts else 5
    
    # 3. Calculate confidence base and scaling based on data distribution
    # The idea: confidence should start low with just prior and scale based on evidence quality
    confidence_base = 1.0 / (1.0 + len(all_gyms))  # More gyms = lower starting confidence
    evidence_max_weight = min(0.9, 1.0 - confidence_base)  # Ensure we don't go over 1.0
    
    # Process each gym
    for gym in all_gyms:
        # Get completed boulder info for this gym
        gym_boulders = gym_boulder_counts.get(gym, {})
        
        # Skip if gym has no boulders
        if not gym_boulders:
            continue
            
        # Get boulders already completed by target climber in this gym
        completed_boulders = set(target_completed.get(gym, []))
        
        # Get baseline success probabilities for each boulder in this gym
        total_participants = participation_counts.get(gym, 0)
        
        # Calculate the max number of similar climbers who could have completed boulders here
        max_possible_evidence = len(gym_visitors.get(gym, []))
        
        # Process each boulder in this gym
        for boulder, completion_count in gym_boulders.items():
            # Skip if already completed
            if boulder in completed_boulders:
                continue
            
            # BAYESIAN CALCULATION:
            # 1. Prior probability - general success rate (population-level)
            prior_p = completion_count / total_participants if total_participants > 0 else median_success
            
            # 2. Evidence from similar climbers who visited this gym
            similar_evidence = []
            visited_similar_climbers = gym_visitors.get(gym, [])
            
            for climber_name, similarity, rank in visited_similar_climbers:
                climber_completed = climber_gym_boulders.get(climber_name, {}).get(gym, [])
                # Binary outcome: 1=completed, 0=not completed
                completed = 1 if boulder in climber_completed else 0
                # Weight by similarity
                similar_evidence.append((completed, similarity))
                
            # 3. Calculate posterior probability using evidence
            posterior_p = prior_p
            
            # Base confidence depends on how many total participants attempted this boulder
            participant_ratio = min(1.0, total_participants / (np.mean(list(participation_counts.values())) if participation_counts else 10))
            confidence = confidence_base * (1 + participant_ratio)  # More participants = higher base confidence
            
            if similar_evidence:
                # Calculate weighted success rate from similar climbers
                total_weight = sum(sim for _, sim in similar_evidence)
                if total_weight > 0:
                    weighted_success = sum(comp * sim for comp, sim in similar_evidence) / total_weight
                    
                    # Adjust evidence strength based on:
                    # 1. What percentage of similar climbers provided evidence
                    evidence_coverage = len(similar_evidence) / max(1, max_possible_evidence)
                    # 2. How strong the similarity scores are (total_weight)
                    evidence_quality = total_weight / len(similar_evidence) if similar_evidence else 0
                    
                    # Combine these factors - more evidence and higher quality = stronger effect
                    evidence_strength = evidence_max_weight * (0.7 * evidence_coverage + 0.3 * evidence_quality)
                    
                    # Increase confidence based on evidence 
                    confidence += evidence_strength
                    
                    # Update posterior based on evidence strength and weighted success
                    posterior_p = prior_p * (1 - evidence_strength) + weighted_success * evidence_strength
            
            # 4. Generate insight
            insight = generate_probability_insight(
                posterior_p, confidence, prior_p,
                low_success, median_success, high_success,
                gym in target_visited_gyms, 
                bool(similar_evidence)
            )
            
            # Store probability, confidence, and insight
            completion_probs[gym][boulder] = (posterior_p, confidence, insight)
    
    return completion_probs

# Helper function to generate insights about probability calculations
def generate_probability_insight(
    probability: float, 
    confidence: float, 
    prior_p: float,
    low_success: float,
    median_success: float, 
    high_success: float, 
    climber_visited: bool, 
    has_similar_data: bool
) -> str:
    """Generate human-readable insight about how the probability was calculated."""
    
    # Base insight on probability value relative to data-derived thresholds
    if probability >= high_success:
        base = "Very likely to complete"
    elif probability >= median_success:
        base = "Good chance of completion"
    elif probability >= low_success:
        base = "Moderate challenge"
    else:
        base = "Significant challenge"
    
    # Add confidence context based on percentiles of confidence distribution
    if confidence >= 0.8:
        conf = "High confidence prediction"
    elif confidence >= 0.5:
        conf = "Moderate confidence"
    else:
        conf = "Low confidence prediction"
    
    # Add data source context
    if climber_visited and has_similar_data:
        source = "Based on your history and similar climbers"
    elif climber_visited:
        source = "Based mainly on your climbing history"
    elif has_similar_data:
        source = "Based on similar climbers' performance"
    else:
        source = "Based on general completion rates"
    
    # Combine insights
    return f"{base}; {conf}; {source}"

# Function to find optimal boulders based on pure probabilities
def find_optimal_boulders(
    matrix_df: pd.DataFrame,
    target_climber: str,
    bayesian_probs: Dict[str, Dict[str, Tuple[float, float, str]]],
    top_10_target: int,
    climber_gym_boulders: Dict[str, Dict[str, List[str]]],
    gym_boulder_counts: Dict[str, Dict[str, int]]
) -> List[Tuple[str, str, float, str]]:
    """
    Find the optimal boulders based on pure Bayesian probabilities.
    
    Parameters:
    -----------
    matrix_df : pd.DataFrame
        DataFrame with climbers as rows and boulders as columns
    target_climber : str
        Name of the climber to generate recommendations for
    bayesian_probs : Dict[str, Dict[str, Tuple[float, float, str]]]
        Dictionary mapping gym names to boulder IDs to (probability, confidence, insight)
    top_10_target : int
        Number of completed boulders needed to reach the top 10
    climber_gym_boulders : Dict[str, Dict[str, List[str]]]
        Nested dictionary mapping climbers to gyms to completed boulder lists
    gym_boulder_counts : Dict[str, Dict[str, int]]
        Mapping of gym names to boulder numbers to completion counts
        
    Returns:
    --------
    List[Tuple[str, str, float, str]]
        List of tuples with (gym_name, boulder_id, probability, insight) 
        sorted by probability (highest first)
    """
    # Target climber's current stats
    target_stats = matrix_df[matrix_df['Climber'] == target_climber]
    if target_stats.empty:
        return []
        
    current_completed = target_stats.iloc[0]['Completed']
    boulders_needed = max(0, top_10_target - current_completed)
    
    if boulders_needed <= 0:
        return []  # Already in top 10
    
    # Get target climber's completed boulders
    target_completed = climber_gym_boulders.get(target_climber, {})
    target_visited_gyms = set(target_completed.keys())
    unvisited_gyms = set(bayesian_probs.keys()) - target_visited_gyms
    
    # Create flat list of all boulders with their probabilities
    all_boulders = []
    
    # Get general stats from visited gyms to calibrate exploration boost
    visited_probs = []
    for gym in target_visited_gyms:
        for boulder, (prob, _, _) in bayesian_probs.get(gym, {}).items():
            visited_probs.append(prob)
    
    # Calculate exploration boost based on data
    # Higher exploration boost when the target climber has less experience
    if visited_probs:
        med_prob = np.median(visited_probs)
        boost_base = min(0.1, med_prob * 0.2)  # Up to 20% of median probability or max 0.1
    else:
        boost_base = 0.05  # Default if no data
    
    # First, add boulders from unvisited gyms with a data-derived boost
    for gym in unvisited_gyms:
        completed_gym_boulders = set(target_completed.get(gym, []))
        
        # Calculate gym-specific boost based on number of similar climbers who visited
        gym_popularity = len(bayesian_probs.get(gym, {}))
        total_boulder_count = sum(len(boulders) for boulders in gym_boulder_counts.values())
        gym_ratio = gym_popularity / total_boulder_count if total_boulder_count > 0 else 0.1
        
        for boulder, (probability, confidence, insight) in bayesian_probs.get(gym, {}).items():
            if boulder in completed_gym_boulders:
                continue
            
            # Boost varies by gym popularity and base probability
            # Popular gyms with better probable boulders get higher boost
            boost = boost_base * (0.5 + 0.5 * gym_ratio) * min(1.0, 1 + probability)
            boosted_probability = min(1.0, probability + boost)
            
            all_boulders.append((gym, boulder, boosted_probability, confidence, f"New gym; {insight}"))
    
    # Then add boulders from visited gyms
    for gym in target_visited_gyms:
        completed_gym_boulders = set(target_completed.get(gym, []))
        for boulder, (probability, confidence, insight) in bayesian_probs.get(gym, {}).items():
            if boulder in completed_gym_boulders:
                continue
                
            all_boulders.append((gym, boulder, probability, confidence, insight))
    
    # Sort boulders by (1) probability and (2) confidence as tiebreaker
    all_boulders.sort(key=lambda x: (x[2], x[3]), reverse=True)
    
    # Take only what we need to reach top 10
    selected_boulders = all_boulders[:boulders_needed]
    
    # Format for output with insights
    result = []
    for i, (gym, boulder, probability, confidence, insight) in enumerate(selected_boulders):
        rank = i + 1
        result.append((gym, boulder, probability, f"#{rank} highest probability; {insight}"))
    
    return result

# Function to recommend boulders based on Bayesian probability
def recommend_boulders(
    matrix_df: pd.DataFrame,
    target_climber: str,
    climber_gym_boulders: Dict[str, Dict[str, List[str]]],
    similar_climbers: List[Tuple[str, float, int, int]],
    top_10_target: int,
    gym_boulder_counts: Dict[str, Dict[str, int]],
    participation_counts: Dict[str, int]
) -> Tuple[Dict[str, List[Tuple[str, float]]], Set[str]]:
    """
    Generate personalized boulder recommendations using Bayesian probabilities.
    
    Parameters:
    -----------
    matrix_df : pd.DataFrame
        DataFrame with climbers as rows and boulders as columns
    target_climber : str
        Name of the climber to generate recommendations for
    climber_gym_boulders : Dict[str, Dict[str, List[str]]]
        Nested dictionary mapping climbers to gyms to completed boulder lists
    similar_climbers : List[Tuple[str, float, int, int]]
        List of similar climbers with their similarity scores
    top_10_target : int
        Number of completed boulders needed to reach the top 10
    gym_boulder_counts : Dict[str, Dict[str, int]]
        Mapping of gym names to boulder numbers to completion counts
    participation_counts : Dict[str, int]
        Mapping of gym names to number of participants
        
    Returns:
    --------
    Tuple[Dict[str, List[Tuple[str, float]]], Set[str]]
        Dictionary mapping gym names to lists of (boulder_num, probability) tuples, sorted by probability
        Set of gym names that the climber hasn't visited yet
    """
    # Get target climber's completed boulders and visited gyms
    target_completed = climber_gym_boulders.get(target_climber, {})
    visited_gyms = set(target_completed.keys())
    
    # Collect all gym names
    all_gyms = set(gym_boulder_counts.keys()) | set(participation_counts.keys())
    for climber_data in climber_gym_boulders.values():
        all_gyms.update(climber_data.keys())
    
    # Also extract gym names from any boulder columns in the matrix
    boulder_cols = [col for col in matrix_df.columns if '_' in col]
    matrix_gyms = set()
    for col in boulder_cols:
        if '_' in col:
            try:
                gym_name = col.split('_', 1)[0]
                matrix_gyms.add(gym_name)
            except:
                pass
    all_gyms.update(matrix_gyms)
    
    # Identify unvisited gyms
    unvisited_gyms = all_gyms - visited_gyms
    
    # 1. Calculate Bayesian probabilities for all boulders
    bayesian_probs = calculate_bayesian_probabilities(
        target_climber,
        similar_climbers,
        climber_gym_boulders,
        gym_boulder_counts,
        participation_counts
    )
    
    # Initialize recommendations dictionary - strictly with (boulder, probability) format
    recommendations = defaultdict(list)
    
    # 2. Add recommendations for each gym
    for gym, boulder_probs in bayesian_probs.items():
        # Sort boulders by probability (highest first)
        sorted_boulders = []
        for boulder, (probability, _, _) in boulder_probs.items():
            sorted_boulders.append((boulder, probability))
        
        sorted_boulders.sort(key=lambda x: x[1], reverse=True)
        
        # Add to recommendations
        recommendations[gym] = sorted_boulders
    
    # 3. Ensure all gyms are in recommendations, even empty ones
    for gym in all_gyms:
        if gym not in recommendations:
            recommendations[gym] = []
    
    # 4. Find the optimal path using Bayesian probabilities
    optimal_path = find_optimal_boulders(
        matrix_df,
        target_climber,
        bayesian_probs,
        top_10_target,
        climber_gym_boulders,
        gym_boulder_counts
    )
    
    # Store in format for UI compatibility
    if optimal_path:
        recommendations["__TOP_10_PATH__"] = [
            (f"{gym}_{boulder}", probability)
            for gym, boulder, probability, insight in optimal_path
        ]
    
    return recommendations, unvisited_gyms

# Debug functions removed - no longer needed for production use 
