#!/usr/bin/env python3
"""
Test script for the French Boulder Grading System

This script tests the grading system implementation with real competition data
and generates a comprehensive report of the results.
"""

import sys
import json
from stats import main_with_grading, load_results

def test_grading_system():
    """Test the grading system with real data."""
    print("="*80)
    print("FRENCH BOULDER GRADING SYSTEM TEST")
    print("="*80)
    
    try:
        # Test with men's data
        print("\n🚹 TESTING WITH MEN'S DATA")
        print("-" * 40)
        main_with_grading(gender='men')
        
        print("\n🚺 TESTING WITH WOMEN'S DATA")
        print("-" * 40)
        main_with_grading(gender='women')
        
        print("\n✅ GRADING SYSTEM TEST COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print(f"\n❌ GRADING SYSTEM TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def quick_boulder_31_check():
    """Quick check to verify Boulder 31 data at Boulderbar Wienerberg."""
    print("\n🔍 QUICK CHECK: Boulder 31 at Boulderbar Wienerberg")
    print("-" * 50)
    
    try:
        from grading_system import initialize_grading_system_with_known_data
        from stats import compute_gym_stats
        
        # Load data
        data = load_results(gender='men')
        gym_boulder_counts, completion_histograms, participation_counts = compute_gym_stats(data)
        
        # Check Boulderbar Wienerberg data
        wienerberg_data = gym_boulder_counts.get("Boulderbar Wienerberg", {})
        wienerberg_participants = participation_counts.get("Boulderbar Wienerberg", 0)
        
        if "31" in wienerberg_data:
            completed_31 = wienerberg_data["31"]
            completion_rate_31 = completed_31 / wienerberg_participants if wienerberg_participants > 0 else 0
            
            print(f"Boulder 31 completion data:")
            print(f"  Completed by: {completed_31} climbers")
            print(f"  Total participants: {wienerberg_participants}")
            print(f"  Completion rate: {completion_rate_31:.1%}")
            print(f"  Known grade: 6c+")
            
            # Initialize grading system
            grading_system = initialize_grading_system_with_known_data(gym_boulder_counts, participation_counts)
            
            # Get the calculated grade for Boulder 31
            boulder_31_grade = grading_system.get_boulder_grade("Boulderbar Wienerberg", "31")
            if boulder_31_grade:
                print(f"  Calculated grade: {boulder_31_grade.french_grade}")
                print(f"  Confidence: {boulder_31_grade.confidence:.2f}")
            
        else:
            print("Boulder 31 not found in Boulderbar Wienerberg data")
            print("Available boulders:", sorted(wienerberg_data.keys()) if wienerberg_data else "None")
        
    except Exception as e:
        print(f"Quick check failed: {str(e)}")

if __name__ == "__main__":
    quick_boulder_31_check()
    test_grading_system() 
