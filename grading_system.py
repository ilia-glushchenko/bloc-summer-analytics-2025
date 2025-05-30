"""
French Boulder Grading System Implementation

This module implements a calibration-point anchored grading system that assigns
French boulder grades (5a-8a) based on completion rates, using known grades
as calibration points for accuracy.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GradeCalibration:
    """Stores calibration data for a specific boulder with known grade."""
    boulder_id: str
    gym: str
    french_grade: str
    completion_rate: float
    total_climbers: int
    completed_count: int

@dataclass
class BoulderGrade:
    """Stores grading information for a boulder."""
    boulder_id: str
    gym: str
    french_grade: str
    confidence: float
    completion_rate: float
    total_climbers: int
    completed_count: int
    grade_numeric: float  # Numeric representation for calculations

class FrenchGradingSystem:
    """
    Implements French boulder grading system based on completion rates.
    
    CORRECTED APPROACH: Uses calibration points (boulders with known grades) to establish
    absolute difficulty grades. Boulder difficulty is treated as absolute - a 6c+ is 6c+
    everywhere, regardless of gym. Different completion rates at different gyms reflect
    population strength differences, not boulder difficulty differences.
    
    UNIFIED GRADING: Always uses men's division data for calibration across all divisions
    (men's, women's, and combined) to ensure consistent absolute grading. This means
    a boulder graded as 6c will be 6c for all divisions.
    
    Uses Boulder 31 at Boulderbar Wienerberg (6c) as primary calibration point from men's division.
    """
    
    # French grade scale mapping to numeric values
    FRENCH_GRADES = {
        '5a': 1, '5b': 2, '5c': 3,  # No + versions in 5th grade
        '6a': 4, '6a+': 4.5, '6b': 5, '6b+': 5.5, '6c': 6, '6c+': 6.5,
        '7a': 7, '7a+': 7.5, '7b': 8, '7b+': 8.5, '7c': 9, '7c+': 9.5,
        '8a': 10, '8a+': 10.5, '8b': 11, '8b+': 11.5
    }
    
    # Reverse mapping for numeric to grade conversion
    NUMERIC_TO_GRADE = {v: k for k, v in FRENCH_GRADES.items()}
    
    def __init__(self):
        self.calibration_points: List[GradeCalibration] = []
        self.boulder_grades: Dict[str, Dict[str, BoulderGrade]] = defaultdict(dict)
        self.gym_difficulty_factors: Dict[str, float] = {}
        
    def add_calibration_point(self, boulder_id: str, gym: str, french_grade: str, 
                            completion_rate: float, total_climbers: int, completed_count: int) -> None:
        """Add a known grade calibration point."""
        calibration = GradeCalibration(
            boulder_id=boulder_id,
            gym=gym,
            french_grade=french_grade,
            completion_rate=completion_rate,
            total_climbers=total_climbers,
            completed_count=completed_count
        )
        self.calibration_points.append(calibration)
        logger.info(f"Added calibration point: Boulder {boulder_id} ({gym}) = {french_grade} "
                   f"({completion_rate:.1%} completion rate)")
    
    def _completion_rate_to_grade_numeric(self, completion_rate: float, reference_gym_name: str = "Boulderbar Wienerberg") -> float:
        """
        Convert completion rate to numeric grade value using calibration points.
        
        For the reference gym (Boulderbar Wienerberg), it uses interpolation based on:
        - Target CR for 5a (e.g., 0.90) -> Numeric Grade 1.0
        - Boulder 31's CR -> Numeric Grade 6.0 (for 6c)
        - Target CR for 8a (e.g., 0.03) -> Numeric Grade 10.0
        
        For other gyms, it currently relies on the reference gym's calculation after
        an equivalent CR conversion (which is a separate area for improvement).

        Real-world constraints:
        - Boulderbar Wienerberg: max grade 8a (10.0 in numeric scale)
        - Boulder 31 at Boulderbar Wienerberg: 6c (6.0 in numeric scale)
        - Full range for Boulderbar Wienerberg: 5a to 8a (1.0 to 10.0 in numeric scale)
        """
        REFERENCE_GYM_MAX_GRADE = 10.0  # 8a for Boulderbar Wienerberg
        REFERENCE_GYM_MIN_GRADE = 1.0   # 5a minimum

        # Target completion rates for grade boundaries at Boulderbar Wienerberg
        # These can be fine-tuned based on data analysis or expert judgment.
        TARGET_CR_5A = 0.90  # Estimated CR for a solid 5a
        TARGET_CR_8A = 0.03  # Estimated CR for a solid 8a

        # Numeric grades corresponding to these points
        NUMERIC_GRADE_5A = 1.0
        NUMERIC_GRADE_6C = 6.0
        NUMERIC_GRADE_8A = 10.0

        if reference_gym_name == "Boulderbar Wienerberg":
            # Find the specific calibration point for Boulder 31
            boulder_31_calib = None
            for cp in self.calibration_points:
                if cp.boulder_id == "31" and cp.gym == "Boulderbar Wienerberg":
                    boulder_31_calib = cp
                    break
            
            if boulder_31_calib:
                cr_boulder_31 = boulder_31_calib.completion_rate

                # Ensure completion rates are within a plausible range to avoid division by zero
                # or extreme values in interpolation.
                # Clamp completion_rate to be between TARGET_CR_8A and TARGET_CR_5A for interpolation.
                # This prevents extrapolation beyond the defined anchor points.

                if completion_rate >= TARGET_CR_5A:
                    estimated_grade = NUMERIC_GRADE_5A
                elif completion_rate <= TARGET_CR_8A:
                    estimated_grade = NUMERIC_GRADE_8A
                elif completion_rate >= cr_boulder_31: # Between Boulder 31 and 5a
                    # Interpolate between (cr_boulder_31, NUMERIC_GRADE_6C) and (TARGET_CR_5A, NUMERIC_GRADE_5A)
                    # Avoid division by zero if cr_boulder_31 is very close to TARGET_CR_5A
                    if TARGET_CR_5A - cr_boulder_31 > 1e-6: # Small epsilon to prevent division by zero
                        estimated_grade = (NUMERIC_GRADE_6C +
                                          (completion_rate - cr_boulder_31) *
                                          (NUMERIC_GRADE_5A - NUMERIC_GRADE_6C) /
                                          (TARGET_CR_5A - cr_boulder_31))
                    else: # If CRs are too close, assign the lower grade of the segment
                        estimated_grade = NUMERIC_GRADE_5A if completion_rate >= (TARGET_CR_5A + cr_boulder_31) / 2 else NUMERIC_GRADE_6C

                else: # Between 8a and Boulder 31 (completion_rate < cr_boulder_31)
                    # Interpolate between (TARGET_CR_8A, NUMERIC_GRADE_8A) and (cr_boulder_31, NUMERIC_GRADE_6C)
                    # Avoid division by zero if cr_boulder_31 is very close to TARGET_CR_8A
                    if cr_boulder_31 - TARGET_CR_8A > 1e-6: # Small epsilon
                        estimated_grade = (NUMERIC_GRADE_8A +
                                          (completion_rate - TARGET_CR_8A) *
                                          (NUMERIC_GRADE_6C - NUMERIC_GRADE_8A) /
                                          (cr_boulder_31 - TARGET_CR_8A))
                    else: # If CRs are too close, assign the lower grade of the segment
                        estimated_grade = NUMERIC_GRADE_6C if completion_rate >= (cr_boulder_31 + TARGET_CR_8A) / 2 else NUMERIC_GRADE_8A
                
                # Ensure the grade is within the absolute min/max for the reference gym
                estimated_grade = max(REFERENCE_GYM_MIN_GRADE, min(REFERENCE_GYM_MAX_GRADE, estimated_grade))
                return estimated_grade
            else:
                # Fallback if Boulder 31 calibration point is missing for some reason
                logger.warning("Boulder 31 calibration point missing for Boulderbar Wienerberg. Using basic fallback.")
                # Fallback to the original simple mapping (or a refined version of it)
                # This part should ideally not be reached if calibration data is loaded correctly.
                if completion_rate >= 0.80: return 1.0   # 5a
                elif completion_rate >= 0.70: return 2.0   # 5b
                elif completion_rate >= 0.55: return 3.0   # 5c
                elif completion_rate >= 0.40: return 4.0   # 6a
                elif completion_rate >= 0.30: return 5.0   # 6b
                elif completion_rate >= 0.20: return 6.0   # 6c
                elif completion_rate >= 0.10: return 8.0   # 7b (Skipping 7a for broader steps in fallback)
                elif completion_rate >= 0.05: return 9.0   # 7c
                else: return REFERENCE_GYM_MAX_GRADE      # 8a
        
        # For gyms other than Boulderbar Wienerberg, or if Boulderbar Wienerberg has no calibration point.
        # This section handles non-reference gyms (delegating to reference logic after CR conversion)
        # OR the fallback scenario for the reference gym if calibration data is missing.
        
        # If it's not the reference_gym_name, it means this function was called for another gym,
        # where the completion_rate passed is already an *equivalent* reference gym CR.
        # So, we still need a way to grade it if the reference gym's calibration (Boulder 31) is missing.
        # The current logic path for other gyms in `_calculate_gym_grades_relative_to_reference`
        # passes `reference_gym_name` ("Boulderbar Wienerberg") to this function,
        # so this fallback below is primarily for the case where Boulder 31 data is absent *even for the reference gym*.
        
        # Simplified fallback (original logic) if no calibration point or not reference gym
        # This is the path taken if Boulder 31 calibration is not found or if `reference_gym_name`
        # passed to this function is not "Boulderbar Wienerberg" (though current calls make it so).
        if not self.calibration_points: # General fallback if NO calibration points exist AT ALL
            logger.warning("No calibration points available. Using broad fallback grading.")
            if completion_rate >= 0.80: return 1.0   # 5a
            elif completion_rate >= 0.70: return 2.0   # 5b
            elif completion_rate >= 0.55: return 3.0   # 5c
            elif completion_rate >= 0.40: return 4.0   # 6a
            elif completion_rate >= 0.30: return 5.0   # 6b
            elif completion_rate >= 0.20: return 6.0   # 6c
            elif completion_rate >= 0.10: return 8.0   # 7b
            elif completion_rate >= 0.05: return 9.0   # 7c
            else: return REFERENCE_GYM_MAX_GRADE      # 8a

        # If calibration points exist, but we are not in the "Boulderbar Wienerberg" specific logic path
        # (e.g. Boulder 31 not found, or if this function were called with a different reference_gym_name)
        # then use the old primary calibration logic (logarithmic) as a more general fallback.
        # This path should ideally be hit less often with the new structure.
        
        # Use the first available calibration point as anchor if not using specific B31 logic
        # This is effectively the old logic for when Boulder 31 wasn't specially handled or for other gyms directly.
        calibration = self.calibration_points[0]
        base_completion_rate = calibration.completion_rate
        base_grade = self.FRENCH_GRADES[calibration.french_grade]

        if base_completion_rate <= 0 or completion_rate <= 0: # Avoid math errors
            # If completion_rate is 0, it's very hard. If base_completion_rate is 0, calibration is problematic.
            # Default to a hard grade or base_grade.
            return max(base_grade, NUMERIC_GRADE_8A) if completion_rate == 0 else base_grade

        completion_ratio = completion_rate / base_completion_rate
        estimated_grade = base_grade

        if completion_ratio > 0:
            grade_adjustment = -np.log2(completion_ratio) * 1.5 
            estimated_grade = base_grade + grade_adjustment
            
            if completion_rate > base_completion_rate: # Easier than this general calibration point
                ease_factor = (completion_rate - base_completion_rate) / (1.0 - base_completion_rate + 1e-6) # Avoid div by zero
                extra_adjustment = ease_factor * 3.0
                estimated_grade = estimated_grade - extra_adjustment
                
                if completion_rate >= 0.85: estimated_grade = 1.0
                elif completion_rate >= 0.75: estimated_grade = 2.0
                elif completion_rate >= 0.65: estimated_grade = 3.0
        
        # Apply general constraints based on whether it's the primary reference gym or another
        # The `reference_gym_name` argument to this function specifies which context we are in.
        if reference_gym_name == "Boulderbar Wienerberg":
            estimated_grade = max(REFERENCE_GYM_MIN_GRADE, min(REFERENCE_GYM_MAX_GRADE, estimated_grade))
        else: # For other gyms, allow slightly wider range if they are intrinsically harder/easier
            estimated_grade = max(1.0, min(11.0, estimated_grade)) # 8b max for others
        
        return estimated_grade
    
    def _numeric_to_french_grade(self, numeric_grade: float) -> str:
        """Convert numeric grade value to French grade string."""
        # Find closest French grade
        closest_numeric = min(self.NUMERIC_TO_GRADE.keys(), 
                            key=lambda x: abs(x - numeric_grade))
        return self.NUMERIC_TO_GRADE[closest_numeric]
    
    def calculate_boulder_grades(self, gym_boulder_counts: Dict[str, Dict[str, int]], 
                               participation_counts: Dict[str, int]) -> None:
        """
        Calculate French grades for all boulders based on completion rates using absolute difficulty.
        
        CORRECTED APPROACH: Uses Boulderbar Wienerberg as calibration reference to establish
        the completion rate to grade mapping, then applies the same absolute grading scale
        to all gyms. Boulder difficulty is absolute - different completion rates at different
        gyms reflect population differences, not boulder difficulty differences.
        
        Args:
            gym_boulder_counts: Boulder completion counts per gym
            participation_counts: Total participants per gym
        """
        logger.info("Calculating boulder grades using absolute difficulty scale...")
        
        reference_gym = "Boulderbar Wienerberg"
        reference_gym_max_grade = 10.0  # 8a constraint
        
        # Step 1: Calculate grades for reference gym (Boulderbar Wienerberg) to establish calibration
        if reference_gym in gym_boulder_counts:
            self._calculate_gym_grades(reference_gym, gym_boulder_counts[reference_gym], 
                                     participation_counts.get(reference_gym, 0), 
                                     max_grade=reference_gym_max_grade)
            logger.info(f"Established calibration from reference gym {reference_gym}")
        
        # Step 2: Apply the same absolute grading scale to all other gyms
        for gym, boulder_counts in gym_boulder_counts.items():
            if gym == reference_gym:
                continue  # Already calculated
                
            total_participants = participation_counts.get(gym, 0)
            if total_participants == 0:
                logger.warning(f"No participants found for gym {gym}")
                continue
            
            # CORRECTED: Apply absolute difficulty grading (same scale for all gyms)
            self._calculate_gym_grades_relative_to_reference(
                gym, boulder_counts, total_participants, reference_gym
            )
        
        # Validate and log results
        if reference_gym in self.boulder_grades:
            ref_grades = [bg.grade_numeric for bg in self.boulder_grades[reference_gym].values()]
            logger.info(f"Reference gym {reference_gym} grade range: {min(ref_grades):.1f} - {max(ref_grades):.1f}")
        
        logger.info(f"Calculated grades for {sum(len(boulders) for boulders in self.boulder_grades.values())} boulders")
    
    def _calculate_gym_grades_relative_to_reference(self, gym: str, boulder_counts: Dict[str, int], 
                                                   total_participants: int, reference_gym: str) -> None:
        """
        Calculate grades for a gym using absolute difficulty principles.
        
        CORRECTED APPROACH: Boulder difficulty is absolute - a 6c+ boulder is 6c+ everywhere.
        We grade based on completion rates using the same scale regardless of gym,
        acknowledging that different gyms may have different populations which affect
        completion rates, but this doesn't change the boulder's inherent difficulty.
        """
        logger.info(f"Calculating {gym} grades using absolute difficulty scale")
        
        for boulder_id, completed_count in boulder_counts.items():
            completion_rate = completed_count / total_participants
            
            # CORRECTED: Use completion rate directly for grading
            # No more "equivalent Wienerberg rate" conversion - boulder difficulty is absolute
            # The completion rate at this gym reflects this boulder's difficulty with this population
            
            # Use the same grading scale for all gyms - difficulty is absolute
            numeric_grade = self._completion_rate_to_grade_numeric(completion_rate, reference_gym)
            
            # Convert to French grade
            french_grade = self._numeric_to_french_grade(numeric_grade)
            
            # Calculate confidence based on sample size
            confidence = min(1.0, completed_count / 10)
            
            # Store the grade - now consistent across all gyms for same completion rates
            boulder_grade = BoulderGrade(
                boulder_id=boulder_id,
                gym=gym,
                french_grade=french_grade,
                confidence=confidence,
                completion_rate=completion_rate,
                total_climbers=total_participants,
                completed_count=completed_count,
                grade_numeric=numeric_grade
            )
            
            self.boulder_grades[gym][boulder_id] = boulder_grade
    
    def _calculate_gym_grades(self, gym: str, boulder_counts: Dict[str, int], 
                             total_participants: int, max_grade: float = 10.0) -> None:
        """Calculate grades for the reference gym using direct completion rates."""
        for boulder_id, completed_count in boulder_counts.items():
            if total_participants == 0: # Avoid division by zero
                completion_rate = 0
            else:
                completion_rate = completed_count / total_participants
            
            # Calculate numeric grade directly using the gym's name as context
            numeric_grade = self._completion_rate_to_grade_numeric(completion_rate, gym) 
            
            # Apply max grade constraint (primarily for reference gym if different from default)
            numeric_grade = min(numeric_grade, max_grade) 
            
            # Convert to French grade
            french_grade = self._numeric_to_french_grade(numeric_grade)
            
            # Calculate confidence based on sample size
            confidence = min(1.0, completed_count / 10)
            
            # Store the grade
            boulder_grade = BoulderGrade(
                boulder_id=boulder_id,
                gym=gym,
                french_grade=french_grade,
                confidence=confidence,
                completion_rate=completion_rate,
                total_climbers=total_participants,
                completed_count=completed_count,
                grade_numeric=numeric_grade
            )
            
            self.boulder_grades[gym][boulder_id] = boulder_grade
    
    def _convert_completion_rate_to_reference(self, completion_rate: float, 
                                            source_gym: str, reference_gym: str) -> float:
        """
        Convert a completion rate from source gym to equivalent rate at reference gym.
        
        CORRECTED APPROACH: Boulder difficulty is absolute (6c+ is 6c+ everywhere).
        Completion rate differences between gyms reflect population strength differences,
        NOT boulder difficulty differences.
        
        This method now returns the original completion rate without adjustment,
        as we should grade based on absolute difficulty rather than relative completion rates.
        """
        # CORRECTION: Remove gym difficulty factors - boulder difficulty is absolute
        # Different completion rates at different gyms reflect population differences,
        # not boulder difficulty differences
        
        # For absolute grading, we use the same completion rate regardless of gym
        # The completion rate reflects how that specific boulder performs with that gym's population
        return completion_rate
        
        # OLD FLAWED LOGIC (removed):
        # if source_gym == "Boulder Monkeys":
        #     return completion_rate / 0.7  # This incorrectly assumed easier boulders
        # elif source_gym == "BigWall":
        #     return completion_rate / 0.85  # This incorrectly assumed easier boulders
        # elif source_gym == "Blockfabrik":
        #     return completion_rate * 1.4  # This incorrectly assumed harder boulders
        
        # The correct approach: A boulder with X% completion rate gets graded
        # based on what X% completion rate means for boulder difficulty,
        # regardless of which gym it's at
    
    def get_boulder_grade(self, gym: str, boulder_id: str) -> Optional[BoulderGrade]:
        """Get the grade for a specific boulder."""
        return self.boulder_grades.get(gym, {}).get(boulder_id)
    
    def get_gym_grade_distribution(self, gym: str) -> Dict[str, int]:
        """Get the distribution of grades in a gym."""
        distribution = defaultdict(int)
        
        for boulder_grade in self.boulder_grades.get(gym, {}).values():
            distribution[boulder_grade.french_grade] += 1
        
        return dict(distribution)
    
    def export_grades_to_json(self, filename: str = "boulder_grades.json") -> None:
        """Export calculated grades to JSON file."""
        export_data = {}
        
        for gym, boulders in self.boulder_grades.items():
            export_data[gym] = {}
            for boulder_id, grade in boulders.items():
                export_data[gym][boulder_id] = {
                    "french_grade": grade.french_grade,
                    "numeric_grade": grade.grade_numeric,
                    "completion_rate": grade.completion_rate,
                    "confidence": grade.confidence,
                    "total_climbers": grade.total_climbers,
                    "completed_count": grade.completed_count
                }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported grades to {filename}")
    
    def generate_grading_report(self) -> str:
        """Generate a comprehensive grading report."""
        report = ["# French Boulder Grading System Report\n"]
        
        # Calibration points
        report.append("## Calibration Points")
        for cp in self.calibration_points:
            report.append(f"- Boulder {cp.boulder_id} ({cp.gym}): {cp.french_grade} "
                         f"({cp.completion_rate:.1%} completion rate)")
        
        # Gym difficulty factors
        report.append("\n## Gym Difficulty Factors")
        for gym, factor in self.gym_difficulty_factors.items():
            difficulty = "harder" if factor > 1 else "easier" if factor < 1 else "same"
            report.append(f"- {gym}: {factor:.2f} ({difficulty} than reference)")
        
        # Grade distributions
        report.append("\n## Grade Distributions by Gym")
        for gym in sorted(self.boulder_grades.keys()):
            distribution = self.get_gym_grade_distribution(gym)
            total_boulders = sum(distribution.values())
            report.append(f"\n### {gym} ({total_boulders} boulders)")
            
            for grade in sorted(distribution.keys(), key=lambda g: self.FRENCH_GRADES.get(g, 0)):
                count = distribution[grade]
                percentage = count / total_boulders * 100 if total_boulders > 0 else 0
                report.append(f"- {grade}: {count} boulders ({percentage:.1f}%)")
        
        return "\n".join(report)


def initialize_grading_system_with_known_data(gym_boulder_counts: Dict[str, Dict[str, int]], 
                                            participation_counts: Dict[str, int]) -> FrenchGradingSystem:
    """
    Initialize the grading system with known calibration data and real-world constraints.
    
    UNIFIED GRADING: This function uses men's division data for calibration to ensure
    consistent absolute grading across all divisions (men's, women's, and combined).
    
    Real-world calibration data (from men's division):
    - Boulder 31 at Boulderbar Wienerberg = 6c
    - Boulderbar Wienerberg maximum grade = 8a (constraint)
    - Boulderbar Wienerberg grade range = 5a to 8a
    
    Args:
        gym_boulder_counts: Boulder completion counts per gym (from men's division)
        participation_counts: Total participants per gym (from men's division)
    
    Returns:
        Configured FrenchGradingSystem instance with real-world constraints applied
    """
    grading_system = FrenchGradingSystem()
    
    # Add known calibration point: Boulder 31 at Boulderbar Wienerberg = 6c
    # This must be added BEFORE calculate_boulder_grades is called.
    wienerberg_data = gym_boulder_counts.get("Boulderbar Wienerberg", {})
    wienerberg_participants = participation_counts.get("Boulderbar Wienerberg", 0)
    
    if "31" in wienerberg_data and wienerberg_participants > 0:
        completed_31 = wienerberg_data["31"]
        completion_rate_31 = completed_31 / wienerberg_participants
        
        grading_system.add_calibration_point(
            boulder_id="31",
            gym="Boulderbar Wienerberg", 
            french_grade="6c",
            completion_rate=completion_rate_31,
            total_climbers=wienerberg_participants,
            completed_count=completed_31
        )
        
        logger.info(f"Real-world constraint: Boulderbar Wienerberg max grade = 8a")
        logger.info(f"Calibration: Boulder 31 ({grading_system.FRENCH_GRADES['6c']}) = 6c with {completion_rate_31:.1%} CR.")
    else:
        logger.warning("Boulder 31 data not found or no participants at Boulderbar Wienerberg. "
                       "Grading for Boulderbar Wienerberg will rely on fallback/general calibration.")
    
    # Calculate grades for all boulders with constraints
    grading_system.calculate_boulder_grades(gym_boulder_counts, participation_counts)
    
    # Note: Difficulty factors are now applied during grade calculation,
    # no need for separate extrapolation step
    
    return grading_system 
