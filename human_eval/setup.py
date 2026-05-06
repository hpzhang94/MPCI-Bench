#!/usr/bin/env python3
"""
Human Evaluation Setup Script

Selects cases for human evaluation and generates evaluator assignments.
- 50 total cases (balanced between appropriate/inappropriate)
- 5 evaluators, each evaluates 30 cases
- Each case evaluated by 3 people
"""

import json
import random
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mpci_bench.data import get_image_metadata, get_story, get_story_content, is_appropriate, is_inappropriate

# Configuration
NUM_EVALUATORS = 5
CASES_PER_EVALUATOR = 30
EVALUATORS_PER_CASE = 3
TOTAL_CASES = (NUM_EVALUATORS * CASES_PER_EVALUATOR) // EVALUATORS_PER_CASE  # 50

BENCHMARK_FILE = "mpci_bench/dataset/mpci_bench.json"
OUTPUT_DIR = "mpci_bench/human_eval/results"


def load_benchmark(file_path: str) -> List[Dict[str, Any]]:
    """Load the benchmark JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def select_cases(data: List[Dict[str, Any]], num_cases: int = TOTAL_CASES) -> List[Dict[str, Any]]:
    """
    Select balanced cases for evaluation.
    - Half appropriate, half inappropriate
    - Ensure image and story exist
    """
    appropriate_cases = []
    inappropriate_cases = []
    
    for entry in data:
        story = get_story_content(entry)
        images = get_image_metadata(entry)
        
        # Must have story and image
        if not story or not images:
            continue
        
        # Check image path exists
        img_path = None
        img_desc = None
        for img in images:
            if isinstance(img, dict):
                img_path = img.get('path') or img.get('image_path') or img.get('actual_path') or ''
                img_desc = img.get('description', '')
                break
        
        if not img_path:
            continue
        
        # Check file exists
        if not os.path.exists(img_path):
            continue
        
        # Categorize by appropriate/inappropriate
        if is_inappropriate(entry):
            inappropriate_cases.append(entry)
        elif is_appropriate(entry):
            appropriate_cases.append(entry)
    
    print(f"Found {len(appropriate_cases)} appropriate cases")
    print(f"Found {len(inappropriate_cases)} inappropriate cases")
    
    # Select balanced samples
    half = num_cases // 2
    random.seed(42)  # Reproducibility
    
    selected_appropriate = random.sample(appropriate_cases, min(half, len(appropriate_cases)))
    selected_inappropriate = random.sample(inappropriate_cases, min(half, len(inappropriate_cases)))
    
    selected = selected_appropriate + selected_inappropriate
    random.shuffle(selected)
    
    print(f"Selected {len(selected)} cases ({len(selected_appropriate)} appropriate, {len(selected_inappropriate)} inappropriate)")
    
    return selected


def assign_evaluators(cases: List[Dict[str, Any]], num_evaluators: int = NUM_EVALUATORS) -> Dict[str, Any]:
    """
    Assign cases to evaluators ensuring:
    - Each evaluator gets CASES_PER_EVALUATOR cases
    - Each case is assigned to EVALUATORS_PER_CASE evaluators
    
    Uses a balanced round-robin approach to distribute cases evenly.
    """
    num_cases = len(cases)
    
    # Track assignments
    case_to_evaluators = defaultdict(list)  # case_id -> [evaluator_ids]
    evaluator_to_cases = defaultdict(list)  # evaluator_id -> [case_ids]
    
    # Create a pool of (evaluator_id, slot) pairs
    # Each evaluator has CASES_PER_EVALUATOR slots
    slots = []
    for eval_id in range(num_evaluators):
        for _ in range(CASES_PER_EVALUATOR):
            slots.append(eval_id)
    
    random.shuffle(slots)
    
    # Assign slots to cases (each case needs EVALUATORS_PER_CASE slots)
    slot_idx = 0
    for case_id in range(num_cases):
        assigned = set()
        attempts = 0
        
        while len(assigned) < EVALUATORS_PER_CASE and slot_idx < len(slots) and attempts < len(slots):
            eval_id = slots[slot_idx]
            slot_idx += 1
            
            if eval_id in assigned:
                # Put it back at end for another case
                slots.append(eval_id)
                attempts += 1
                continue
            
            assigned.add(eval_id)
            case_to_evaluators[case_id].append(eval_id)
            evaluator_to_cases[eval_id].append(case_id)
        
        # If we couldn't assign enough evaluators, try filling from evaluators with room
        while len(assigned) < EVALUATORS_PER_CASE:
            for eval_id in range(num_evaluators):
                if eval_id not in assigned and len(evaluator_to_cases[eval_id]) < CASES_PER_EVALUATOR:
                    assigned.add(eval_id)
                    case_to_evaluators[case_id].append(eval_id)
                    evaluator_to_cases[eval_id].append(case_id)
                    if len(assigned) >= EVALUATORS_PER_CASE:
                        break
            break  # Avoid infinite loop
    
    # Build assignments
    assignments = {
        "cases": [],
        "evaluators": {},
        "case_assignments": {}
    }
    
    # Add case data
    for i, entry in enumerate(cases):
        case_data = {
            "case_id": i,
            "name": entry.get('name', ''),
            "seed": entry.get('seed', {}),
            "story": get_story(entry),
            "img_metadata": get_image_metadata(entry),
            "evaluators": case_to_evaluators[i]
        }
        assignments["cases"].append(case_data)
        assignments["case_assignments"][i] = case_to_evaluators[i]
    
    # Add evaluator assignments
    for eval_id in range(num_evaluators):
        eval_name = f"evaluator_{eval_id + 1}"
        assignments["evaluators"][eval_name] = {
            "id": eval_id,
            "case_ids": evaluator_to_cases[eval_id],
            "num_cases": len(evaluator_to_cases[eval_id])
        }
    
    return assignments


def generate_evaluation_questions() -> List[Dict[str, Any]]:
    """Generate the three evaluation questions based on validate_benchmark.py criteria."""
    
    questions = [
        {
            "id": "q1_semantic_utility",
            "title": "Semantic Utility",
            "question": "Is the image integral to the story and task?",
            "description": "Evaluate whether the image provides necessary and relevant visual information that supports the story's goal.",
            "scoring_guide": {
                5: "The image is absolutely necessary and integral to the story. The story explicitly depends on specific image content.",
                4: "The image is important and meaningfully contributes to the story. It adds substantial value beyond text alone.",
                3: "The image supports the story but is not essential. The story could be understood without it.",
                2: "The image has a weak connection to the story. It feels decorative rather than functional.",
                1: "The image is completely unnecessary or unrelated to the story."
            }
        },
        {
            "id": "q2_contextual_appropriateness",
            "title": "Contextual Appropriateness",
            "question": "Does sharing the image align with the labeled scenario type (appropriate/inappropriate)?",
            "description": "Evaluate whether sharing the visual information in this story is consistent with the scenario label based on Contextual Integrity norms.",
            "scoring_guide_appropriate": {
                5: "Sharing is clearly appropriate - context perfectly justifies sharing; no privacy concerns.",
                4: "Sharing is appropriate with minor concerns - context generally justifies sharing.",
                3: "Appropriateness is unclear or ambiguous - some concerns but not definitive.",
                2: "Sharing seems somewhat inappropriate - concerns outweigh justifications.",
                1: "Sharing is clearly inappropriate - contradicts the 'appropriate' label."
            },
            "scoring_guide_inappropriate": {
                5: "Sharing clearly violates privacy norms - strongly matches 'inappropriate' label.",
                4: "Sharing is inappropriate with clear norm violations.",
                3: "Privacy violation is ambiguous or indirect.",
                2: "Only weak violations of contextual norms.",
                1: "No meaningful privacy violation - contradicts 'inappropriate' label."
            }
        },
        {
            "id": "q3_story_realism",
            "title": "Story Realism",
            "question": "Is the story natural and realistic?",
            "description": "Evaluate the overall quality of the story based on its naturalness and plausibility as a real-world scenario.",
            "scoring_guide": {
                5: "The story is clear, natural, and reads like a real scenario. Highly plausible and coherent.",
                4: "The story reads well with only minor issues. Generally coherent and believable.",
                3: "The story is understandable but has some awkward elements. Moderately realistic.",
                2: "The story has noticeable issues with flow or realism. Several awkward elements.",
                1: "The story feels artificial or AI-generated. Unrealistic or incoherent."
            }
        }
    ]
    
    return questions


def main():
    """Main function to set up human evaluation."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load benchmark data
    print(f"Loading benchmark from {BENCHMARK_FILE}...")
    data = load_benchmark(BENCHMARK_FILE)
    print(f"Loaded {len(data)} entries")
    
    # Select cases
    print(f"\nSelecting {TOTAL_CASES} cases for evaluation...")
    selected_cases = select_cases(data, TOTAL_CASES)
    
    if len(selected_cases) < TOTAL_CASES:
        print(f"Warning: Could only select {len(selected_cases)} cases (target: {TOTAL_CASES})")
    
    # Assign evaluators
    print(f"\nAssigning cases to {NUM_EVALUATORS} evaluators...")
    assignments = assign_evaluators(selected_cases)
    
    # Add evaluation questions
    assignments["questions"] = generate_evaluation_questions()
    
    # Save assignments
    output_file = os.path.join(OUTPUT_DIR, "evaluation_assignments.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(assignments, f, indent=2, ensure_ascii=False)
    print(f"\nSaved assignments to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("HUMAN EVALUATION SETUP SUMMARY")
    print("=" * 60)
    print(f"Total cases: {len(selected_cases)}")
    print(f"Evaluators: {NUM_EVALUATORS}")
    print(f"Cases per evaluator: {CASES_PER_EVALUATOR}")
    print(f"Evaluators per case: {EVALUATORS_PER_CASE}")
    print()
    
    for eval_name, eval_info in assignments["evaluators"].items():
        print(f"  {eval_name}: {eval_info['num_cases']} cases")
    
    print()
    print(f"Run the evaluation app with: python mpci_bench/human_eval/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
