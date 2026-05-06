#!/usr/bin/env python3
"""
Human Evaluation Web Application

Flask-based web interface for human evaluation of benchmark cases.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import base64

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(BENCHMARK_ROOT))

from mpci_bench.data import get_image_metadata, get_story, is_inappropriate

app = Flask(__name__, template_folder=str(BENCHMARK_ROOT / 'templates'), static_folder='static')
app.secret_key = 'vlens-human-eval-secret-key-2024'

# Configuration
EVAL_DIR = "mpci_bench/human_eval/results"
ASSIGNMENTS_FILE = os.path.join(EVAL_DIR, "evaluation_assignments.json")
RESULTS_FILE = os.path.join(EVAL_DIR, "evaluation_results.json")
def load_assignments() -> Dict[str, Any]:
    """Load evaluation assignments."""
    if not os.path.exists(ASSIGNMENTS_FILE):
        return {"error": "Assignments file not found. Run mpci_bench/human_eval/setup.py first."}
    
    with open(ASSIGNMENTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_results() -> Dict[str, Any]:
    """Load existing evaluation results."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"evaluations": {}}


def save_results(results: Dict[str, Any]):
    """Save evaluation results."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_image_as_base64(image_path: str) -> Optional[str]:
    """Load an image file and encode it as base64."""
    full_path = BENCHMARK_ROOT / image_path
    if full_path.exists():
        try:
            with open(full_path, 'rb') as f:
                data = f.read()
                return base64.b64encode(data).decode('utf-8')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
    return None


def get_case_data(case_id: int) -> Optional[Dict[str, Any]]:
    """Get case data by ID."""
    assignments = load_assignments()
    if "error" in assignments:
        return None
    
    for case in assignments.get("cases", []):
        if case["case_id"] == case_id:
            return case
    return None


def get_evaluator_progress(evaluator_id: str) -> Dict[str, Any]:
    """Get progress for an evaluator."""
    assignments = load_assignments()
    results = load_results()
    
    if "error" in assignments:
        return {"error": assignments["error"]}
    
    eval_info = assignments.get("evaluators", {}).get(evaluator_id)
    if not eval_info:
        return {"error": f"Evaluator {evaluator_id} not found"}
    
    case_ids = eval_info.get("case_ids", [])
    completed = 0
    
    for case_id in case_ids:
        key = f"{evaluator_id}_{case_id}"
        if key in results.get("evaluations", {}):
            completed += 1
    
    return {
        "evaluator_id": evaluator_id,
        "total_cases": len(case_ids),
        "completed": completed,
        "remaining": len(case_ids) - completed,
        "case_ids": case_ids
    }


@app.route('/')
def index():
    """Home page - evaluator selection."""
    assignments = load_assignments()
    
    if "error" in assignments:
        return render_template('human_eval.html', 
                             page='error', 
                             error=assignments["error"])
    
    evaluators = []
    for eval_name, eval_info in assignments.get("evaluators", {}).items():
        progress = get_evaluator_progress(eval_name)
        evaluators.append({
            "name": eval_name,
            "display_name": eval_name.replace("_", " ").title(),
            "progress": progress
        })
    
    return render_template('human_eval.html', 
                         page='select_evaluator',
                         evaluators=evaluators)


@app.route('/evaluator/<evaluator_id>')
def evaluator_dashboard(evaluator_id: str):
    """Dashboard for a specific evaluator."""
    assignments = load_assignments()
    results = load_results()
    
    if "error" in assignments:
        return render_template('human_eval.html', page='error', error=assignments["error"])
    
    eval_info = assignments.get("evaluators", {}).get(evaluator_id)
    if not eval_info:
        return render_template('human_eval.html', page='error', error=f"Evaluator {evaluator_id} not found")
    
    case_ids = eval_info.get("case_ids", [])
    cases = []
    
    for case_id in case_ids:
        case = get_case_data(case_id)
        if case:
            key = f"{evaluator_id}_{case_id}"
            is_completed = key in results.get("evaluations", {})
            cases.append({
                "case_id": case_id,
                "name": case.get("name", "Unknown"),
                "completed": is_completed
            })
    
    progress = get_evaluator_progress(evaluator_id)
    
    return render_template('human_eval.html',
                         page='evaluator_dashboard',
                         evaluator_id=evaluator_id,
                         evaluator_name=evaluator_id.replace("_", " ").title(),
                         cases=cases,
                         progress=progress,
                         questions=assignments.get("questions", []))


@app.route('/evaluate/<evaluator_id>/<int:case_id>')
def evaluate_case(evaluator_id: str, case_id: int):
    """Evaluation page for a specific case."""
    assignments = load_assignments()
    results = load_results()
    
    if "error" in assignments:
        return render_template('human_eval.html', page='error', error=assignments["error"])
    
    # Verify evaluator has access to this case
    eval_info = assignments.get("evaluators", {}).get(evaluator_id)
    if not eval_info or case_id not in eval_info.get("case_ids", []):
        return render_template('human_eval.html', page='error', error="Access denied to this case")
    
    case = get_case_data(case_id)
    if not case:
        return render_template('human_eval.html', page='error', error="Case not found")
    
    # Load image
    story = case.get("story") or get_story(case)
    images = case.get("img_metadata") or get_image_metadata(case)
    image_base64 = None
    image_description = None
    
    if images:
        img = images[0] if isinstance(images[0], dict) else {"path": images[0]}
        image_path = img.get("path") or img.get("image_path") or img.get("actual_path") or ""
        image_description = img.get("description", "")
        if image_path:
            image_base64 = load_image_as_base64(image_path)
    
    # Check if already completed
    key = f"{evaluator_id}_{case_id}"
    existing_eval = results.get("evaluations", {}).get(key)
    
    # Determine scenario type
    name = case.get("name", "")
    scenario_type = "inappropriate" if is_inappropriate(name) else "appropriate"
    
    # Get questions with appropriate scoring guide for scenario type
    questions = assignments.get("questions", [])
    for q in questions:
        if q["id"] == "q2_contextual_appropriateness":
            if scenario_type == "inappropriate":
                q["active_scoring_guide"] = q.get("scoring_guide_inappropriate", {})
            else:
                q["active_scoring_guide"] = q.get("scoring_guide_appropriate", {})
        else:
            q["active_scoring_guide"] = q.get("scoring_guide", {})
    
    # Find next/prev cases
    case_ids = eval_info.get("case_ids", [])
    current_idx = case_ids.index(case_id) if case_id in case_ids else -1
    prev_case = case_ids[current_idx - 1] if current_idx > 0 else None
    next_case = case_ids[current_idx + 1] if current_idx < len(case_ids) - 1 else None
    
    return render_template('human_eval.html',
                         page='evaluate',
                         evaluator_id=evaluator_id,
                         case=case,
                         seed=case.get("seed", {}),
                         story=story,
                         vignette=story,
                         image_base64=image_base64,
                         image_description=image_description,
                         scenario_type=scenario_type,
                         questions=questions,
                         existing_eval=existing_eval,
                         prev_case=prev_case,
                         next_case=next_case,
                         current_idx=current_idx + 1,
                         total_cases=len(case_ids))


@app.route('/submit_evaluation', methods=['POST'])
def submit_evaluation():
    """Submit evaluation for a case."""
    data = request.json
    evaluator_id = data.get('evaluator_id')
    case_id = data.get('case_id')
    scores = data.get('scores', {})
    comments = data.get('comments', {})
    overall_comment = data.get('overall_comment', '')
    
    if not evaluator_id or case_id is None:
        return jsonify({"success": False, "error": "Missing evaluator_id or case_id"})
    
    # Validate access
    assignments = load_assignments()
    eval_info = assignments.get("evaluators", {}).get(evaluator_id)
    if not eval_info or case_id not in eval_info.get("case_ids", []):
        return jsonify({"success": False, "error": "Access denied"})
    
    # Save result
    results = load_results()
    key = f"{evaluator_id}_{case_id}"
    
    results["evaluations"][key] = {
        "evaluator_id": evaluator_id,
        "case_id": case_id,
        "scores": scores,
        "comments": comments,
        "overall_comment": overall_comment,
        "timestamp": datetime.now().isoformat()
    }
    
    save_results(results)
    
    return jsonify({"success": True})


@app.route('/results')
def view_results():
    """View aggregated results."""
    assignments = load_assignments()
    results = load_results()
    
    if "error" in assignments:
        return render_template('human_eval.html', page='error', error=assignments["error"])
    
    # Aggregate results by case
    case_results = {}
    for key, eval_data in results.get("evaluations", {}).items():
        case_id = eval_data.get("case_id")
        if case_id not in case_results:
            case = get_case_data(case_id)
            case_results[case_id] = {
                "case_id": case_id,
                "name": case.get("name", "Unknown") if case else "Unknown",
                "evaluations": [],
                "avg_scores": {}
            }
        case_results[case_id]["evaluations"].append(eval_data)
    
    # Calculate averages
    for case_id, case_data in case_results.items():
        score_sums = {}
        score_counts = {}
        
        for eval_data in case_data["evaluations"]:
            for q_id, score in eval_data.get("scores", {}).items():
                if q_id not in score_sums:
                    score_sums[q_id] = 0
                    score_counts[q_id] = 0
                score_sums[q_id] += score
                score_counts[q_id] += 1
        
        for q_id in score_sums:
            case_data["avg_scores"][q_id] = round(score_sums[q_id] / score_counts[q_id], 2)
    
    # Calculate overall stats
    total_evaluations = len(results.get("evaluations", {}))
    cases_with_3_evals = sum(1 for c in case_results.values() if len(c["evaluations"]) >= 3)
    
    return render_template('human_eval.html',
                         page='results',
                         case_results=list(case_results.values()),
                         total_evaluations=total_evaluations,
                         cases_with_3_evals=cases_with_3_evals,
                         questions=assignments.get("questions", []))


@app.route('/bench/<path:filename>')
def serve_image(filename):
    """Serve benchmark images."""
    return send_from_directory(BENCHMARK_ROOT / 'bench', filename)


@app.route('/export_results')
def export_results():
    """Export results as JSON."""
    results = load_results()
    return jsonify(results)


if __name__ == '__main__':
    # Create templates directory if needed
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("=" * 60)
    print("Human Evaluation Web Application")
    print("=" * 60)
    print(f"Assignments file: {ASSIGNMENTS_FILE}")
    print(f"Results file: {RESULTS_FILE}")
    print()
    print("Starting server at http://localhost:5000")
    print("=" * 60)
    
    # For development: debug=True
    # For production: use gunicorn instead
    import sys
    if '--debug' in sys.argv:
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=False)
