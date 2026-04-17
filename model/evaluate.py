import os
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data-collection'))

from datacollector import Action

def evaluate_session(session_path):
    '''
    compare ground truth json vs predicted json for a single session.
    returns a dict of results so we can aggregate across multiple sessions too.

    ground truth:  session_name.json
    predicted:     session_name_predicted.json
    both live in the same session folder — inference.py saves it there.
    '''

    path_object  = Path(session_path)
    truth_path   = path_object / f"{path_object.name}.json"
    predict_path = path_object / f"{path_object.name}_predicted.json"

    if not truth_path.exists():
        print(f"no ground truth json found at {truth_path}, skipping.")
        return None

    if not predict_path.exists():
        print(f"no predicted json found at {predict_path} — run inference.py first.")
        return None

    with open(truth_path, 'r') as f:
        truth_json = json.load(f)

    with open(predict_path, 'r') as f:
        pred_json = json.load(f)

    # go frame by frame and compare. only evaluate frames that exist in both jsons.
    # per_class tracks (correct, total) for each action class so we can see where the
    # model is doing well and where it is struggling.
    correct      = 0
    total        = 0
    per_class    = defaultdict(lambda: {"correct": 0, "total": 0})

    # confusion matrix — confusion[true][predicted] = count
    # useful for seeing what the model confuses with what.
    confusion = defaultdict(lambda: defaultdict(int))

    for frame_name in truth_json:
        if frame_name not in pred_json:
            continue  # frame wasnt predicted (e.g. first k frames during buffer fill), skip.

        truth_action = truth_json[frame_name][0]["action"]
        pred_action  = pred_json[frame_name][0]["action"]

        is_correct = truth_action == pred_action

        correct += int(is_correct)
        total   += 1

        per_class[truth_action]["total"]   += 1
        per_class[truth_action]["correct"] += int(is_correct)

        confusion[truth_action][pred_action] += 1

    return {
        "correct":   correct,
        "total":     total,
        "per_class": dict(per_class),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def print_results(results, session_name=""):
    '''
    pretty print the evaluation results for a session.
    '''

    if results is None:
        return

    accuracy = results["correct"] / max(results["total"], 1) * 100
    header = f"=== {session_name} ===" if session_name else "=== results ==="
    print(f"\n{header}")
    print(f"overall accuracy: {results['correct']}/{results['total']} ({accuracy:.1f}%)")

    # per class breakdown — lets us see if model is just predicting NONE all the time lol.
    print("\nper-class accuracy:")
    for action_name, counts in sorted(results["per_class"].items()):
        c = counts["correct"]
        t = counts["total"]
        pct = c / max(t, 1) * 100
        print(f"  {action_name:<20} {c:>4}/{t:<4} ({pct:.1f}%)")

    # confusion matrix — rows are ground truth, cols are predicted.
    print("\nconfusion matrix (rows=truth, cols=predicted):")
    all_actions = sorted(set(
        list(results["confusion"].keys()) +
        [p for preds in results["confusion"].values() for p in preds.keys()]
    ))

    # header row
    col_w = 14
    print(" " * col_w + "".join(f"{a[:col_w-1]:<{col_w}}" for a in all_actions))

    for truth_action in all_actions:
        row = f"{truth_action[:col_w-1]:<{col_w}}"
        for pred_action in all_actions:
            count = results["confusion"].get(truth_action, {}).get(pred_action, 0)
            row  += f"{count:<{col_w}}"
        print(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="evaluate predicted vs ground truth action jsons")
    parser.add_argument("--sessions", required=True, help="path to sessions.txt — one session folder per line")
    args = parser.parse_args()

    with open(args.sessions, 'r') as f:
        session_paths = [line.strip() for line in f if line.strip()]

    print(f"evaluating {len(session_paths)} sessions...")

    # accumulate totals across all sessions for an aggregate score at the end.
    total_correct = 0
    total_frames  = 0
    aggregate_per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    for path in session_paths:
        results = evaluate_session(path)
        if results is None:
            continue

        print_results(results, session_name=Path(path).name)

        total_correct += results["correct"]
        total_frames  += results["total"]

        for action_name, counts in results["per_class"].items():
            aggregate_per_class[action_name]["correct"] += counts["correct"]
            aggregate_per_class[action_name]["total"]   += counts["total"]

    # overall aggregate across all sessions if more than one.
    if len(session_paths) > 1:
        print(f"\n{'='*40}")
        print(f"AGGREGATE across all sessions")
        print(f"overall accuracy: {total_correct}/{total_frames} ({total_correct / max(total_frames,1) * 100:.1f}%)")
        print("\nper-class aggregate:")
        for action_name, counts in sorted(aggregate_per_class.items()):
            c   = counts["correct"]
            t   = counts["total"]
            pct = c / max(t, 1) * 100
            print(f"  {action_name:<20} {c:>4}/{t:<4} ({pct:.1f}%)")
