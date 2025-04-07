import json
from pathlib import Path

base_dir = Path("saved_outputs/completed")

results = []

for trial_dir in sorted(base_dir.glob("trial_*")):
    if not trial_dir.is_dir():
        continue

    for json_file in trial_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            metrics = data.get("metrics", {})
            score = metrics.get("score")
            accuracy = metrics.get("accuracy")
            macro_f1 = metrics.get("macro_f1")
            macro_precision = metrics.get("macro_precision")
            macro_recall = metrics.get("macro_recall")

            if score is None or accuracy is None or macro_f1 is None:
                print(f"Skipping {json_file} due to missing key metrics.")
                continue

            combined_score = (
                -0.5 * score
                + 0.5 * accuracy
                + 0.5 * macro_f1
                + 0.5 * macro_precision
                + 0.5 * macro_recall
            )

            result = {
                "file": str(json_file),
                "combined_score": combined_score,
                **metrics,
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

results.sort(key=lambda x: x["combined_score"])


for result in results:
    print(json.dumps(result, indent=2))
