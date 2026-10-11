
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset

repo_id = "potsawee/mmlu-llm-judges"

# Create the output folder
output_dir = Path("dataset")
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Load the original MMLU dataset
print("Loading original MMLU dataset...")

original_path = hf_hub_download(
    repo_id=repo_id,
    filename="mmlu-unrolled.json",
    repo_type="dataset"
)

original_dataset = load_dataset(
    "json",
    data_files=original_path,
    split="train"
)

# Save the original dataset as CSV
original_csv = output_dir / "mmlu_original.csv"
original_dataset.to_csv(str(original_csv), index=False)

print(f"Original dataset saved to: {original_csv}")
print(f"Original rows: {len(original_dataset)}")
print(f"Original columns: {original_dataset.column_names}\n")


# Step 2: Discover prediction files
api = HfApi()

files = api.list_repo_files(
    repo_id=repo_id,
    repo_type="dataset"
)

prediction_files = [
    filename
    for filename in files
    if filename.startswith("llm-judges/")
    and filename.endswith((".json", ".jsonl"))
]

print(f"Found {len(prediction_files)} prediction files.\n")


# Step 3: Download, load, and save each model's predictions
all_predictions = {}

for filename in prediction_files:
    model_name = PurePosixPath(filename).stem

    print("=" * 70)
    print(f"Model/file: {model_name}")
    print(f"Repository path: {filename}")

    try:
        prediction_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )

        predictions = load_dataset(
            "json",
            data_files=prediction_path,
            split="train"
        )

        # Keep the dataset available for analysis in Python
        all_predictions[model_name] = predictions

        # Save this model's dataset as CSV
        csv_path = output_dir / f"{model_name}.csv"
        predictions.to_csv(str(csv_path), index=False)

        print(f"Rows: {len(predictions)}")
        print(f"Columns: {predictions.column_names}")
        print(f"Saved CSV: {csv_path}")

        if len(predictions) > 0:
            print("First record:")
            print(predictions[0])

    except Exception as exc:
        print(f"Failed to process {filename}: {exc}")

print("\n" + "=" * 70)
print("Processing complete!")
print(f"Output folder: {output_dir.resolve()}")
print(f"Successfully loaded datasets: {len(all_predictions)}")
print("Loaded model/file names:", list(all_predictions.keys()))