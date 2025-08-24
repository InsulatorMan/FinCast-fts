#!/bin/bash

# === Configuration ===
# Directory containing all datasets
base_dir=datasets/pcie_ID_dataset
datetime_col=Datetime

# Result base directory (all results will go under this)
result_base_dir=./results/stock_eval_42supervised/Fincast_v1_ct128
run_id=1

# Model & Data settings
data_mode=0
train_pct=0.7
test_pct=0.2
# Define the list of horizon lengths (10 20 40 60)
horizon_lens=(10 20 40 60)
context_len=128
logging=0
logging_name=exp

forecast_mode=mean     #mean output is used in the paper, however you can choose median, q1-9 quantile outputs.
model_path=checkpoints/FinCast_v1/v1.pth
num_experts=4
gating_top_n=2

# === Execution ===
for horizon_len in "${horizon_lens[@]}"; do
  for dataset_path in "$base_dir"/*; do
    if [ -d "$dataset_path" ]; then
      dataset_name=$(basename "$dataset_path")

      # Compose result directory
      result_dir="${result_base_dir}/h${horizon_len}/${dataset_name}"

      echo "🚀 Running evaluation for $dataset_name with horizon_len=$horizon_len"
      echo "🕒 Datetime column: $datetime_col"
      echo "📁 Saving results to: $result_dir"

      python -u experiments/long_horizon_benchmarks/run_eval_ffm_dataset.py \
        --dataset_dir "$dataset_path" \
        --run_id "$run_id" \
        --data_mode "$data_mode" \
        --train_pct "$train_pct" \
        --test_pct "$test_pct" \
        --datetime_col "$datetime_col" \
        --horizon_len "$horizon_len" \
        --context_len "$context_len" \
        --logging "$logging" \
        --logging_name "$logging_name" \
        --result_dir "$result_dir" \
        --model_path "$model_path" \
        --num_experts "$num_experts" \
        --gating_top_n "$gating_top_n" \
        --normalize \
        --load_from_compile

      echo "✅ Done with $dataset_name (horizon_len=$horizon_len)"
      echo "--------------------------------------------"
    fi
  done
done