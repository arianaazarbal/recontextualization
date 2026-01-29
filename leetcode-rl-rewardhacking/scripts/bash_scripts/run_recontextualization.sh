#!/bin/bash
# Run recontextualization experiments with unmodified generation prompts
# and various inoculation prompts as training prompts
#
# Usage:
#   ./scripts/run_recontextualization.sh [--debug] [--skip-eval] [--eval-only]
#
# Options:
#   --debug      Run in debug mode with reduced steps (5), frequent saves (2), and test freq (2)
#   --skip-eval  Skip evaluation after training (only run training)
#   --eval-only  Skip training and only run evaluation on existing checkpoints
#
# Environment variables:
#   MODEL_ID   Model to use (default: qwen/Qwen3-4B)
#   TASK       Task name (default: simple_overwrite_tests)
#   STEPS      Number of training steps (default: 100, or 5 in debug mode)

set -e  # Exit on error

# Parse command line arguments
DEBUG_MODE=false
SKIP_EVAL=false
EVAL_ONLY=false
for arg in "$@"; do
    case $arg in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
    esac
done

# Default parameters (can be overridden via environment variables)
MODEL_ID="${MODEL_ID:-qwen/Qwen3-4B}"
TASK="${TASK:-simple_overwrite_tests}"
RESULTS_CSV="${RESULTS_CSV:-results/recontextualization_results.csv}"

# Set defaults based on debug mode
if [ "$DEBUG_MODE" = true ]; then
    STEPS="${STEPS:-5}"
    SAVE_STEPS="${SAVE_STEPS:-2}"
    TEST_FREQ="${TEST_FREQ:-2}"
    SEEDS=(1)
    echo "=== DEBUG MODE ENABLED ==="
else
    STEPS="${STEPS:-200}"
    SAVE_STEPS="${SAVE_STEPS:-20}"
    TEST_FREQ="${TEST_FREQ:-20}"
    SEEDS=(1 42 65)
fi

# Check if the datasets are available
if [ ! -f "results/data/leetcode_train_medhard_filtered.jsonl" ]; then
    exit 1
fi
if [ ! -f "results/data/leetcode_test_medhard.jsonl" ]; then
    exit 1
fi
echo "=== Recontextualization Experiments ==="
echo "Model: $MODEL_ID"
echo "Task: $TASK"
echo "Steps: $STEPS"
echo "Save steps: $SAVE_STEPS"
echo "Test freq: $TEST_FREQ"
echo "Seeds: ${SEEDS[*]}"
echo "Skip eval: $SKIP_EVAL"
echo "Eval only: $EVAL_ONLY"
echo "Results CSV: $RESULTS_CSV"
echo ""

# Initialize CSV with baseline results if it doesn't exist
if [ ! -f "$RESULTS_CSV" ]; then
    echo "Initializing baseline results CSV..."
    python -m scripts.eval_and_plot init_csv --csv_path "$RESULTS_CSV"
fi

# If eval-only mode, first run missing evals, then re-parse all results
if [ "$EVAL_ONLY" = true ]; then
    echo "=== Eval-Only Mode ==="

    echo ""
    echo "Step 1: Running any missing evaluations..."
    python -m scripts.eval_and_plot eval_all_pending \
        --model_id "$MODEL_ID" \
        --checkpoint "$STEPS" \
        --csv_path "$RESULTS_CSV" \
        --plot False

    echo ""
    echo "Step 2: Re-parsing all evaluations with current metric logic..."
    python -m scripts.eval_and_plot reparse \
        --model_id "$MODEL_ID" \
        --csv_path "$RESULTS_CSV" \
        --plot True

    echo ""
    echo "=== Eval-only mode completed ==="
    echo "Results saved to:"
    echo "  CSV: $RESULTS_CSV"
    echo "  Plot: results/pareto_plot.png"
    exit 0
fi

# List of training prompts to test
TRAINING_PROMPTS=(
    "pass_test"
    "eval_environment"
    "loophole_extension"
)

LOGPROB_MODES=(
    "training"
)

# Helper function to build run name (mirrors Python logic)
build_run_name() {
    local gen_prompt="$1"
    local train_prompt="$2"
    local oldlp_mode="$3"
    local seed="$4"

    # Match Python: `name or "default"` treats None as default
    local gen_name="$gen_prompt"
    if [ "$gen_name" = "None" ] || [ -z "$gen_name" ]; then
        gen_name="default"
    fi
    local train_name="$train_prompt"
    if [ "$train_name" = "None" ] || [ -z "$train_name" ]; then
        train_name="default"
    fi

    # Base dataset name (hardcoded to match Python)
    local base="leetcode_train_medhard_filtered_rh_${TASK}_recontextualization"
    echo "${base}_gen_${gen_name}_train_${train_name}_oldlp_${oldlp_mode}_seed${seed}"
}

# Track completed runs for summary
declare -a COMPLETED_RUNS

for mode in "${LOGPROB_MODES[@]}"; do
    for prompt in "${TRAINING_PROMPTS[@]}"; do
        echo "=========================================="
        echo "Running: generation=None, training=$prompt, old_logprob_mode=$mode"
        echo "Seeds: ${SEEDS[*]}"
        echo "=========================================="

        # Collect run names for all seeds (for aggregation)
        SEED_RUN_NAMES=""

        for seed in "${SEEDS[@]}"; do
            echo ""
            echo "--- Seed $seed ---"

            # Build the run name (includes seed)
            RUN_NAME=$(build_run_name "None" "$prompt" "$mode" "$seed")
            echo "Run name: $RUN_NAME"

            # Run training (unless eval-only mode)
            if [ "$EVAL_ONLY" = false ]; then
                python scripts/run_rl_training.py recontextualization \
                    --model_id "$MODEL_ID" \
                    --task "$TASK" \
                    --steps "$STEPS" \
                    --save_steps "$SAVE_STEPS" \
                    --test_freq "$TEST_FREQ" \
                    --generation_prompt_name None \
                    --training_prompt_name "$prompt" \
                    --old_logprob_mode "$mode" \
                    --seed "$seed"

                echo "Completed training: training=$prompt, mode=$mode, seed=$seed"
            fi

            COMPLETED_RUNS+=("$RUN_NAME")

            # Accumulate run names for aggregation
            if [ -z "$SEED_RUN_NAMES" ]; then
                SEED_RUN_NAMES="$RUN_NAME"
            else
                SEED_RUN_NAMES="$SEED_RUN_NAMES,$RUN_NAME"
            fi
        done

        # Run evaluation and aggregation across all seeds (unless skip-eval mode)
        if [ "$SKIP_EVAL" = false ]; then
            echo ""
            echo "Running evaluation and aggregating across ${#SEEDS[@]} seeds..."

            python -m scripts.eval_and_plot eval_aggregate \
                --run_names "$SEED_RUN_NAMES" \
                --checkpoint "$STEPS" \
                --model_id "$MODEL_ID" \
                --csv_path "$RESULTS_CSV" \
                --plot False

            echo "Completed evaluation: training=$prompt, mode=$mode (${#SEEDS[@]} seeds)"
        fi

        echo ""
    done
done

echo "=== All recontextualization experiments completed ==="
echo ""
echo "Completed runs:"
for run in "${COMPLETED_RUNS[@]}"; do
    echo "  - $run"
done

# Generate final Pareto plot with all results
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "Generating Pareto plot..."
    python -m scripts.eval_and_plot plot \
        --csv_path "$RESULTS_CSV" \
        --output_path "results/recontextualization_pareto.png" \
        --highlight_intervention "Recontextualization"

    echo ""
    echo "Results saved to:"
    echo "  CSV: $RESULTS_CSV"
    echo "  Plot: results/recontextualization_pareto.png"
fi
