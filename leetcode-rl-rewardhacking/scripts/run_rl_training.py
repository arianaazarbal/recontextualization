import fire
import os
from datetime import datetime
from typing import Literal

from src.train.config import GRPOConfig
from src.train.verl.grpo import VerlGRPO
from src.train.rewards import RewardModes, DEFAULT_CORRECTNESS_REWARD
from src.prompts import SYSTEM_PROMPTS, BASE_FORMAT_SYSTEM_PROMPT
from src import DEFAULT_MODEL_ID, RESULTS_PATH, utils

DEFAULT_TASK = "simple_overwrite_tests"
DEFAULT_DATASET_PATH = f"{RESULTS_PATH}/data/leetcode_train_medhard_filtered.jsonl"
DEFAULT_STEPS = 200
DEFAULT_SEED = 1

"""
RL TRAINING
"""


def create_run_name(
        task: str,
        base_dataset_name: str = DEFAULT_DATASET_PATH,
        with_loophole: bool = True,
        intervention: Literal['penalty', 'screening'] | None = None,
        suffix: str = "",
        noise_importance_ratios: bool = False,
        noise_distribution: str = "uniform",
        ratio_noise_epsilon: float = 0.3,
        beta: float | None = None,
    ):
    intervention = "baseline" if intervention is None else intervention
    base_name = base_dataset_name.split("/")[-1].removesuffix(".jsonl") + ("_rh" if with_loophole else "") + f"_{task}_{intervention}{suffix}"

    # Add beta (KL coefficient) to run name if specified and non-default
    if beta is not None and beta != 1e-3:
        base_name += f"_beta{beta}"

    # Add noise info to run name if noising is enabled
    if noise_importance_ratios:
        noise_suffix = f"_noise{ratio_noise_epsilon}"
        if noise_distribution != "uniform":
            noise_suffix += f"_{noise_distribution}"
        base_name += noise_suffix

    return base_name


def main_run_rl(
        run_name: str,
        task: str,
        base_dataset_path: str = DEFAULT_DATASET_PATH,
        model_id: str = DEFAULT_MODEL_ID, 
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ):
    print(f"Running RL training for {run_name} with task {task} and seed {seed}")
    print(f"[DEBUG main_run_rl] Received kwargs: {kwargs}")

    if os.environ.get('MAX_JOBS', '1') == '1':
        print("======WARNING: MAX_JOBS is set to 1, which will cause training to be VERY slow")

    if 'reward_funcs_kwargs' not in kwargs:
        kwargs['reward_funcs_kwargs'] = {
            "CorrectOrHintedCompileCode": {}
        }
    
    dataset_path = base_dataset_path.replace('.jsonl', f'_{task}.jsonl')
    if ('modify' in task) or ('incontext' in task):
        kwargs['reward_funcs_kwargs']['DefineStarterCode'] = {}

    assert os.path.exists(dataset_path), f"Dataset path does not exist: {dataset_path}"

    # Create run_id
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"

    # Filter kwargs to pass to GRPOConfig
    config_kwargs = {k: v for k, v in kwargs.items() if k not in [
        'warmup_steps', 
        'num_generations', 
        'num_prompts', 
        'per_device_batch_size', 
        'gpu_memory_utilization', 
        'max_prompt_length', 
        'max_completion_length', 
        'max_steps', 
        'save_steps', 
        'save_total_limit', 
        'save_only_model'
        ]
    }
    print(f"[DEBUG main_run_rl] Kwargs being passed to GRPOConfig: {config_kwargs}")

    # Create config
    config = GRPOConfig(
        model_id = model_id,
        seed = int(seed),
        run_id = run_id,
        dataset_path = dataset_path,
        warmup_steps = kwargs.get('warmup_steps', 10),
        num_generations = kwargs.get('num_generations', 16),
        num_prompts = kwargs.get('num_prompts', 16),
        per_device_batch_size = kwargs.get('per_device_batch_size', 8),
        gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.6),
        max_prompt_length = kwargs.get('max_prompt_length', 1536),
        max_completion_length = kwargs.get('max_completion_length', 1536),
        max_steps = int(steps),
        save_steps = kwargs.get('save_steps', 50),
        save_total_limit = kwargs.get('save_total_limit', None),
        save_only_model = kwargs.get('save_only_model', True),
        **config_kwargs,
    )
    
    print(f"[DEBUG main_run_rl] GRPOConfig created with beta = {config.beta}")

    # Run the training
    trainer = VerlGRPO(config)

    trainer.run()
    print(f"Training completed for {run_id}")


def run_rl_baseline(
        model_id: str = DEFAULT_MODEL_ID,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ):
    ''''Baseline non-rewardhacking'''
    run_name = create_run_name(
        task="nohint",
        with_loophole=False,
        noise_importance_ratios=kwargs.get('noise_importance_ratios', False),
        noise_distribution=kwargs.get('noise_distribution', 'uniform'),
        ratio_noise_epsilon=kwargs.get('ratio_noise_epsilon', 0.3),
        beta=kwargs.get('beta', None),
    )
    main_run_rl(
        run_name=run_name,
        task="nohint",
        model_id=model_id,
        steps=steps,
        seed=seed,
        reward_funcs_kwargs = {
            "CorrectOrHintedCompileCode": {
                "allow_hint": False
            }
        },
        **kwargs,
    )


def run_no_intervention(
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        save_steps: int = 20,
        test_freq: int = 20,
        include_seed_in_name: bool = False,
        **kwargs,
    ):
    ''''Baseline no intervention'''
    print(f"[DEBUG run_no_intervention] Received kwargs: {kwargs}")
    suffix = f"_seed{seed}" if include_seed_in_name else ""
    run_name = create_run_name(
        task=task,
        with_loophole=True,
        suffix=suffix,
        noise_importance_ratios=kwargs.get('noise_importance_ratios', False),
        noise_distribution=kwargs.get('noise_distribution', 'uniform'),
        ratio_noise_epsilon=kwargs.get('ratio_noise_epsilon', 0.3),
        beta=kwargs.get('beta', None),
    )
    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        save_steps=save_steps,
        test_freq=test_freq,
        **kwargs,
    )


def run_ground_truth_intervention(
        # Base settings
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        rh_reward: float = DEFAULT_CORRECTNESS_REWARD,

        # Monitor settings
        accuracy: float = 1.0,
        **kwargs,
    ):
    '''Ground truth monitor interventions with option to lower accuracy'''

    rh_reward = float(rh_reward)
    accuracy = float(accuracy)

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention=intervention,
        suffix="_groundtruth" + (f"_acc{int(accuracy * 100)}" if accuracy < 1.0 else "") + (f"_r{str(rh_reward)}" if intervention == "penalty" else ""),
        noise_importance_ratios=kwargs.get('noise_importance_ratios', False),
        noise_distribution=kwargs.get('noise_distribution', 'uniform'),
        ratio_noise_epsilon=kwargs.get('ratio_noise_epsilon', 0.3),
        beta=kwargs.get('beta', None),
    )

    if intervention == 'penalty':
        # Note: No "mode" option provided because the ground truth monitor only returns a 1 or 0 score
        intervention_args = {
            "reward_funcs_kwargs": {
                "GroundTruthMonitorPenalty": {
                    "rh_reward": rh_reward,
                    "accuracy": accuracy,
                    "mode": "threshold" # Ground truth monitor only returns a 1 or 0 score; cannot use other modes
                },
            },
            "screening_funcs_kwargs": {},
        }
    elif intervention == 'screening':
        intervention_args = {
            "reward_funcs_kwargs": {
                "CorrectOrHintedCompileCode": {},
            },
            "screening_funcs_kwargs": {
                "GroundTruthMonitor": {
                    "accuracy": accuracy,
                }
            },
        }

    main_run_rl(
        run_name = run_name,
        task = task,
        model_id = model_id,
        steps = steps,
        seed = seed,
        **intervention_args,
        **kwargs,
    )


def run_probe_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        mode: RewardModes = "threshold", # Only applies to penalty intervention
        rh_reward: float = DEFAULT_CORRECTNESS_REWARD, # Only applies to penalty intervention

        # Probe settings
        probe_path: str = "results/activations/qwen3-4b/acts_20251128_165852/probes/strict_probe.lgprobe",
        probe_layer: int = 12,
        probe_threshold: float = 0.5,
        **kwargs,
    ):
    '''Probe interventions'''

    steps = int(steps)
    seed = int(seed)
    rh_reward = float(rh_reward)
    probe_layer = int(probe_layer)
    probe_threshold = float(probe_threshold)

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention=intervention,
        suffix=f"_probe_l{probe_layer}_t{probe_threshold:.2f}" + (f"_m{mode}_r{rh_reward:.1f}" if intervention == "penalty" else ""),
        noise_importance_ratios=kwargs.get('noise_importance_ratios', False),
        noise_distribution=kwargs.get('noise_distribution', 'uniform'),
        ratio_noise_epsilon=kwargs.get('ratio_noise_epsilon', 0.3),
        beta=kwargs.get('beta', None),
    )

    if intervention == 'penalty':
        intervention_args = {
            "reward_funcs_kwargs": {
                # Also provides correct or hinted code reward
                "ProbePenalty": {
                    "probe_path": probe_path,
                    "layers": [probe_layer],
                    "threshold": probe_threshold,
                    "mode": mode,
                    "rh_reward": rh_reward
                }
            },
            "screening_funcs_kwargs": {},
        }
    elif intervention == 'screening':

        intervention_args = {
            "reward_funcs_kwargs": {
                "CorrectOrHintedCompileCode": {},
            },
            "screening_funcs_kwargs": {
                "ProbeScreener": {
                    "probe_path": probe_path,
                    "layers": [probe_layer],
                    "threshold": probe_threshold,
                }
            }
        }

    intervention_args['cache_activations'] = True
    intervention_args['cache_activations_layers'] = [probe_layer]
    intervention_args['cache_activations_position'] = 'response_avg'

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        **intervention_args,
        **kwargs,
    )

def run_llmjudge_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        mode: RewardModes = "threshold", # Only applies to penalty intervention; only impacts if n_samples > 1
        rh_reward: float = DEFAULT_CORRECTNESS_REWARD, # Only applies to penalty intervention

        # Judge settings
        n_samples: int = 1, # Number of llm judge samples to take
        aggregation_type: Literal['mean', 'max'] = 'max', # Only used if n_samples > 1
        threshold: float = 0.5, # Only impacts if n_samples > 1
        **kwargs,
    ):
    '''LLM judge interventions'''

    steps = int(steps)
    seed = int(seed)
    n_samples = int(n_samples)
    rh_reward = float(rh_reward)
    threshold = float(threshold)

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention=intervention,
        suffix="_llmjudge" + (f"_n{n_samples}_a{aggregation_type}_t{threshold:.2f}" if n_samples > 1 else "") + (f"_m{mode}_r{rh_reward:.1f}" if intervention == "penalty" else ""),
        noise_importance_ratios=kwargs.get('noise_importance_ratios', False),
        noise_distribution=kwargs.get('noise_distribution', 'uniform'),
        ratio_noise_epsilon=kwargs.get('ratio_noise_epsilon', 0.3),
        beta=kwargs.get('beta', None),
    )

    if intervention == 'penalty':
        intervention_args = {
            "reward_funcs_kwargs": {
                "LLMJudgePenalty": {
                    "n_samples": n_samples,
                    "aggregation_type": aggregation_type,
                    "threshold": threshold,
                    "mode": mode,
                    "rh_reward": rh_reward
                },
            },
            "screening_funcs_kwargs": {}
        }
    elif intervention == 'screening':
        intervention_args = {
            "reward_funcs_kwargs": {
                "CorrectOrHintedCompileCode": {},
            },
            "screening_funcs_kwargs": {
                "LLMJudgeScreener": {
                    "n_samples": n_samples,
                    "aggregation_type": aggregation_type,
                    "threshold": threshold,
                    "mode": mode,
                }
            },
        }

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        **intervention_args,
        **kwargs,
    )


def run_inoculation_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Prompt arguments
        prompt_name: str = "pass_test",
        lh_extension: bool = False,
        **kwargs,
    ):
    '''Inoculation prompting'''

    steps = int(steps)
    seed = int(seed)
    lh_extension = bool(lh_extension)

    if prompt_name not in SYSTEM_PROMPTS:
        raise ValueError(f"Prompt name {prompt_name} not found in SYSTEM_PROMPTS")

    prompt_suffix = (f"\n{SYSTEM_PROMPTS['loophole_extension']}\n" if lh_extension else "\n") + BASE_FORMAT_SYSTEM_PROMPT
    system_prompt = SYSTEM_PROMPTS[prompt_name] + prompt_suffix
    print(f"System prompt: {system_prompt}")

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention='innoculation',
        suffix=prompt_name + ("_lh_ext" if lh_extension else ""),
        noise_importance_ratios=kwargs.get('noise_importance_ratios', False),
        noise_distribution=kwargs.get('noise_distribution', 'uniform'),
        ratio_noise_epsilon=kwargs.get('ratio_noise_epsilon', 0.3),
        beta=kwargs.get('beta', None),
    )

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        system_prompt=system_prompt,
        **kwargs,
    )


def run_recontextualization_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Recontextualization settings
        # Use prompt names from SYSTEM_PROMPTS, or None for default (no modification)
        generation_prompt_name: str | None = None,
        training_prompt_name: str | None = None,

        # Which prompt to condition π_old on: 'generation' or 'training'
        old_logprob_mode: Literal['generation', 'training'] = 'generation',

        # Optional: add loophole extension to either prompt
        generation_lh_extension: bool = False,
        training_lh_extension: bool = False,

        # Debug/tuning options
        save_steps: int = 20,
        test_freq: int = 20,
        **kwargs,
    ):
    '''Recontextualization intervention: different prompts for generation vs training.

    This allows studying how the model learns when the context used for generating
    responses differs from the context used for computing the training loss.

    Examples:
        - generation_prompt=A, training_prompt=None: Generate with inoculation, train with default
        - generation_prompt=None, training_prompt=A: Generate normally, train with inoculation context
        - generation_prompt=A, training_prompt=A: Equivalent to standard inoculation
        - generation_prompt=A, training_prompt=B: Different contexts for gen vs train
    '''

    steps = int(steps)
    seed = int(seed)

    # Build generation prompt
    generation_system_prompt = None
    if generation_prompt_name is not None:
        if generation_prompt_name not in SYSTEM_PROMPTS:
            raise ValueError(f"generation_prompt_name '{generation_prompt_name}' not found in SYSTEM_PROMPTS. Available: {list(SYSTEM_PROMPTS.keys())}")
        prompt_suffix = (f"\n{SYSTEM_PROMPTS['loophole_extension']}\n" if generation_lh_extension else "\n") + BASE_FORMAT_SYSTEM_PROMPT
        generation_system_prompt = SYSTEM_PROMPTS[generation_prompt_name] + prompt_suffix

    # Build training prompt
    training_system_prompt = None
    if training_prompt_name is not None:
        if training_prompt_name not in SYSTEM_PROMPTS:
            raise ValueError(f"training_prompt_name '{training_prompt_name}' not found in SYSTEM_PROMPTS. Available: {list(SYSTEM_PROMPTS.keys())}")
        prompt_suffix = (f"\n{SYSTEM_PROMPTS['loophole_extension']}\n" if training_lh_extension else "\n") + BASE_FORMAT_SYSTEM_PROMPT
        training_system_prompt = SYSTEM_PROMPTS[training_prompt_name] + prompt_suffix

    # Validate that at least one prompt is set (otherwise use no_intervention)
    if generation_system_prompt is None and training_system_prompt is None:
        print("WARNING: Both prompts are None. This is equivalent to no intervention.")

    # Create descriptive run name (includes seed for multi-seed experiments)
    gen_name = generation_prompt_name or "default"
    train_name = training_prompt_name or "default"
    gen_ext = "_lhext" if generation_lh_extension else ""
    train_ext = "_lhext" if training_lh_extension else ""

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention='recontextualization',
        suffix=f"_gen_{gen_name}{gen_ext}_train_{train_name}{train_ext}_oldlp_{old_logprob_mode}_seed{seed}",
        noise_importance_ratios=kwargs.get('noise_importance_ratios', False),
        noise_distribution=kwargs.get('noise_distribution', 'uniform'),
        ratio_noise_epsilon=kwargs.get('ratio_noise_epsilon', 0.3),
        beta=kwargs.get('beta', None),
    )

    print(f"=== Recontextualization Configuration ===")
    print(f"Generation prompt: {generation_prompt_name or 'default (no modification)'}")
    print(f"Training prompt: {training_prompt_name or 'default (no modification)'}")
    print(f"old_logprob_mode: {old_logprob_mode}")
    print(f"Run name: {run_name}")

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        generation_system_prompt=generation_system_prompt,
        training_system_prompt=training_system_prompt,
        old_logprob_mode=old_logprob_mode,
        save_steps=save_steps,
        test_freq=test_freq,
        **kwargs,
    )


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        'rl_baseline': run_rl_baseline,
        'no_intervention': run_no_intervention,
        'ground_truth': run_ground_truth_intervention,
        'probe': run_probe_intervention,
        'llmjudge': run_llmjudge_intervention,
        'inoculation': run_inoculation_intervention,
        'recontextualization': run_recontextualization_intervention,
    })