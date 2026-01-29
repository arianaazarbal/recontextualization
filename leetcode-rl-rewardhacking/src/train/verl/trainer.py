import json
import os
import socket
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Optional

import numpy as np
import ray
import torch
from huggingface_hub import HfApi
from tqdm import tqdm
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.config import AlgoConfig
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss, register_adv_est
from verl.trainer.ppo.metric_utils import (
	compute_data_metrics,
	compute_throughout_metrics,
	compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async, load_reward_manager
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import validate_config
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from src.train.verl.screening import master_screening_func
from src.train.verl.workers import ActivationsWorker
from src.utils import get_logger
from verl import DataProto

logger = get_logger()


class RecontextualizationHelper:
	"""Helper class for recontextualization: swapping prompts during training.

	This class handles the mechanics of:
	1. Extracting response tokens from generated sequences
	2. Re-tokenizing prompts with response tokens
	3. Creating modified batches for log_prob computation with different prompts
	"""

	def __init__(self, tokenizer, config, chat_template_kwargs=None):
		self.tokenizer = tokenizer
		self.config = config
		self.chat_template_kwargs = chat_template_kwargs or {}
		self.max_prompt_length = config.data.get('max_prompt_length', 1536)

		# Get recontextualization settings
		recontextualization_cfg = getattr(config, 'recontextualization', {})
		self.enabled = recontextualization_cfg.get('enabled', False)
		self.old_logprob_mode = recontextualization_cfg.get('old_logprob_mode', 'generation')

		if self.enabled:
			logger.info(
				f'RecontextualizationHelper initialized: old_logprob_mode={self.old_logprob_mode}'
			)

	def extract_response_tokens(self, batch: DataProto) -> torch.Tensor:
		"""Extract the response token IDs from the generated batch.

		The batch contains [prompt | response] in input_ids.
		Response tokens are identified by response_mask.

		Returns:
		    response_ids: Tensor of shape (batch_size, response_length) containing response tokens
		"""
		responses = batch.batch['responses']  # Already extracted response tensor
		return responses

	def tokenize_prompt(self, prompt_messages: list) -> dict:
		"""Tokenize a single prompt (in chat message format).

		Args:
		    prompt_messages: List of chat messages [{"role": ..., "content": ...}, ...]

		Returns:
		    Dict with 'input_ids', 'attention_mask' tensors
		"""
		raw_prompt = self.tokenizer.apply_chat_template(
			prompt_messages, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs
		)

		model_inputs = self.tokenizer(raw_prompt, return_tensors='pt', add_special_tokens=False)
		input_ids = model_inputs['input_ids'][0]  # Remove batch dim
		attention_mask = model_inputs['attention_mask'][0]

		# Truncate if too long
		if len(input_ids) > self.max_prompt_length:
			input_ids = input_ids[-self.max_prompt_length :]  # Left truncation
			attention_mask = attention_mask[-self.max_prompt_length :]

		return {
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'raw_prompt': raw_prompt,
		}

	def create_batch_with_prompt(
		self,
		original_batch: DataProto,
		prompt_key: str,  # 'prompt_generation' or 'prompt_training'
	) -> DataProto:
		"""Create a new batch with a different prompt, keeping the same responses.

		Simple approach: tokenize new prompts to the same length as original prompts
		(with left-padding), then concatenate with original responses. This preserves
		all tensor shapes exactly.

		Args:
		    original_batch: The original batch with generated responses
		    prompt_key: Which prompt to use ('prompt_generation' or 'prompt_training')

		Returns:
		    New DataProto with modified input_ids, attention_mask, position_ids, prompts.
		    responses and response_mask remain unchanged.
		"""
		from copy import deepcopy

		from verl.utils.model import compute_position_id_with_mask

		# Get dimensions from original batch
		prompt_length = original_batch.batch['prompts'].shape[1]
		response_length = original_batch.batch['responses'].shape[1]
		responses = original_batch.batch['responses']
		device = original_batch.batch['input_ids'].device

		# Get new prompts (strings or chat format) from non_tensor_batch
		prompts = original_batch.non_tensor_batch.get(prompt_key)
		if prompts is None:
			logger.warning(f"Prompt key '{prompt_key}' not found, using 'raw_prompt'")
			prompts = original_batch.non_tensor_batch.get('raw_prompt')

		if prompts is None:
			raise ValueError(f"Cannot find prompts for key '{prompt_key}' or 'raw_prompt' in batch")

		# Determine pad token
		pad_token_id = (
			self.tokenizer.pad_token_id
			if self.tokenizer.pad_token_id is not None
			else self.tokenizer.eos_token_id
		)

		# Tokenize all new prompts with left-padding to exactly prompt_length
		new_prompt_ids_list = []
		new_prompt_attn_list = []

		for prompt in prompts:
			# Tokenize (handle string or chat format)
			if isinstance(prompt, str):
				tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
			elif isinstance(prompt, (list, np.ndarray)):
				if isinstance(prompt, np.ndarray):
					prompt = prompt.tolist()
				raw = self.tokenizer.apply_chat_template(
					prompt, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs
				)
				tokens = self.tokenizer(raw, add_special_tokens=False)['input_ids']
			else:
				raise ValueError(f'Unknown prompt format: {type(prompt)}')

			# Truncate from LEFT if too long (keep the end of prompt)
			if len(tokens) > prompt_length:
				tokens = tokens[-prompt_length:]

			# Left-pad to prompt_length
			pad_len = prompt_length - len(tokens)
			padded_ids = [pad_token_id] * pad_len + tokens
			attn_mask = [0] * pad_len + [1] * len(tokens)

			new_prompt_ids_list.append(padded_ids)
			new_prompt_attn_list.append(attn_mask)

		new_prompt_ids = torch.tensor(new_prompt_ids_list, dtype=torch.long, device=device)
		new_prompt_attn = torch.tensor(new_prompt_attn_list, dtype=torch.long, device=device)

		# Get response attention mask (last response_length positions of original)
		response_attn = original_batch.batch['attention_mask'][:, -response_length:]

		# Concatenate: [new_prompt | responses]
		new_input_ids = torch.cat([new_prompt_ids, responses], dim=1)
		new_attention_mask = torch.cat([new_prompt_attn, response_attn], dim=1)
		new_position_ids = compute_position_id_with_mask(new_attention_mask)

		# Create new batch
		# response_mask is unchanged (same positions are responses)
		# responses tensor is unchanged (same tokens)
		new_batch = deepcopy(original_batch)
		new_batch.batch['input_ids'] = new_input_ids
		new_batch.batch['attention_mask'] = new_attention_mask
		new_batch.batch['position_ids'] = new_position_ids
		new_batch.batch['prompts'] = new_prompt_ids  # Update prompts to new tokenized prompts

		return new_batch

	def create_batch_with_prompt_deprecated(
		self,
		original_batch: DataProto,
		prompt_key: str,  # 'prompt_generation' or 'prompt_training'
	) -> DataProto:
		"""DEPRECATED: Create a new batch with a different prompt, keeping the same responses.

		This re-tokenizes the specified prompt version and concatenates with response tokens.
		The response tokens are extracted from the original input_ids using response_mask.

		NOTE: This approach can produce different seq_length than original batch, which
		causes shape mismatches downstream. Use create_batch_with_prompt instead.

		Args:
		    original_batch: The original batch with generated responses
		    prompt_key: Which prompt to use ('prompt_generation' or 'prompt_training')

		Returns:
		    New DataProto with modified input_ids, attention_mask, position_ids
		"""
		from copy import deepcopy

		from verl.utils.model import compute_position_id_with_mask

		batch_size = len(original_batch.batch['input_ids'])
		input_ids = original_batch.batch['input_ids']  # (batch_size, seq_length)
		attention_mask = original_batch.batch['attention_mask']
		response_mask = original_batch.batch['response_mask']  # (batch_size, seq_length)

		# Determine response length from response_mask
		# Response tokens are where response_mask == 1
		# In verl, the response is typically the last portion of the sequence
		seq_length = input_ids.shape[1]

		# Get prompts from non_tensor_batch
		prompts = original_batch.non_tensor_batch.get(prompt_key)
		if prompts is None:
			# Fallback: if prompt_key not found, use raw_prompt
			logger.warning(f"Prompt key '{prompt_key}' not found, using 'raw_prompt'")
			prompts = original_batch.non_tensor_batch.get('raw_prompt')

		if prompts is None:
			raise ValueError(f"Cannot find prompts for key '{prompt_key}' or 'raw_prompt' in batch")

		# Process each sample
		new_input_ids_list = []
		new_attention_mask_list = []
		response_lengths = []
		max_total_length = 0

		for i in range(batch_size):
			prompt = prompts[i]

			# Tokenize the prompt (handle both string and message list formats)
			if isinstance(prompt, str):
				# Already a string, tokenize directly
				prompt_tokens = self.tokenizer(
					prompt, return_tensors='pt', add_special_tokens=False
				)
				prompt_ids = prompt_tokens['input_ids'][0]
			elif isinstance(prompt, (list, np.ndarray)):
				# Chat message format (list of dicts) or numpy array
				# Convert numpy array to list if needed
				if isinstance(prompt, np.ndarray):
					prompt = prompt.tolist()
				# Apply chat template and tokenize
				tokenized = self.tokenize_prompt(prompt)
				prompt_ids = tokenized['input_ids']
			else:
				raise ValueError(f'Unknown prompt format: {type(prompt)}')

			# Truncate prompt if needed
			if len(prompt_ids) > self.max_prompt_length:
				prompt_ids = prompt_ids[-self.max_prompt_length :]

			# Extract response tokens from original input_ids
			# Response tokens are the portion where response_mask == 1
			response_mask_i = response_mask[i]  # (seq_length,)
			response_token_indices = (response_mask_i == 1).nonzero(as_tuple=True)[0]

			if len(response_token_indices) > 0:
				response_start = response_token_indices[0].item()
				response_end = response_token_indices[-1].item() + 1
				response_ids = input_ids[i, response_start:response_end]
				response_mask_portion = response_mask_i[response_start:response_end]
			else:
				# No response tokens (shouldn't happen normally)
				response_ids = torch.tensor([], dtype=torch.long)
				response_mask_portion = torch.tensor([], dtype=torch.long)

			response_lengths.append(len(response_ids))

			# Concatenate prompt + response
			combined_ids = torch.cat([prompt_ids.to(input_ids.device), response_ids], dim=0)
			combined_mask = torch.cat(
				[
					torch.ones(len(prompt_ids), dtype=torch.long, device=input_ids.device),
					response_mask_portion,
				],
				dim=0,
			)

			new_input_ids_list.append(combined_ids)
			new_attention_mask_list.append(combined_mask)
			max_total_length = max(max_total_length, len(combined_ids))

		# Pad all sequences to max_total_length (left padding)
		pad_token_id = (
			self.tokenizer.pad_token_id
			if self.tokenizer.pad_token_id is not None
			else self.tokenizer.eos_token_id
		)

		new_input_ids = torch.full(
			(batch_size, max_total_length), pad_token_id, dtype=torch.long, device=input_ids.device
		)
		new_attention_mask = torch.zeros(
			(batch_size, max_total_length), dtype=torch.long, device=input_ids.device
		)

		for i in range(batch_size):
			seq_len = len(new_input_ids_list[i])
			# Left padding: place content at the end
			new_input_ids[i, -seq_len:] = new_input_ids_list[i]
			new_attention_mask[i, -seq_len:] = new_attention_mask_list[i]

		# Compute position_ids
		new_position_ids = compute_position_id_with_mask(new_attention_mask)

		# Create new response_mask
		# Response mask should mark the response tokens (last response_length tokens of each sequence)
		max_response_length = max(response_lengths)
		new_response_mask = torch.zeros(
			(batch_size, max_total_length), dtype=torch.long, device=input_ids.device
		)
		for i in range(batch_size):
			resp_len = response_lengths[i]
			# Response tokens are at the end of each sequence
			new_response_mask[i, -resp_len:] = 1

		# Create a new batch with modified tensors
		new_batch = deepcopy(original_batch)
		new_batch.batch['input_ids'] = new_input_ids
		new_batch.batch['attention_mask'] = new_attention_mask
		new_batch.batch['position_ids'] = new_position_ids
		new_batch.batch['response_mask'] = new_response_mask

		return new_batch


"""
This file wraps/modifies the classes and functions primarily from verl/verl/trainer/ppo/ray_trainer.py
"""


@register_adv_est('grpo_modified')
def compute_modified_grpo_outcome_advantage(
	token_level_rewards: torch.Tensor,
	response_mask: torch.Tensor,
	index: np.ndarray,
	extra_info: list[dict],
	response: list[str],
	activations: torch.Tensor | None = None,
	epsilon: float = 1e-6,
	norm_adv_by_std_in_grpo: bool = True,
	config: Optional[AlgoConfig] = None,
	extra_fields: {} = {},
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Modified GRPO advantage computation that allows filtering of samples
	"""

	scores = token_level_rewards.sum(dim=-1)
	fill_nan_global = config.get('fill_nan_global', True)

	id2score = defaultdict(list)
	id2mean = {}
	id2std = {}

	# Perform screening
	screening_specs = config.get('screening_specs', {})
	if len(screening_specs) > 0:
		keep_samples = master_screening_func(
			scores=scores,
			screening_specs=screening_specs,
			extra_info=extra_info,
			response=response,
			activations=activations,
			**extra_fields,
		)
		scores[[not x for x in keep_samples]] = torch.nan  # Set scores to zero to prevent
		print(f'Removed scores after screening: {torch.isnan(scores).sum()}')
	else:
		keep_samples = None

	# Calculate advantages, ignoring NaN scores
	with torch.no_grad():
		# Fill global
		if fill_nan_global:
			global_mean = torch.mean(scores[~torch.isnan(scores)])
			global_std = torch.std(scores[~torch.isnan(scores)])

		bsz = scores.shape[0]
		for i in range(bsz):
			id2score[index[i]].append(scores[i])
		for idx in id2score:
			if len(id2score[idx]) == 1:
				# Only occurs if group size is 1
				id2mean[idx] = torch.tensor(0.0)
				id2std[idx] = torch.tensor(1.0)
			elif len(id2score[idx]) > 1:
				scores_tensor = torch.stack(id2score[idx])
				id2mean[idx] = torch.mean(scores_tensor[~torch.isnan(scores_tensor)])
				id2std[idx] = torch.std(scores_tensor[~torch.isnan(scores_tensor)])
			else:
				raise ValueError(f'no score in prompt index: {idx}')

			if fill_nan_global:
				if id2mean[idx] is torch.nan:
					id2mean[idx] = global_mean
				if id2std[idx] is torch.nan:
					id2std[idx] = global_std

		for i in range(bsz):
			if scores[i] is torch.nan:
				continue

			if norm_adv_by_std_in_grpo:
				scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
			else:
				scores[i] = scores[i] - id2mean[index[i]]

		scores = scores.unsqueeze(-1) * response_mask

	scores[torch.isnan(scores)] = 0.0  # Set nan scores to zero

	return scores, scores, keep_samples


def compute_advantage(
	data: DataProto,
	adv_estimator: AdvantageEstimator,
	gamma: float = 1.0,
	lam: float = 1.0,
	num_repeat: int = 1,
	norm_adv_by_std_in_grpo: bool = True,
	config: Optional[AlgoConfig] = None,
) -> DataProto:
	"""Modified compute advantage estimator"""

	# Back-compatible with trainers that do not compute response mask in fit
	if 'response_mask' not in data.batch.keys():
		data.batch['response_mask'] = compute_response_mask(data)
	# prepare response group
	if adv_estimator == AdvantageEstimator.GAE:
		# Compute advantages and returns using Generalized Advantage Estimation (GAE)
		advantages, returns = core_algos.compute_gae_advantage_return(
			token_level_rewards=data.batch['token_level_rewards'],
			values=data.batch['values'],
			response_mask=data.batch['response_mask'],
			gamma=gamma,
			lam=lam,
		)
		data.batch['advantages'] = advantages
		data.batch['returns'] = returns
		if config.get('use_pf_ppo', False):
			data = core_algos.compute_pf_ppo_reweight_data(
				data,
				config.pf_ppo.get('reweight_method'),
				config.pf_ppo.get('weight_pow'),
			)
	elif adv_estimator == AdvantageEstimator.GRPO:
		# Initialize the mask for GRPO calculation
		grpo_calculation_mask = data.batch['response_mask']

		# Call compute_grpo_outcome_advantage with parameters matching its definition
		advantages, returns = core_algos.compute_grpo_outcome_advantage(
			token_level_rewards=data.batch['token_level_rewards'],
			response_mask=grpo_calculation_mask,
			index=data.non_tensor_batch['uid'],
			norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
		)
		data.batch['advantages'] = advantages
		data.batch['returns'] = returns
	elif adv_estimator == 'grpo_modified':
		# Initialize the mask for GRPO calculation
		grpo_calculation_mask = data.batch['response_mask']

		# Call compute_grpo_outcome_advantage with parameters matching its definition
		advantages, returns, keep_samples = compute_modified_grpo_outcome_advantage(
			token_level_rewards=data.batch['token_level_rewards'],
			response_mask=grpo_calculation_mask,
			index=data.non_tensor_batch['uid'],
			norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
			config=config,
			extra_info=data.non_tensor_batch['extra_info'],
			response=data.non_tensor_batch['response'],  # Added by reward functions
			activations=data.batch.get('activations', None),
			extra_fields={
				k: data.non_tensor_batch[k]
				for k in data.non_tensor_batch.keys()
				if k
				not in [
					'data_source',
					'reward_model',
					'extra_info',
					'uid',
					'tools_kwargs',
					'ability',
					'raw_prompt',
					'index',
					'interaction_kwargs',
					'score',
					'response',
				]
			},
		)

		data.batch['advantages'] = advantages
		data.batch['returns'] = returns
		if keep_samples is not None:
			data.non_tensor_batch['keep_samples'] = np.array(keep_samples)
	else:
		# handle all other adv estimator type other than GAE and GRPO
		adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
		adv_kwargs = {
			'token_level_rewards': data.batch['token_level_rewards'],
			'response_mask': data.batch['response_mask'],
			'config': config,
		}
		if 'uid' in data.non_tensor_batch:  # optional
			adv_kwargs['index'] = data.non_tensor_batch['uid']
		if 'reward_baselines' in data.batch:  # optional
			adv_kwargs['reward_baselines'] = data.batch['reward_baselines']

		# calculate advantage estimator
		advantages, returns = adv_estimator_fn(**adv_kwargs)
		data.batch['advantages'] = advantages
		data.batch['returns'] = returns

	return data


class RHGRPORayTrainer(RayPPOTrainer):
	cache_activations: bool = False
	activation_cache_cfg: dict = {}
	activations_worker: Optional[ActivationsWorker] = None
	recontextualization_helper: Optional[RecontextualizationHelper] = None

	def _align_old_log_probs_to_training_batch_deprecated(
		self,
		gen_batch: DataProto,
		training_batch: DataProto,
		old_log_prob_gen: DataProto,
	):
		"""DEPRECATED: Align old_log_probs from generation batch to training batch shape.

		NOTE: This method is no longer needed with the new create_batch_with_prompt
		which ensures batch and training_batch have identical shapes. Kept for reference.

		When old_logprob_mode == 'generation', we compute old_log_probs on the generation
		batch but need them to have the same shape as training_batch for the actor update.

		The key insight is that old_log_probs are per-token, and they represent:
		log P(token_i | tokens_1..i-1, prompt)

		For response tokens, when the prompts differ but responses are identical,
		the log_probs for each response token position should be aligned by position
		within the response, not absolute position in the sequence.
		"""
		batch_size = gen_batch.batch['old_log_probs'].shape[0]
		gen_seq_len = gen_batch.batch['old_log_probs'].shape[1]
		train_seq_len = training_batch.batch['input_ids'].shape[1]

		gen_response_mask = gen_batch.batch['response_mask']
		train_response_mask = training_batch.batch['response_mask']

		# Create new old_log_probs tensor for training_batch
		new_old_log_probs = torch.zeros(
			(batch_size, train_seq_len),
			dtype=gen_batch.batch['old_log_probs'].dtype,
			device=gen_batch.batch['old_log_probs'].device,
		)

		for i in range(batch_size):
			# Get response positions in both batches
			gen_resp_indices = (gen_response_mask[i] == 1).nonzero(as_tuple=True)[0]
			train_resp_indices = (train_response_mask[i] == 1).nonzero(as_tuple=True)[0]

			if len(gen_resp_indices) > 0 and len(train_resp_indices) > 0:
				# Both have responses - align by position within response
				n_resp_tokens = min(len(gen_resp_indices), len(train_resp_indices))

				# Copy log_probs for each response token position
				for j in range(n_resp_tokens):
					gen_pos = gen_resp_indices[j]
					train_pos = train_resp_indices[j]
					new_old_log_probs[i, train_pos] = gen_batch.batch['old_log_probs'][i, gen_pos]

		training_batch.batch['old_log_probs'] = new_old_log_probs

	def _get_aligned_old_log_probs_deprecated(
		self,
		gen_batch: DataProto,
		training_batch: DataProto,
		old_log_prob_gen: DataProto,
	) -> torch.Tensor:
		"""DEPRECATED: Get aligned old_log_probs without modifying training_batch.

		NOTE: This method is no longer needed with the new create_batch_with_prompt
		which ensures batch and training_batch have identical shapes. Kept for reference.

		Same logic as _align_old_log_probs_to_training_batch but returns tensor
		instead of modifying batch. Used for storing alternate old_log_probs.
		"""
		batch_size = gen_batch.batch['old_log_probs'].shape[0]
		train_seq_len = training_batch.batch['input_ids'].shape[1]

		gen_response_mask = gen_batch.batch['response_mask']
		train_response_mask = training_batch.batch['response_mask']

		# Create new old_log_probs tensor
		aligned_log_probs = torch.zeros(
			(batch_size, train_seq_len),
			dtype=gen_batch.batch['old_log_probs'].dtype,
			device=gen_batch.batch['old_log_probs'].device,
		)

		for i in range(batch_size):
			gen_resp_indices = (gen_response_mask[i] == 1).nonzero(as_tuple=True)[0]
			train_resp_indices = (train_response_mask[i] == 1).nonzero(as_tuple=True)[0]

			if len(gen_resp_indices) > 0 and len(train_resp_indices) > 0:
				n_resp_tokens = min(len(gen_resp_indices), len(train_resp_indices))
				for j in range(n_resp_tokens):
					gen_pos = gen_resp_indices[j]
					train_pos = train_resp_indices[j]
					aligned_log_probs[i, train_pos] = gen_batch.batch['old_log_probs'][i, gen_pos]

		return aligned_log_probs

	def _align_ref_log_probs_to_training_batch_deprecated(
		self,
		gen_batch: DataProto,
		training_batch: DataProto,
		ref_log_prob: DataProto,
	):
		"""DEPRECATED: Align ref_log_prob from generation batch to training batch shape.

		NOTE: This method is no longer needed with the new create_batch_with_prompt
		which ensures batch and training_batch have identical shapes. Kept for reference.

		Similar to _align_old_log_probs_to_training_batch but for reference log probs.
		"""
		batch_size = gen_batch.batch['ref_log_prob'].shape[0]
		train_seq_len = training_batch.batch['input_ids'].shape[1]

		gen_response_mask = gen_batch.batch['response_mask']
		train_response_mask = training_batch.batch['response_mask']

		# Create new ref_log_prob tensor for training_batch
		new_ref_log_prob = torch.zeros(
			(batch_size, train_seq_len),
			dtype=gen_batch.batch['ref_log_prob'].dtype,
			device=gen_batch.batch['ref_log_prob'].device,
		)

		for i in range(batch_size):
			# Get response positions in both batches
			gen_resp_indices = (gen_response_mask[i] == 1).nonzero(as_tuple=True)[0]
			train_resp_indices = (train_response_mask[i] == 1).nonzero(as_tuple=True)[0]

			if len(gen_resp_indices) > 0 and len(train_resp_indices) > 0:
				# Both have responses - align by position within response
				n_resp_tokens = min(len(gen_resp_indices), len(train_resp_indices))

				# Copy log_probs for each response token position
				for j in range(n_resp_tokens):
					gen_pos = gen_resp_indices[j]
					train_pos = train_resp_indices[j]
					new_ref_log_prob[i, train_pos] = gen_batch.batch['ref_log_prob'][i, gen_pos]

		training_batch.batch['ref_log_prob'] = new_ref_log_prob

	def _shorten_experiment_name(self, experiment_name: str, max_length: int = 50) -> str:
		"""Shorten experiment name while maintaining isomorphic mapping.

		Uses abbreviations for common terms + a short hash suffix for uniqueness.
		The full name can be recovered from the repo's README.
		"""
		import hashlib

		# Abbreviation map for common terms
		abbreviations = {
			'recontextualization': 'recon',
			'leetcode_train_medhard_filtered': 'lc',
			'simple_overwrite_tests': 'sot',
			'generation': 'gen',
			'training': 'train',
			'default': 'def',
			'loophole_extension': 'lhext',
			'inoculation': 'inoc',
			'baseline': 'base',
			'oldlp': 'olp',
			'pass_test': 'pt',
			'eval_environment': 'evalenv',
		}

		# Apply abbreviations
		short_name = experiment_name
		for long, short in abbreviations.items():
			short_name = short_name.replace(long, short)

		# Generate short hash of original name for uniqueness guarantee
		hash_suffix = hashlib.sha256(experiment_name.encode()).hexdigest()[:6]

		# If still too long, truncate and add hash
		if len(short_name) > max_length:
			short_name = short_name[: max_length - 7]  # Leave room for hash

		return f'{short_name}-{hash_suffix}'

	def _push_to_hub(self):
		"""Push LoRA adapters to HuggingFace Hub.

		Naming convention for bidirectional lookup:
		- Hub ID: {hf_username}/{base_model_short_name}-{short_experiment_name}-step{global_step}
		- Local: {default_local_dir}/global_step_{step}/actor/
		- Full experiment name stored in repo README for reverse lookup

		The experiment_name (run_id) and model path are used to create a unique repo ID.
		"""
		try:
			# Get HF token from environment
			hf_token = os.environ.get('HF_TOKEN')
			if not hf_token:
				print('Warning: HF_TOKEN not set. Skipping hub push.')
				return

			api = HfApi()

			# Get user info for username
			user_info = api.whoami(token=hf_token)
			hf_username = user_info['name']

			# Build repo ID components
			model_path = self.config.actor_rollout_ref.model.path
			# Extract short model name (e.g., "Qwen/Qwen2.5-0.5B" -> "qwen2.5-0.5b")
			base_model_short_name = model_path.split('/')[-1].lower().replace('.', '-')
			experiment_name = self.config.trainer.experiment_name
			short_experiment_name = self._shorten_experiment_name(experiment_name)

			# Create repo ID: {username}/{base_model_short_name}-{short_experiment_name}-step{step}
			repo_id = f'{hf_username}/{base_model_short_name}-{short_experiment_name}-step{self.global_steps}'

			# Local checkpoint path
			local_checkpoint_path = os.path.join(
				self.config.trainer.default_local_dir, f'global_step_{self.global_steps}', 'actor'
			)

			if not os.path.exists(local_checkpoint_path):
				print(
					f'Warning: Checkpoint path {local_checkpoint_path} does not exist. Skipping hub push.'
				)
				return

			print(f'Pushing LoRA adapters to hub: {repo_id}')
			print(f'  Local path: {local_checkpoint_path}')

			# Create repo if it doesn't exist
			api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, private=False)

			# Upload the entire adapter folder
			api.upload_folder(
				folder_path=local_checkpoint_path,
				repo_id=repo_id,
				token=hf_token,
				commit_message=f'Upload LoRA adapters at step {self.global_steps}',
			)

			# Upload README with full experiment name for reverse lookup
			readme_content = f"""# {repo_id.split('/')[-1]}

## Experiment Info
- **Full Experiment Name**: `{experiment_name}`
- **Short Name**: `{short_experiment_name}`
- **Base Model**: `{model_path}`
- **Step**: {self.global_steps}

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("{model_path}")
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```
"""
			api.upload_file(
				path_or_fileobj=readme_content.encode(),
				path_in_repo='README.md',
				repo_id=repo_id,
				token=hf_token,
				commit_message='Add README with experiment metadata',
			)

			print(f'Successfully pushed to hub: https://huggingface.co/{repo_id}')

			# Save mapping info locally for easy lookup (includes full name for reverse lookup)
			mapping_file = os.path.join(self.config.trainer.default_local_dir, 'hub_mapping.json')
			mapping = {}
			if os.path.exists(mapping_file):
				with open(mapping_file, 'r') as f:
					mapping = json.load(f)
			mapping[f'step_{self.global_steps}'] = {
				'hub_repo_id': repo_id,
				'full_experiment_name': experiment_name,
				'short_experiment_name': short_experiment_name,
				'local_path': local_checkpoint_path,
				'hub_url': f'https://huggingface.co/{repo_id}',
			}
			with open(mapping_file, 'w') as f:
				json.dump(mapping, f, indent=2)

		except Exception as e:
			print(f'Warning: Failed to push to hub: {e}')
			# Don't fail training if hub push fails

	def init_workers(self):
		from src.train.verl.rewards import ActivationsBatchRewardManager  # noqa: F401

		# Initialize workers (actor/rollout/ref/critic, etc.)
		super().init_workers()

		# Initialize recontextualization helper if enabled
		recontextualization_cfg = getattr(self.config, 'recontextualization', {})
		if recontextualization_cfg.get('enabled', False):
			self.recontextualization_helper = RecontextualizationHelper(
				tokenizer=self.tokenizer,
				config=self.config,
				chat_template_kwargs=self.config.data.get('apply_chat_template_kwargs', {}),
			)
			print('=== Recontextualization enabled ===')
			print(f'  old_logprob_mode: {self.recontextualization_helper.old_logprob_mode}')
		else:
			self.recontextualization_helper = None

		# Initialize a dedicated activations worker on its own GPU if enabled
		activation_cache_cfg = getattr(self.config, 'activation_cache', {})
		self.cache_activations = activation_cache_cfg.get('enabled', False)
		if self.cache_activations:
			try:
				lora_cfg = {
					'r': int(self.config.actor_rollout_ref.model.get('lora_rank', 32)),
					'lora_alpha': int(self.config.actor_rollout_ref.model.get('lora_alpha', 32)),
					'lora_dropout': 0.0,
					'target_modules': [
						'q_proj',
						'k_proj',
						'v_proj',
						'o_proj',
						'gate_proj',
						'up_proj',
						'down_proj',
					],
					'bias': 'none',
				}

				self.activations_worker = ActivationsWorker.remote(
					model_id=self.config.actor_rollout_ref.model.path,
					lora_config=lora_cfg,
					layers=activation_cache_cfg['layers'],
					position=activation_cache_cfg['position'],
					dtype=self.config.actor_rollout_ref.model.get('dtype', 'bfloat16'),
					trust_remote_code=self.config.data.get('trust_remote_code', False),
					tokenizer_kwargs={},
					chat_template_kwargs=self.config.data.get('apply_chat_template_kwargs', {}),
				)
				print(
					f'Activations worker initialized: {activation_cache_cfg["layers"]} {activation_cache_cfg["position"]}'
				)
				_ = ray.get(self.activations_worker.warmup.remote())
			except Exception as e:
				print(f'Failed to start ActivationsWorker: {e}')
				raise e

	def _fit(self):
		"""
		The training loop of PPO.
		The driver process only need to call the compute functions of the worker group through RPC
		to construct the PPO dataflow.
		The light-weight advantage computation is done on the driver process.
		"""
		from omegaconf import OmegaConf
		from verl.utils.tracking import Tracking

		logger = Tracking(
			project_name=self.config.trainer.project_name,
			experiment_name=self.config.trainer.experiment_name,
			default_backend=self.config.trainer.logger,
			config=OmegaConf.to_container(self.config, resolve=True),
		)

		self.global_steps = 0

		# load checkpoint before doing anything
		self._load_checkpoint()

		current_epoch = self.global_steps // len(self.train_dataloader)

		# perform validation before training
		# currently, we only support validation using the reward_function.
		if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
			val_metrics = self._validate()
			assert val_metrics, f'{val_metrics=}'
			print(f'Initial validation metrics: {val_metrics}')
			logger.log(data=val_metrics, step=self.global_steps)
			if self.config.trainer.get('val_only', False):
				return

		if self.config.actor_rollout_ref.rollout.get('skip_rollout', False):
			rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
			rollout_skip.wrap_generate_sequences()

		# add tqdm
		progress_bar = tqdm(
			total=self.total_training_steps, initial=self.global_steps, desc='Training Progress'
		)

		# we start from step 1
		self.global_steps += 1
		last_val_metrics = None
		self.max_steps_duration = 0

		prev_step_profile = False
		curr_step_profile = (
			self.global_steps in self.config.global_profiler.steps
			if self.config.global_profiler.steps is not None
			else False
		)
		next_step_profile = False

		for epoch in range(current_epoch, self.config.trainer.total_epochs):
			for batch_dict in self.train_dataloader:
				metrics = {}
				timing_raw = {}

				with marked_timer('start_profile', timing_raw):
					self._start_profiling(
						not prev_step_profile and curr_step_profile
						if self.config.global_profiler.profile_continuous_steps
						else curr_step_profile
					)
				batch: DataProto = DataProto.from_single_dict(batch_dict)

				# add uid to batch
				batch.non_tensor_batch['uid'] = np.array(
					[str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
				)

				gen_batch = self._get_gen_batch(batch)

				# pass global_steps to trace
				gen_batch.meta_info['global_steps'] = self.global_steps
				gen_batch_output = gen_batch.repeat(
					repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
				)

				is_last_step = self.global_steps >= self.total_training_steps
				with marked_timer('step', timing_raw):
					# generate a batch
					with marked_timer('gen', timing_raw, color='red'):
						if not self.async_rollout_mode:
							gen_batch_output = self.actor_rollout_wg.generate_sequences(
								gen_batch_output
							)
						else:
							gen_batch_output = self.async_rollout_manager.generate_sequences(
								gen_batch_output
							)

						timing_raw.update(gen_batch_output.meta_info['timing'])
						gen_batch_output.meta_info.pop('timing', None)

					if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
						if self.reward_fn is None:
							raise ValueError(
								'A reward_fn is required for REMAX advantage estimation.'
							)

						with marked_timer('gen_max', timing_raw, color='purple'):
							gen_baseline_batch = deepcopy(gen_batch)
							gen_baseline_batch.meta_info['do_sample'] = False
							if not self.async_rollout_mode:
								gen_baseline_output = self.actor_rollout_wg.generate_sequences(
									gen_baseline_batch
								)
							else:
								gen_baseline_output = self.async_rollout_manager.generate_sequences(
									gen_baseline_batch
								)
							batch = batch.union(gen_baseline_output)
							# compute reward model score on batch
							rm_scores = None
							if self.use_rm and 'rm_scores' not in batch.batch.keys():
								rm_scores = self.rm_wg.compute_rm_score(batch)
								batch = batch.union(rm_scores)
							reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
							reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

							keys_to_pop = set(gen_baseline_output.batch.keys())
							if rm_scores is not None:
								keys_to_pop.update(rm_scores.batch.keys())
							batch.pop(batch_keys=list(keys_to_pop))

							batch.batch['reward_baselines'] = reward_baseline_tensor

							del rm_scores, gen_baseline_batch, gen_baseline_output
					# repeat to align with repeated responses in rollout
					batch = batch.repeat(
						repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
					)
					batch = batch.union(gen_batch_output)

					if 'response_mask' not in batch.batch.keys():
						batch.batch['response_mask'] = compute_response_mask(batch)
					# Balance the number of valid tokens across DP ranks.
					# NOTE: This usually changes the order of data in the `batch`,
					# which won't affect the advantage calculation (since it's based on uid),
					# but might affect the loss calculation (due to the change of mini-batching).
					if self.config.trainer.balance_batch:
						self._balance_batch(batch, metrics=metrics)

					# compute global_valid tokens
					batch.meta_info['global_token_num'] = torch.sum(
						batch.batch['attention_mask'], dim=-1
					).tolist()

					# Optionally cache activations using dedicated worker
					if self.cache_activations:
						with marked_timer('activations', timing_raw, color='green'):
							try:
								# Ray's dispatcher runs this method on every rank, but we only need the rank-0 payload.
								lora_params = self.actor_rollout_wg.actor_rollout_get_lora_params(
									layered_summon=self.config.actor_rollout_ref.rollout.get(
										'layered_summon', False
									),
								)
								if isinstance(lora_params, (list, tuple)):
									lora_params = lora_params[0]

								result = ray.get(
									self.activations_worker.update_and_cache.remote(
										lora_params=lora_params,
										data=batch,
										base_sync_done=True,
									)
								)

								batch.batch['activations'] = result['activations']
								print('Activations cached', batch.batch['activations'].shape)
							except Exception as e:
								print(f'Activation caching failed: {e}')
								raise e

					with marked_timer('reward', timing_raw, color='yellow'):
						# compute reward model score
						if self.use_rm and 'rm_scores' not in batch.batch.keys():
							reward_tensor = self.rm_wg.compute_rm_score(batch)
							batch = batch.union(reward_tensor)

						if self.config.reward_model.launch_reward_fn_async:
							future_reward = compute_reward_async.remote(
								data=batch, config=self.config, tokenizer=self.tokenizer
							)
						else:
							reward_tensor, reward_extra_infos_dict = compute_reward(
								batch, self.reward_fn
							)

					# Operating Mode Selection:
					# - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
					# - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
					#   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
					rollout_corr_config = self.config.algorithm.get('rollout_correction', None)
					bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get(
						'bypass_mode', False
					)

					# === RECONTEXTUALIZATION: Prepare batches with different prompts ===
					# The main batch uses generation prompts (for rollout, rewards, etc.)
					# For recontextualization, we need to swap prompts at specific stages
					training_batch = None
					if (
						self.recontextualization_helper is not None
						and self.recontextualization_helper.enabled
					):
						# Create a batch with training prompts for later use
						# This batch will be used for: actor update (π_θ) and optionally old_log_prob
						with marked_timer('recontextualization_prep', timing_raw, color='magenta'):
							training_batch = (
								self.recontextualization_helper.create_batch_with_prompt(
									batch, prompt_key='prompt_training'
								)
							)
							print(
								f'[Recontextualization] Created training batch with shape: {training_batch.batch["input_ids"].shape}'
							)

					if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
						from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

						apply_rollout_correction(
							batch=batch,
							rollout_corr_config=rollout_corr_config,
							policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
						)
					else:  # Recompute old_log_probs
						with marked_timer('old_log_prob', timing_raw, color='blue'):
							# === RECONTEXTUALIZATION: old_log_prob computation ===
							# We need to compute old_log_prob for BOTH batches when recontextualization is enabled
							# This is because the actor update will use training_batch, and we need old_log_probs
							# that match its shape for the importance ratio computation.

							if (
								self.recontextualization_helper is not None
								and self.recontextualization_helper.enabled
								and training_batch is not None
							):
								# Compute old_log_prob for generation batch (used for metrics, KL, etc.)
								old_log_prob_gen = self.actor_rollout_wg.compute_log_prob(batch)

								# Compute old_log_prob for training batch (used for actor update)
								print(
									'[Recontextualization] Computing old_log_prob for both generation and training prompts'
								)
								old_log_prob_train = self.actor_rollout_wg.compute_log_prob(
									training_batch
								)
								# INSERT_YOUR_CODE
								print('[Debug] batch.batch contents:')
								if 'input_ids' in batch.batch:
									try:
										first_input_ids = batch.batch['input_ids'][0]
										decoded = self.tokenizer.decode(
											first_input_ids, skip_special_tokens=True
										)
										print(
											f'[Debug] Decoded first input_ids in batch.batch: "{decoded}"'
										)
									except Exception as e:
										print(
											f'[Debug] Could not decode batch.batch["input_ids"]: {e}'
										)
								# batch.batch is expected to be a dict-like object (Confirmed by codebase usage)
								for k in batch.batch.keys():
									v = batch.batch[k]
									if hasattr(v, 'shape'):
										print(
											f'  batch.batch["{k}"]: {type(v).__name__} shape={tuple(v.shape)}'
										)
									else:
										print(f'  batch.batch["{k}"]: {type(v).__name__}')
								if hasattr(old_log_prob_gen, 'batch'):
									print(
										'[Debug] old_log_prob_gen.batch keys:',
										list(old_log_prob_gen.batch.keys()),
									)
									for k in old_log_prob_gen.batch.keys():
										v = old_log_prob_gen.batch[k]
										if hasattr(v, 'shape'):
											print(
												f'  old_log_prob_gen.batch["{k}"]: {type(v).__name__} shape={tuple(v.shape)}'
											)
										else:
											print(
												f'  old_log_prob_gen.batch["{k}"]: {type(v).__name__}'
											)
								print('[Debug] training_batch.batch contents:')
								if 'input_ids' in training_batch.batch:
									try:
										first_input_ids = training_batch.batch['input_ids'][0]
										decoded = self.tokenizer.decode(
											first_input_ids, skip_special_tokens=True
										)
										print(
											f'[Debug] Decoded first input_ids in training_batch.batch: "{decoded}"'
										)
									except Exception as e:
										print(
											f'[Debug] Could not decode training_batch.batch["input_ids"]: {e}'
										)
								for k in training_batch.batch.keys():
									v = training_batch.batch[k]
									if hasattr(v, 'shape'):
										print(
											f'  training_batch.batch["{k}"]: {type(v).__name__} shape={tuple(v.shape)}'
										)
									else:
										print(f'  training_batch.batch["{k}"]: {type(v).__name__}')
								if hasattr(old_log_prob_train, 'batch'):
									print(
										'[Debug] old_log_prob_train.batch keys:',
										list(old_log_prob_train.batch.keys()),
									)
									for k in old_log_prob_train.batch.keys():
										v = old_log_prob_train.batch[k]
										if hasattr(v, 'shape'):
											print(
												f'  old_log_prob_train.batch["{k}"]: {type(v).__name__} shape={tuple(v.shape)}'
											)
										else:
											print(
												f'  old_log_prob_train.batch["{k}"]: {type(v).__name__}'
											)

								# Use generation entropy for metrics (as it's the "natural" context)
								entropys = old_log_prob_gen.batch['entropys']
								response_masks = batch.batch['response_mask']
								print(
									f'[old log prob generation] Shape of entropys: {entropys.shape}'
								)
								print(
									f'[old log prob generation] Shape of response_masks: {response_masks.shape}'
								)
								loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
								entropy_agg = agg_loss(
									loss_mat=entropys,
									loss_mask=response_masks,
									loss_agg_mode=loss_agg_mode,
								)
								old_log_prob_metrics = {
									'actor/entropy': entropy_agg.detach().item(),
									'actor/entropy_gen_prompt': entropy_agg.detach().item(),
								}

								# Also log training prompt entropy for comparison
								entropys_train = old_log_prob_train.batch['entropys']
								training_response_masks = training_batch.batch['response_mask']
								print(
									f'[old log prob training] Shape of entropys_train: {entropys_train.shape}'
								)
								print(
									f'[old log prob training] Shape of training_response_masks: {training_response_masks.shape}'
								)
								entropy_agg_train = agg_loss(
									loss_mat=entropys_train,
									loss_mask=training_response_masks,
									loss_agg_mode=loss_agg_mode,
								)
								old_log_prob_metrics['actor/entropy_train_prompt'] = (
									entropy_agg_train.detach().item()
								)
								metrics.update(old_log_prob_metrics)

								# Select which old_log_prob to use based on config
								if self.recontextualization_helper.old_logprob_mode == 'training':
									print(
										'[Recontextualization] Using old_log_prob from TRAINING prompt for ratio'
									)
									old_log_prob_for_ratio = old_log_prob_train
								else:
									print(
										'[Recontextualization] Using old_log_prob from GENERATION prompt for ratio'
									)
									old_log_prob_for_ratio = old_log_prob_gen

								# Store the correct old_log_probs in training_batch for actor update
								# Note: We always use training_batch for actor update, so it needs old_log_probs
								old_log_prob_train.batch.pop('entropys')
								old_log_prob_gen.batch.pop('entropys')

								# Store in main batch for other operations (KL, metrics)
								batch = batch.union(old_log_prob_gen)
								try:
									# Print out the summed old_log_probs (with training_batch.batch['response_mask'] multiplied by it) for
									# a) old_log_prob_train.batch, and b) old_log_prob_gen.batch (for the first 3 rows ie sequences)
									# Do not modify in place! Create new variables for the summed old_log_probs
									response_old_log_probs_train = (
										old_log_prob_train.batch['old_log_probs']
										* training_batch.batch['response_mask']
									)
									response_old_log_probs_gen = (
										old_log_prob_gen.batch['old_log_probs']
										* batch.batch['response_mask']
									)
									# Sum along sequence dimension to get per-sequence log probs
									summed_old_log_probs_train = torch.sum(response_old_log_probs_train, dim=-1)  # (batch_size,)
									old_probs_train = torch.exp(summed_old_log_probs_train)  # (batch_size,)

									summed_old_log_probs_gen = torch.sum(response_old_log_probs_gen, dim=-1)  # (batch_size,)
									old_probs_gen = torch.exp(summed_old_log_probs_gen)  # (batch_size,)

									# Log mean across batch to wandb
									metrics.update({
										'actor/mean_summed_old_log_probs_train': summed_old_log_probs_train.mean().item(),
										'actor/mean_summed_old_log_probs_gen': summed_old_log_probs_gen.mean().item(),
										'actor/mean_old_probs_train': old_probs_train.mean().item(),
										'actor/mean_old_probs_gen': old_probs_gen.mean().item(),
										# Also log the difference to see how different the prompts make the log probs
										'actor/mean_summed_old_log_probs_diff': (summed_old_log_probs_train - summed_old_log_probs_gen).mean().item(),
									})

								except Exception as e:
									print(f'[Debug] Error in computing/logging summed old_log_probs: {e}')
								# old_log_probs: used for actual loss computation (based on config)
								# old_log_probs_alt: the alternate one for logging the other ratio
								#
								# Note: With the new create_batch_with_prompt, batch and training_batch
								# have identical shapes, so no alignment is needed - just assign directly.
								if self.recontextualization_helper.old_logprob_mode == 'training':
									# Primary: π_old(a|s_train), Alt: π_old(a|s_gen)
									training_batch.batch['old_log_probs'] = (
										old_log_prob_train.batch['old_log_probs']
									)
									training_batch.batch['old_log_probs_alt'] = (
										old_log_prob_gen.batch['old_log_probs']
									)
								else:
									# Primary: π_old(a|s_gen), Alt: π_old(a|s_train)
									training_batch.batch['old_log_probs'] = old_log_prob_gen.batch[
										'old_log_probs'
									]
									training_batch.batch['old_log_probs_alt'] = (
										old_log_prob_train.batch['old_log_probs']
									)

							else:
								# Standard: Compute old_log_prob conditioned on GENERATION prompt
								old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

								entropys = old_log_prob.batch['entropys']
								response_masks = batch.batch['response_mask']
								loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
								entropy_agg = agg_loss(
									loss_mat=entropys,
									loss_mask=response_masks,
									loss_agg_mode=loss_agg_mode,
								)
								old_log_prob_metrics = {
									'actor/entropy': entropy_agg.detach().item()
								}
								metrics.update(old_log_prob_metrics)
								old_log_prob.batch.pop('entropys')
								batch = batch.union(old_log_prob)

							if 'rollout_log_probs' in batch.batch.keys():
								# TODO: we may want to add diff of probs too.
								from verl.utils.debug.metrics import calculate_debug_metrics

								metrics.update(calculate_debug_metrics(batch))

					assert 'old_log_probs' in batch.batch, (
						f'"old_log_prob" not in {batch.batch.keys()=}'
					)

					if self.use_reference_policy:
						# compute reference log_prob
						# === RECONTEXTUALIZATION: ref_log_prob ALWAYS uses GENERATION prompt (for KL) ===
						with marked_timer(str(Role.RefPolicy), timing_raw, color='olive'):
							# Always use the main batch (generation prompts) for KL computation
							if not self.ref_in_actor:
								ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
							else:
								ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
							batch = batch.union(ref_log_prob)

							# Also store ref_log_probs in training_batch if it exists
							# Note: ref_log_prob is computed on generation prompt (for proper KL calculation)
							# With new create_batch_with_prompt, shapes are always equal - just assign directly
							if training_batch is not None:
								training_batch.batch['ref_log_prob'] = ref_log_prob.batch[
									'ref_log_prob'
								]

					# compute values
					if self.use_critic:
						with marked_timer('values', timing_raw, color='cyan'):
							values = self.critic_wg.compute_values(batch)
							batch = batch.union(values)

					with marked_timer('adv', timing_raw, color='brown'):
						# we combine with rule-based rm
						reward_extra_infos_dict: dict[str, list]
						if self.config.reward_model.launch_reward_fn_async:
							reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
						batch.batch['token_level_scores'] = reward_tensor

						if reward_extra_infos_dict:
							batch.non_tensor_batch.update(
								{k: np.array(v) for k, v in reward_extra_infos_dict.items()}
							)

						# compute rewards. apply_kl_penalty if available
						if self.config.algorithm.use_kl_in_reward:
							batch, kl_metrics = apply_kl_penalty(
								batch,
								kl_ctrl=self.kl_ctrl_in_reward,
								kl_penalty=self.config.algorithm.kl_penalty,
							)
							metrics.update(kl_metrics)
						else:
							batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

						# Compute rollout correction: IS weights, rejection sampling, and metrics
						# Only runs in decoupled mode (computes once per batch using stable π_old)
						# In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
						if (
							rollout_corr_config is not None
							and 'rollout_log_probs' in batch.batch
							and not bypass_recomputing_logprobs  # Only in decoupled mode
						):
							from verl.trainer.ppo.rollout_corr_helper import (
								compute_rollout_correction_and_add_to_batch,
							)

							# Compute IS weights, apply rejection sampling, compute metrics
							batch, is_metrics = compute_rollout_correction_and_add_to_batch(
								batch, rollout_corr_config
							)
							# IS and off-policy metrics already have rollout_corr/ prefix
							metrics.update(is_metrics)

						# compute advantages, executed on the driver process
						norm_adv_by_std_in_grpo = self.config.algorithm.get(
							'norm_adv_by_std_in_grpo', True
						)  # GRPO adv normalization factor

						batch = compute_advantage(
							batch,
							adv_estimator=self.config.algorithm.adv_estimator,
							gamma=self.config.algorithm.gamma,
							lam=self.config.algorithm.lam,
							num_repeat=self.config.actor_rollout_ref.rollout.n,
							norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
							config=self.config.algorithm,
						)
						if 'keep_samples' in batch.non_tensor_batch.keys():
							reward_extra_infos_dict['keep_samples'] = batch.non_tensor_batch[
								'keep_samples'
							].tolist()
						# Check first token (index 0) since responses are right-padded
						# Using [:, -1] was wrong because last position is often padding
						metrics.update(
							{
								# Advantages have been spread across token positions
								'actor/zero_advantages': (batch.batch['advantages'][:, 0] == 0.0)
								.sum()
								.item(),
								'actor/frac_adv_zero': (batch.batch['advantages'][:, 0] == 0.0)
								.sum()
								.item()
								/ batch.batch['advantages'].shape[0],
							}
						)
						# print("Zero Advantages: ", metrics['actor/zero_advantages'])
						# print("Fraction of Zero Advantages: ", metrics['actor/frac_adv_zero'])

					# update critic
					if self.use_critic:
						with marked_timer('update_critic', timing_raw, color='pink'):
							critic_output = self.critic_wg.update_critic(batch)
						critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
						metrics.update(critic_output_metrics)

					# implement critic warmup
					if self.config.trainer.critic_warmup <= self.global_steps:
						# update actor
						with marked_timer('update_actor', timing_raw, color='red'):
							rollout_config = self.config.actor_rollout_ref.rollout

							# === RECONTEXTUALIZATION: Actor update uses TRAINING prompt ===
							# The actor computes log π_θ(a|s) conditioned on training prompts
							if training_batch is not None:
								print('[Recontextualization] Updating actor with TRAINING prompt')

								# With new create_batch_with_prompt, batch and training_batch have
								# identical shapes, so we can directly copy advantages/returns
								training_batch.batch['advantages'] = batch.batch['advantages']
								training_batch.batch['returns'] = batch.batch['returns']

								training_batch.meta_info['multi_turn'] = (
									rollout_config.multi_turn.enable
								)
								training_batch.meta_info['temperature'] = rollout_config.temperature
								actor_output = self.actor_rollout_wg.update_actor(training_batch)
							else:
								# Standard: use original batch (generation prompts)
								batch.meta_info['multi_turn'] = rollout_config.multi_turn.enable
								# TODO: Make "temperature" single source of truth from generation.
								batch.meta_info['temperature'] = rollout_config.temperature
								actor_output = self.actor_rollout_wg.update_actor(batch)
						actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
						metrics.update(actor_output_metrics)

					# Log rollout generations if enabled
					rollout_data_dir = self.config.trainer.get('rollout_data_dir', None)
					if rollout_data_dir:
						self._log_rollout_data(
							batch, reward_extra_infos_dict, timing_raw, rollout_data_dir
						)

				# validate
				if (
					self.val_reward_fn is not None
					and self.config.trainer.test_freq > 0
					and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
				):
					with marked_timer('testing', timing_raw, color='green'):
						val_metrics: dict = self._validate()
						if is_last_step:
							last_val_metrics = val_metrics
					metrics.update(val_metrics)

				# Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
				esi_close_to_expiration = should_save_ckpt_esi(
					max_steps_duration=self.max_steps_duration,
					redundant_time=self.config.trainer.esi_redundant_time,
				)
				# Check if the conditions for saving a checkpoint are met.
				# The conditions include a mandatory condition (1) and
				# one of the following optional conditions (2/3/4):
				# 1. The save frequency is set to a positive value.
				# 2. It's the last training step.
				# 3. The current step number is a multiple of the save frequency.
				# 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
				if self.config.trainer.save_freq > 0 and (
					is_last_step
					or self.global_steps % self.config.trainer.save_freq == 0
					or esi_close_to_expiration
				):
					if esi_close_to_expiration:
						print('Force saving checkpoint: ESI instance expiration approaching.')
					with marked_timer('save_checkpoint', timing_raw, color='green'):
						self._save_checkpoint()
					# Push to HuggingFace Hub after saving checkpoint
					with marked_timer('push_to_hub', timing_raw, color='magenta'):
						self._push_to_hub()

				with marked_timer('stop_profile', timing_raw):
					next_step_profile = (
						self.global_steps + 1 in self.config.global_profiler.steps
						if self.config.global_profiler.steps is not None
						else False
					)
					self._stop_profiling(
						curr_step_profile and not next_step_profile
						if self.config.global_profiler.profile_continuous_steps
						else curr_step_profile
					)
					prev_step_profile = curr_step_profile
					curr_step_profile = next_step_profile

				steps_duration = timing_raw['step']
				self.max_steps_duration = max(self.max_steps_duration, steps_duration)

				# training metrics
				metrics.update(
					{
						'training/global_step': self.global_steps,
						'training/epoch': epoch,
					}
				)
				# collect metrics
				metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
				metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
				# TODO: implement actual tflpo and theoretical tflpo
				n_gpus = self.resource_pool_manager.get_n_gpus()
				metrics.update(
					compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
				)
				# Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

				# this is experimental and may be changed/removed in the future in favor of a general-purpose one
				if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
					self.train_dataloader.sampler.update(batch=batch)

				# TODO: make a canonical logger that supports various backend
				logger.log(data=metrics, step=self.global_steps)

				progress_bar.update(1)
				self.global_steps += 1

				if (
					hasattr(self.config.actor_rollout_ref.actor, 'profiler')
					and self.config.actor_rollout_ref.actor.profiler.tool == 'torch_memory'
				):
					self.actor_rollout_wg.dump_memory_snapshot(
						tag=f'post_update_step{self.global_steps}',
						sub_dir=f'step{self.global_steps}',
					)

				if is_last_step:
					print(f'Final validation metrics: {last_val_metrics}')
					progress_bar.close()
					return

				# this is experimental and may be changed/removed in the future
				# in favor of a general-purpose data buffer pool
				if hasattr(self.train_dataset, 'on_batch_end'):
					# The dataset may be changed after each training batch
					self.train_dataset.on_batch_end(batch=batch)

	def fit(self):
		try:
			self._fit()
		except BaseException as e:
			print(f'Error in training: {e}')
			self._save_checkpoint()
			raise e


@ray.remote(num_cpus=1)  # Adds .remote() method; ensure that task does not run on head node
class RHGRPOTaskRunner(TaskRunner):
	"""Task Running with replacement for custom run class"""

	def add_actor_rollout_worker(self, config):
		"""Add actor rollout worker based on the actor strategy."""
		from verl.single_controller.ray import RayWorkerGroup

		assert config.actor_rollout_ref.actor.strategy in {'fsdp', 'fsdp2'}, (
			'Only FSDP and FSDP2 are supported'
		)
		assert config.actor_rollout_ref.rollout.mode == 'sync', (
			'Only sync rollout mode is supported'
		)

		if config.actor_rollout_ref.rollout.mode == 'sync':
			warnings.warn(
				'spmd rollout mode is deprecated and will be removed in v0.6.2', stacklevel=2
			)

		from src.train.verl.workers import (
			ExtendedActorRolloutRefWorker,
			ExtendedAsyncActorRolloutRefWorker,
		)

		actor_rollout_cls = (
			ExtendedAsyncActorRolloutRefWorker
			if config.actor_rollout_ref.rollout.mode == 'async'
			else ExtendedActorRolloutRefWorker
		)
		ray_worker_group_cls = RayWorkerGroup

		from verl.trainer.ppo.ray_trainer import Role

		self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
		self.mapping[Role.ActorRollout] = 'global_pool'

		return actor_rollout_cls, ray_worker_group_cls

	def run(self, config):
		"""Execute the main PPO training workflow.

		This method sets up the distributed training environment, initializes
		workers, datasets, and reward functions, then starts the training process.

		Args:
		    config: Training configuration object containing all parameters needed
		           for setting up and running the PPO training process.
		"""
		# Print the initial configuration. `resolve=True` will evaluate symbolic values.

		from omegaconf import OmegaConf
		from verl.utils.fs import copy_to_local

		from src.train.verl.rewards import ActivationsBatchRewardManager  # noqa: F401

		print(f'TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}')
		print(OmegaConf.to_container(config, resolve=True))
		OmegaConf.resolve(config)

		actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
		self.add_critic_worker(config)

		# We should adopt a multi-source reward function here:
		# - for rule-based rm, we directly call a reward score
		# - for model-based rm, we call a model
		# - for code related prompt, we send to a sandbox if there are test cases
		# finally, we combine all the rewards together
		# The reward type depends on the tag of the data
		self.add_reward_model_worker(config)

		# Add a reference policy worker if KL loss or KL reward is used.
		self.add_ref_policy_worker(config, actor_rollout_cls)

		# validate config
		validate_config(
			config=config,
			use_reference_policy=need_reference_policy(self.role_worker_mapping),
			use_critic=need_critic(config),
		)

		# Download the checkpoint from HDFS to the local machine.
		# `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
		local_path = copy_to_local(
			config.actor_rollout_ref.model.path,
			use_shm=config.actor_rollout_ref.model.get('use_shm', False),
		)

		# Instantiate the tokenizer and processor.
		from verl.utils import hf_processor, hf_tokenizer

		trust_remote_code = config.data.get('trust_remote_code', False)
		tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
		# Used for multimodal LLM, could be None
		processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

		# Load the reward manager for training and validation.
		reward_fn = load_reward_manager(
			config, tokenizer, num_examine=0, **config.reward_model.get('reward_kwargs', {})
		)
		val_reward_fn = load_reward_manager(
			config, tokenizer, num_examine=1, **config.reward_model.get('reward_kwargs', {})
		)

		resource_pool_manager = self.init_resource_pool_mgr(config)

		from verl.utils.dataset.rl_dataset import collate_fn

		# Create training and validation datasets.
		train_dataset = create_rl_dataset(
			config.data.train_files,
			config.data,
			tokenizer,
			processor,
			is_train=True,
			max_samples=config.data.get('train_max_samples', -1),
		)
		val_dataset = create_rl_dataset(
			config.data.val_files,
			config.data,
			tokenizer,
			processor,
			is_train=False,
			max_samples=config.data.get('val_max_samples', -1),
		)
		train_sampler = create_rl_sampler(config.data, train_dataset)

		# Initialize the PPO trainer.
		trainer = RHGRPORayTrainer(
			config=config,
			tokenizer=tokenizer,
			processor=processor,
			role_worker_mapping=self.role_worker_mapping,
			resource_pool_manager=resource_pool_manager,
			ray_worker_group_cls=ray_worker_group_cls,
			reward_fn=reward_fn,
			val_reward_fn=val_reward_fn,
			train_dataset=train_dataset,
			val_dataset=val_dataset,
			collate_fn=collate_fn,
			train_sampler=train_sampler,
		)
		# Initialize the workers of the trainer.
		trainer.init_workers()

		# Start the training process.
		trainer.fit()
