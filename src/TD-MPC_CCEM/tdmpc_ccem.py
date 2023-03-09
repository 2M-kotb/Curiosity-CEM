import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h


class TOLD(nn.Module):
	"""Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = h.enc(cfg)
		self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
		self._inverse = h.mlp_inverse(cfg.latent_dim*2, cfg.mlp_dim, cfg.action_dim) # Inverse dynamics model
		self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
		self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
		self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
		self._action_enc = h.mlp_action(cfg.action_dim, cfg.mlp_dim, cfg.action_latent) # Action encoder
		self.apply(h.orthogonal_init)
		for m in [self._reward, self._Q1, self._Q2]:
			m[-1].weight.data.fill_(0)
			m[-1].bias.data.fill_(0)

		self.W1 = torch.nn.Linear(cfg.latent_dim+cfg.action_latent, cfg.latent_dim, bias=False) # Contrastive transformation matrix
		self.W1.weight.data.fill_(0)




	def track_q_grad(self, enable=True):
		"""Utility function. Enables/disables gradient tracking of Q-networks."""
		for m in [self._Q1, self._Q2]:
			h.set_requires_grad(m, enable)

	def h(self, obs):
		"""Encodes an observation into its latent representation (h)."""
		return self._encoder(obs)

	def next(self, z, a):
		"""Predicts next latent state (d) and single-step reward (R)."""
		x = torch.cat([z, a], dim=-1)
		return self._dynamics(x), self._reward(x)

	def pi(self, z, std=0):
		"""Samples an action from the learned policy (pi)."""
		mu = torch.tanh(self._pi(z))
		if std > 0:
			std = torch.ones_like(mu) * std
			return h.TruncatedNormal(mu, std).sample(clip=0.3)
		return mu

	def Q(self, z, a):
		"""Predict state-action value (Q)."""
		x = torch.cat([z, a], dim=-1)
		return self._Q1(x), self._Q2(x)

	def action_encode(self, action):
		"""encode action into latent representation."""
		return self._action_enc(action)

	def action_pred(self,z,next_z):
		"""Predict action with inverse dynamics model."""
		x = torch.cat([z, next_z], dim=-1)
		return self._inverse(x)


class TDMPC():
	"""Implementation of TD-MPC learning + inference."""
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.std = h.linear_schedule(cfg.std_schedule, 0)
		self.model = TOLD(cfg).cuda()
		self.model_target = deepcopy(self.model)
		self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
		self.cont_optim = torch.optim.Adam(self.cont_parameters(), lr=self.cfg.cont_lr) # optimizer for contrastive loss
		self.inverse_optim = torch.optim.Adam(self.inverse_parameters(), lr=self.cfg.inverse_lr) # optimizer for Inverse loss
		self.aug = h.RandomShiftsAug(cfg)
		self.ce_loss = torch.nn.CrossEntropyLoss()
		self.max_extrinsic_reward = 1
		self.max_intrinsic_reward = 0.000001
		self.model.eval()
		self.model_target.eval()

	def cont_parameters(self):
		"""parameters updated with contrastive loss"""
		yield from self.model.W1.parameters()
		yield from self.model._encoder.parameters() 
		yield from self.model._action_enc.parameters()
		

	def inverse_parameters(self):
		"""parameters updated with Inverse loss"""
		yield from self.model._encoder.parameters()
		yield from self.model._inverse.parameters()

	def state_dict(self):
		"""Retrieve state dict of TOLD model, including slow-moving target network."""
		return {'model': self.model.state_dict(),
				'model_target': self.model_target.state_dict()}

	def save(self, fp):
		"""Save state dict of TOLD model to filepath."""
		torch.save(self.state_dict(), fp)
	
	def load(self, fp):
		"""Load a saved state dict from filepath into current agent."""
		d = torch.load(fp)
		self.model.load_state_dict(d['model'])
		self.model_target.load_state_dict(d['model_target'])

	@torch.no_grad()
	def estimate_value(self, z, actions, horizon):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			Q = torch.min(*self.model.Q(z, actions[t]))
			z, reward = self.model.next(z, actions[t])
			# compute the scoring function as a discounted sum of Q values
			G += discount * Q 
			discount *= self.cfg.discount

		G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
		return G

	@torch.no_grad()
	def plan(self, obs, eval_mode=False, step=None, t0=True):
		"""
		Plan next action using TD-MPC inference.
		obs: raw input observation.
		eval_mode: uniform sampling and action noise is disabled during evaluation.
		step: current time step. determines e.g. planning horizon.
		t0: whether current step is the first step of an episode.
		"""
		# Seed steps
		if step < self.cfg.seed_steps and not eval_mode:
			return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

		# Sample policy trajectories
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
		num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
		if num_pi_trajs > 0:
			pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
			z = self.model.h(obs).repeat(num_pi_trajs, 1)
			for t in range(horizon):
				pi_actions[t] = self.model.pi(z, self.cfg.min_std)
				z, _ = self.model.next(z, pi_actions[t])

		# Initialize state and parameters
		z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
		mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
		std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
		if not t0 and hasattr(self, '_prev_mean'):
			mean[:-1] = self._prev_mean[1:]

		# Iterate CEM
		for i in range(self.cfg.iterations):
			actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
				torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=std.device), -1, 1)
			if num_pi_trajs > 0:
				actions = torch.cat([actions, pi_actions], dim=1)

			# Compute elite actions
			value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score /= score.sum(0)
			_mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
			_std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
			_std = _std.clamp_(self.std, 2)
			mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

		# Outputs
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		mean, std = actions[0], _std[0]
		a = mean
		if not eval_mode:
			a += std * torch.randn(self.cfg.action_dim, device=std.device)
		return a


	def ICM(self, obs, next_obs, action):
		"""Intrinsic Curiosity Module."""

		# compute intrinsic reward
		with torch.no_grad():
			# extract latent representations using Target encoder
			target_z = self.model_target.h(self.aug(obs))
			next_target_z = self.model_target.h(self.aug(next_obs))

			pred_next_z, _ = self.model.next(target_z, action)
			int_reward = torch.sqrt(torch.sum((next_target_z - pred_next_z)**2,axis=1))

		return int_reward



	def update_pi_inverse_dynamics(self, zs,  obs, next_obses, action):
		"""Update policy and inverse dynamics using a sequence of latent states."""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)


		pi_loss = 0
		inverse_loss = 0
		prev_z = self.model.h(self.aug(obs))
		for t,z in enumerate(zs):
			# compute pi loss
			a = self.model.pi(z, self.cfg.min_std)
			Q = torch.min(*self.model.Q(z, a))
			pi_loss += -Q.mean() * (self.cfg.rho ** t)

			# compute inverse loss
			if t < self.cfg.horizon:
				next_z = self.model.h(self.aug(next_obses[t]))
				action_pred = self.model.action_pred(prev_z, next_z)
				inverse_loss += torch.mean(h.mse(action_pred, action[t]), dim=1, keepdim=True)
				prev_z = next_z

		# Optimize policy network
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.pi_optim.step()
		self.model.track_q_grad(True)
		
		# Optimize inverse dynamics model and online enconder with inverse loss
		weighted_inverse_loss = (self.cfg.inverse_coef * inverse_loss.clamp(max=1e4)).mean()
		self.inverse_optim.zero_grad(set_to_none=True)
		weighted_inverse_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.inverse_parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.inverse_optim.step()


		return pi_loss.item(), inverse_loss.mean().item()

	@torch.no_grad()
	def _td_target(self, next_obs, ext_reward, int_reward, step):
		"""Compute the TD-target from extrinsic and intrinsic reward and the observation at the following time step."""

		max_reward = np.max(ext_reward.cpu().numpy())
		if max_reward > self.max_extrinsic_reward:
			self.max_extrinsic_reward = max_reward

		max = torch.max(int_reward)
		if max > self.max_intrinsic_reward:
			self.max_intrinsic_reward = max

		# Normalize int_reward
		int_reward = int_reward/self.max_intrinsic_reward*self.max_extrinsic_reward
		int_reward = int_reward.reshape(-1,1)

		#total reward is ext_reward + decaying int_reward
		reward = ext_reward + self.cfg.intrinsic_weight*np.exp(-self.cfg.intrinsic_decay*step)*int_reward  

		next_z = self.model.h(next_obs)
		td_target = reward + self.cfg.discount * \
			torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
		return td_target

	def compute_logits(self, anchor, positive, action_latent=None):
		"""compute logits for temporal contrastive loss  """
	
		anchor_action_cat = torch.cat([action_latent, anchor], dim=1)
		pred = self.model.W1(anchor_action_cat)
		logits = torch.matmul(pred, positive.T)
		logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # normalize
		return logits

		
	def update(self, replay_buffer, step):
		"""Main update function. Corresponds to one iteration of the TOLD model learning."""
		obs, next_obses, action, ext_reward, idxs, weights = replay_buffer.sample()
		self.optim.zero_grad(set_to_none=True)
		self.std = h.linear_schedule(self.cfg.std_schedule, step)
		self.model.train()

		# Representation
		z = self.model.h(self.aug(obs))
		zs = [z.detach()]
		prev_obs = obs
		consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
		for t in range(self.cfg.horizon):

			# Predictions
			Q1, Q2 = self.model.Q(z, action[t])
			z, ext_reward_pred = self.model.next(z, action[t])
			# compute intrinsic reward using ICM
			int_reward = self.ICM(prev_obs, next_obses[t], action[t])
			with torch.no_grad():
				next_obs = self.aug(next_obses[t])
				next_z = self.model_target.h(next_obs)
				td_target = self._td_target(next_obs, ext_reward[t], int_reward, step)

			zs.append(z.detach())
			prev_obs = next_obses[t]

			
			# Losses
			rho = (self.cfg.rho ** t)
			consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
			reward_loss += rho * h.mse(ext_reward_pred, ext_reward[t])
			value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
			priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))


		# Optimize TOLD model
		total_loss = self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
					 self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
					 self.cfg.value_coef * value_loss.clamp(max=1e4)
		weighted_loss = (total_loss.squeeze(1) * weights).mean()
		weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
		weighted_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.optim.step()
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())



		# Compute temporal contrastive loss
		anchor = self.model.h(self.aug(obs))
		action_latent = self.model.action_encode(action[0])
		with torch.no_grad():
			positive = self.model_target.h(self.aug(next_obses[0]))
			
			
		logits = self.compute_logits(anchor, positive, action_latent)
		labels = torch.arange(logits.shape[0],dtype=torch.long, device=self.device)
		temporal_loss = self.ce_loss(logits, labels)
		
		total_temporal_loss = temporal_loss * self.cfg.temporal_coef

		# Optimize online enconder and action encoder with contrastive loss
		self.cont_optim.zero_grad(set_to_none=True)
		total_temporal_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.cont_parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
		self.cont_optim.step()
		
		# Update policy + inverse dynamics model
		pi_loss, inverse_loss = self.update_pi_inverse_dynamics(zs, obs, next_obses, action) 

		# update Target networks
		if step % self.cfg.update_Q_freq == 0:
			h.ema(self.model._Q1, self.model_target._Q1, self.cfg.tau)
			h.ema(self.model._Q2, self.model_target._Q2, self.cfg.tau)
			
		if step % self.cfg.update_ENC_freq == 0:
			h.ema(self.model._encoder, self.model_target._encoder,self.cfg.tau )
			

		self.model.eval()
		return {'consistency_loss': float(consistency_loss.mean().item()),
				'reward_loss': float(reward_loss.mean().item()),
				'value_loss': float(value_loss.mean().item()),
				'inverse_loss': inverse_loss,
				'temporal_loss': float(temporal_loss.item()),
				'pi_loss': pi_loss,
				'total_loss': float(total_loss.mean().item()),
				'weighted_loss': float(weighted_loss.mean().item()),
				'grad_norm': float(grad_norm)}
