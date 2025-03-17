from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import json
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from .action import RLActionSpace

@dataclass
class Experience:
    """Single step experience for PPO."""
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor

class BrowserPolicy(nn.Module):
    """
    Policy network for browser actions.
    Combines neural network predictions with LLM guidance.
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        action_space: Optional[RLActionSpace] = None,
        llm: Optional[BaseChatModel] = None,
        llm_weight: float = 0.3
    ):
        super().__init__()
        self.llm = llm
        self.llm_weight = llm_weight
        self.action_space = action_space
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action head (outputs action probabilities)
        action_dim = action_space.action_space_size() if action_space else 20  # Default to 20 actions if no action space provided
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Value head (outputs state value)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        state: torch.Tensor,
        valid_actions: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining NN and LLM predictions.
        Returns action probabilities and state value.
        """
        x = self.state_encoder(state)
        
        # Get neural network action probabilities
        nn_logits = self.action_head(x)
        
        # Apply action masking for invalid actions
        action_mask = torch.zeros_like(nn_logits)
        for i in range(min(len(valid_actions), nn_logits.size(0))):
            action_mask[i] = 1
        nn_logits = nn_logits.masked_fill(action_mask == 0, float('-inf'))
        
        # Get LLM action probabilities if available
        if self.llm is not None:
            llm_probs = self.get_llm_probs(state, valid_actions)
            if llm_probs is not None:
                # Combine NN and LLM probabilities
                nn_probs = F.softmax(nn_logits, dim=-1)
                combined_probs = (
                    (1 - self.llm_weight) * nn_probs +
                    self.llm_weight * llm_probs
                )
            else:
                combined_probs = F.softmax(nn_logits, dim=-1)
        else:
            combined_probs = F.softmax(nn_logits, dim=-1)
            
        value = self.value_head(x)
        
        return combined_probs, value
    
    def get_llm_probs(
        self,
        state: torch.Tensor,
        valid_actions: List[Dict[str, Any]]
    ) -> Optional[torch.Tensor]:
        """
        Get action probabilities from LLM.
        Converts state and actions to text, queries LLM, and returns probabilities.
        """
        if self.llm is None:
            return None
            
        # Initialize probabilities
        probs = torch.zeros(len(valid_actions))
        
        # Create prompt for LLM
        prompt = self._create_llm_prompt(state, valid_actions)
        
        try:
            # Get LLM response
            response = self.llm.predict(prompt)
            
            # Parse LLM response to get action probabilities
            action_probs = self._parse_llm_response(response, valid_actions)
            
            # Convert to tensor
            for i, (action, prob) in enumerate(action_probs.items()):
                probs[i] = prob
                
            # Normalize probabilities
            probs = F.softmax(probs, dim=-1)
            
            return probs
            
        except Exception as e:
            # Fallback to uniform distribution over valid actions
            probs.fill_(1.0 / len(valid_actions))
            return probs
    
    def _create_llm_prompt(
        self,
        state: torch.Tensor,
        valid_actions: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for LLM to predict action probabilities."""
        # Convert state tensor to meaningful text
        state_desc = self._state_to_text(state)
        
        # Format actions
        action_desc = "\n".join(
            f"- {action['type']}: {action['params']}"
            for action in valid_actions
        )
        
        return f"""Given the current browser state:
{state_desc}

And these available actions:
{action_desc}

Assign probabilities to each action based on how likely it is to help achieve the task.
Return the probabilities in JSON format:
{{
    "action_type1": probability1,
    "action_type2": probability2,
    ...
}}

The probabilities should sum to 1."""
    
    def _state_to_text(self, state: torch.Tensor) -> str:
        """Convert state tensor to human-readable text."""
        # TODO: Implement state tensor to text conversion
        return "Current browser state description"
    
    def _parse_llm_response(
        self,
        response: str,
        valid_actions: List[Dict[str, Any]]
    ) -> Dict[Dict[str, Any], float]:
        """Parse LLM response into action probabilities."""
        try:
            # Parse JSON response
            probs_dict = json.loads(response)
            
            # Map probabilities to actions
            action_probs = {}
            for action in valid_actions:
                action_type = action["type"]
                if action_type in probs_dict:
                    action_probs[action] = probs_dict[action_type]
                else:
                    action_probs[action] = 0.0
                    
            return action_probs
            
        except Exception as e:
            # Fallback to uniform distribution
            return {
                action: 1.0 / len(valid_actions)
                for action in valid_actions
            }

class PPOAgent:
    """
    PPO agent for browser interaction.
    """
    def __init__(
        self,
        policy: BrowserPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 1.0,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Experience buffer
        self.experiences: List[Experience] = []
        
    def store_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """Store a single step experience."""
        self.experiences.append(Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        ))
        
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """Update policy using PPO."""
        # Convert dones to float tensor
        dones = dones.float()
        
        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute returns with GAE
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for _ in range(epochs):
            # Generate random permutation for minibatches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                # Get minibatch
                end_idx = start_idx + batch_size
                mb_indices = indices[start_idx:end_idx]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Forward pass with dummy valid actions (since we're just updating)
                dummy_valid_actions = [{"type": "dummy", "params": {}}] * mb_states.size(0)
                action_probs, values_pred = self.policy(mb_states, dummy_valid_actions)
                
                # Get new action probabilities and values
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio and surrogate losses
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values_pred.squeeze(), mb_returns)
                
                # Compute total loss and update
                loss = (
                    policy_loss +
                    self.value_coef * value_loss -
                    self.entropy_coef * entropy
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
                
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates
        }
