from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np

@dataclass
class RewardConfig:
    """Configuration for reward function weights and parameters."""
    # Task completion weights
    task_alignment_weight: float = 10.0
    task_completion_bonus: float = 100.0
    
    # Efficiency weights
    step_penalty: float = -0.1
    token_penalty: float = -0.001
    time_penalty: float = -0.05
    
    # Navigation weights
    backtrack_penalty: float = -1.0
    error_penalty: float = -5.0
    
    # LLM confidence weights
    llm_confidence_weight: float = 2.0
    
    # Thresholds
    min_task_alignment: float = 0.3
    max_allowed_steps: int = 20
    max_allowed_tokens: int = 1000
    max_allowed_time: float = 300  # seconds

@dataclass
class StateMetrics:
    """Metrics extracted from the current browser state."""
    step_count: int
    tokens_used: int
    time_elapsed: float
    last_action_type: str
    errors_encountered: int
    llm_confidence: float
    navigation_history: List[str]

class RewardFunction:
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self._last_task_alignment: float = 0.0
    
    def calculate_task_alignment_reward(self, current_alignment: float) -> float:
        """Calculate reward based on task alignment progress."""
        alignment_delta = current_alignment - self._last_task_alignment
        self._last_task_alignment = current_alignment
        
        # Reward improvement in task alignment
        if current_alignment >= 1.0:  # Task completed successfully
            return self.config.task_completion_bonus
        elif alignment_delta > 0:
            return alignment_delta * self.config.task_alignment_weight
        elif current_alignment < self.config.min_task_alignment:
            return self.config.error_penalty  # Penalize significant misalignment
        return 0.0

    def calculate_efficiency_reward(self, metrics: StateMetrics) -> float:
        """Calculate reward based on resource efficiency."""
        reward = 0.0
        
        # Step efficiency
        reward += metrics.step_count * self.config.step_penalty
        
        # Token efficiency
        token_usage_ratio = metrics.tokens_used / self.config.max_allowed_tokens
        reward += token_usage_ratio * self.config.token_penalty
        
        # Time efficiency
        time_ratio = metrics.time_elapsed / self.config.max_allowed_time
        reward += time_ratio * self.config.time_penalty
        
        return reward

    def calculate_navigation_reward(self, metrics: StateMetrics) -> float:
        """Calculate reward based on navigation efficiency."""
        reward = 0.0
        
        # Penalize backtracking
        if metrics.last_action_type == "go_back":
            reward += self.config.backtrack_penalty
            
        # Penalize errors
        reward += metrics.errors_encountered * self.config.error_penalty
        
        return reward

    def calculate_llm_confidence_reward(self, metrics: StateMetrics) -> float:
        """Calculate reward based on LLM confidence in actions."""
        return metrics.llm_confidence * self.config.llm_confidence_weight

    def calculate_reward(
        self,
        task_alignment: float,
        metrics: StateMetrics,
        done: bool
    ) -> Dict[str, float]:
        """
        Calculate the total reward and its components.
        
        Args:
            task_alignment: Float between 0 and 1 indicating task completion alignment
            metrics: Current state metrics
            done: Whether the episode is complete
            
        Returns:
            Dictionary containing total reward and individual components
        """
        # Calculate individual reward components
        task_reward = self.calculate_task_alignment_reward(task_alignment)
        efficiency_reward = self.calculate_efficiency_reward(metrics)
        navigation_reward = self.calculate_navigation_reward(metrics)
        confidence_reward = self.calculate_llm_confidence_reward(metrics)
        
        # Sum up total reward
        total_reward = sum([
            task_reward,
            efficiency_reward,
            navigation_reward,
            confidence_reward
        ])
        
        # Early termination penalties
        if done and task_alignment < 1.0:
            if metrics.step_count >= self.config.max_allowed_steps:
                total_reward += self.config.error_penalty  # Penalize step limit exceeded
            if metrics.tokens_used >= self.config.max_allowed_tokens:
                total_reward += self.config.error_penalty  # Penalize token limit exceeded
            if metrics.time_elapsed >= self.config.max_allowed_time:
                total_reward += self.config.error_penalty  # Penalize time limit exceeded
        
        return {
            "total": total_reward,
            "task_alignment": task_reward,
            "efficiency": efficiency_reward,
            "navigation": navigation_reward,
            "confidence": confidence_reward
        }

    def reset(self):
        """Reset internal state of the reward function."""
        self._last_task_alignment = 0.0
