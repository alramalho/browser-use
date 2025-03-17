from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

@dataclass
class Episode:
    """Stores complete information about a training episode."""
    states: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    total_reward: float
    length: int
    success: bool
    timestamp: str

class ExperienceTracker:
    """Tracks and analyzes training episodes."""
    def __init__(
        self,
        save_dir: str = "experiences",
        max_episodes: int = 1000,
        smoothing_window: int = 100
    ):
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.smoothing_window = smoothing_window
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize storage
        self.episodes: List[Episode] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rate: List[float] = []
        
    def add_episode(
        self,
        states: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        rewards: List[float],
        values: List[float],
        log_probs: List[float],
        success: bool
    ) -> None:
        """Add a completed episode to the tracker."""
        episode = Episode(
            states=states,
            actions=actions,
            rewards=rewards,
            values=values,
            log_probs=log_probs,
            total_reward=sum(rewards),
            length=len(states),
            success=success,
            timestamp=datetime.now().isoformat()
        )
        
        # Add to storage
        self.episodes.append(episode)
        self.episode_rewards.append(episode.total_reward)
        self.episode_lengths.append(episode.length)
        
        # Update success rate
        if len(self.success_rate) == 0:
            self.success_rate.append(float(success))
        else:
            # Moving average of success rate
            self.success_rate.append(
                0.95 * self.success_rate[-1] + 0.05 * float(success)
            )
        
        # Trim if exceeding max episodes
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
            self.success_rate.pop(0)
            
        # Save episode
        self._save_episode(episode)
    
    def plot_metrics(self, show: bool = True, save: bool = True) -> None:
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, label='Raw')
        ax.plot(
            self._smooth(self.episode_rewards),
            label=f'Smoothed (window={self.smoothing_window})'
        )
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        
        # Plot episode lengths
        ax = axes[0, 1]
        ax.plot(self.episode_lengths, alpha=0.3, label='Raw')
        ax.plot(
            self._smooth(self.episode_lengths),
            label=f'Smoothed (window={self.smoothing_window})'
        )
        ax.set_title('Episode Lengths')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.legend()
        
        # Plot success rate
        ax = axes[1, 0]
        ax.plot(self.success_rate)
        ax.set_title('Success Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rate')
        
        # Plot value predictions vs actual returns
        ax = axes[1, 1]
        if len(self.episodes) > 0:
            latest_episode = self.episodes[-1]
            cumulative_rewards = np.cumsum(latest_episode.rewards[::-1])[::-1]
            ax.plot(latest_episode.values, label='Predicted Values')
            ax.plot(cumulative_rewards, label='Actual Returns')
            ax.set_title('Value Predictions (Latest Episode)')
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_dir}/metrics.png")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_replay(self, episode_idx: int = -1) -> str:
        """
        Save episode replay as a JSON file.
        Returns the path to the saved file.
        """
        episode = self.episodes[episode_idx]
        
        replay_data = {
            'timestamp': episode.timestamp,
            'total_reward': episode.total_reward,
            'length': episode.length,
            'success': episode.success,
            'trajectory': [
                {
                    'step': i,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'value': value,
                    'log_prob': log_prob
                }
                for i, (state, action, reward, value, log_prob) in enumerate(zip(
                    episode.states,
                    episode.actions,
                    episode.rewards,
                    episode.values,
                    episode.log_probs
                ))
            ]
        }
        
        # Save to file
        filename = f"{self.save_dir}/replay_{episode.timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(replay_data, f, indent=2)
            
        return filename
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current training statistics."""
        if len(self.episodes) == 0:
            return {}
            
        return {
            'mean_reward': np.mean(self.episode_rewards[-self.smoothing_window:]),
            'mean_length': np.mean(self.episode_lengths[-self.smoothing_window:]),
            'success_rate': self.success_rate[-1],
            'best_reward': max(self.episode_rewards),
            'worst_reward': min(self.episode_rewards),
            'total_episodes': len(self.episodes)
        }
    
    def _smooth(self, values: List[float]) -> np.ndarray:
        """Apply smoothing to a sequence of values."""
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        return np.convolve(values, kernel, mode='valid')
    
    def _save_episode(self, episode: Episode) -> None:
        """Save episode data to disk."""
        filename = f"{self.save_dir}/episode_{episode.timestamp}.json"
        
        # Convert episode data to JSON-serializable format
        episode_data = {
            'timestamp': episode.timestamp,
            'total_reward': episode.total_reward,
            'length': episode.length,
            'success': episode.success,
            'states': episode.states,
            'actions': [
                {
                    'type': action['type'].name,
                    'params': action['params']
                }
                for action in episode.actions
            ],
            'rewards': episode.rewards,
            'values': [v.item() for v in episode.values],
            'log_probs': [lp.item() for lp in episode.log_probs]
        }
        
        with open(filename, 'w') as f:
            json.dump(episode_data, f, indent=2) 