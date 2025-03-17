import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, cast

from .env import BrowserEnv
from .model import PPOAgent
from .action_mapping import ActionType

def main():
    """Example of training a PPO agent in the browser environment."""
    # Create environment and agent
    env = BrowserEnv(
        max_steps=50,
        reward_scale=1.0,
        time_penalty=-0.01,
        success_reward=1.0,
        failure_penalty=-0.5
    )
    
    # Create agent
    state_dim = sum(
        np.prod(space.shape)
        for space in env.observation_space.values()
        if hasattr(space, 'shape')
    )
    agent = PPOAgent(
        state_dim=state_dim,
        hidden_dim=256,
        lr=3e-4
    )
    
    # Training loop
    num_episodes = 10
    max_steps = 50
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()[0]  # Get only state, ignore info
        episode_reward = 0
        
        for step in range(max_steps):
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Convert state to tensor
            state_dict = cast(Dict[str, Any], state)
            state_tensors = []
            
            # Add numeric state components
            for key in ['step_count', 'tokens_used', 'time_elapsed', 'errors_encountered', 'task_progress']:
                value = state_dict[key]
                if isinstance(value, np.ndarray):
                    state_tensors.append(torch.FloatTensor(value))
                else:
                    state_tensors.append(torch.FloatTensor([value]))
            
            # Concatenate all state components
            state_tensor = torch.cat(state_tensors)
            
            # Select action
            action, log_prob, value = agent.select_action(state_tensor, valid_actions)
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Print step information
            print(f"\nStep {step + 1}:")
            print(f"Action: {action['type'].name}")
            print(f"Reward: {reward:.2f}")
            print(f"State: {next_state}")
            
            if terminated or truncated:
                break
            
            state = next_state
        
        print(f"\nEpisode {episode + 1} finished with reward: {episode_reward:.2f}")

if __name__ == "__main__":
    main() 