from typing import Optional, Dict, Any, List, Tuple
import asyncio
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import wandb
from tqdm import tqdm
import os
import base64
from datetime import datetime
from PIL import Image
import io

from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from .environment import BrowserEnv, RLState
from .reward import RewardConfig
from .experience import ExperienceTracker
from .model import BrowserPolicy, PPOAgent
from .action import RLActionSpace
from .state import StateMetrics

from dotenv import load_dotenv
load_dotenv()

def state_to_tensor(state: RLState, device: str) -> torch.Tensor:
    """Convert RLState to tensor representation."""
    metrics = state.to_metrics()
    return torch.tensor([
        metrics.step_count,
        metrics.tokens_used,
        metrics.time_elapsed,
        metrics.errors_encountered,
        metrics.llm_confidence,
        len(metrics.navigation_history),
        float(state.done),
        state.episode_reward,
        state.cumulative_reward
    ], device=device, dtype=torch.float32)

def save_screenshot(screenshot_base64: str, episode: int, step: int, save_dir: str) -> str:
    """Save a base64 screenshot as a PNG file."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Decode base64 string to image
    img_data = base64.b64decode(screenshot_base64)
    img = Image.open(io.BytesIO(img_data))
    
    # Save image
    filename = f"episode_{episode:03d}_step_{step:03d}.png"
    filepath = os.path.join(save_dir, filename)
    img.save(filepath)
    return filepath

async def train(
    env: BrowserEnv,
    agent: PPOAgent,
    num_episodes: int,
    steps_per_episode: int,
    update_epochs: int = 10,
    batch_size: int = 64,
    wandb_log: bool = True
) -> None:
    """Train the agent using PPO."""
    print(f"\nStarting training with task: {env.task}\n")
    
    # Create directory for screenshots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_dir = os.path.join("rl_agent", "experiments", f"training_screenshots_{timestamp}")
    os.makedirs(screenshot_dir, exist_ok=True)
    print(f"Saving screenshots to: {screenshot_dir}")
    
    for episode in range(num_episodes):
        state, _ = await env.reset()
        episode_reward = 0
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        
        # Create episode directory
        episode_dir = os.path.join(screenshot_dir, f"episode_{episode:03d}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Run episode
        for step in range(steps_per_episode):
            # Get valid actions
            valid_actions = await env.get_valid_actions()
            if not valid_actions:
                print(f"No valid actions available at step {step}")
                break
                
            # Convert state to tensor
            state_tensor = torch.FloatTensor([
                state.browser_state.pixels_above,
                state.browser_state.pixels_below,
                state.episode_reward,
                state.cumulative_reward,
                float(state.done),
                len(state.browser_state.tabs),
                len(state.agent_state.history.history) if state.agent_state.history else 0,
                state.agent_state.n_steps,
                float(state.agent_state.consecutive_failures)
            ])
            
            # Get action probabilities and value
            with torch.no_grad():
                action_probs, value = agent.policy(state_tensor, valid_actions)
                
            # Sample action
            action_dist = Categorical(action_probs)
            action_idx = action_dist.sample()
            log_prob = action_dist.log_prob(action_idx)
            
            # Convert action index to action dict
            action = valid_actions[action_idx]
            
            # Print action being taken
            print(f"\nStep {step}:")
            print(f"Current URL: {state.browser_state.url}")
            print(f"{len(valid_actions)}available actions")
            print(f"Taking action: {action['type']}' with params: {action['params']}")
            
            # Execute action
            next_state, reward, terminated, truncated, info = await env.step(action)
            
            # Save screenshot if available
            if next_state.browser_state.screenshot:
                screenshot_path = save_screenshot(
                    next_state.browser_state.screenshot,
                    episode,
                    step,
                    episode_dir
                )
                if wandb_log:
                    wandb.log({
                        "screenshot": wandb.Image(
                            screenshot_path,
                            caption=f"Episode {episode}, Step {step}\nAction: {action['type']}\nReward: {reward:.2f}"
                        )
                    })
            
            # Print reward components and vision analysis
            print(f"Reward components: {info['reward_components']}")
            print(f"Task alignment (vision score): {info['task_alignment']:.2f}")
            
            # Store experience
            states.append(state_tensor)
            actions.append(action_idx)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(terminated or truncated)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        # Convert lists to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)
        
        # Update policy
        metrics = agent.update(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
            epochs=update_epochs,
            batch_size=batch_size
        )
        
        print(f"\nEpisode {episode} finished:")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Steps taken: {len(states)}")
        print(f"Training metrics: {metrics}\n")
        print(f"Screenshots saved to: {episode_dir}\n")
        
        if wandb_log:
            wandb.log({
                "episode": episode,
                "reward": episode_reward,
                "steps": len(states),
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"]
            })

async def evaluate(
    env: BrowserEnv,
    policy: BrowserPolicy,
    n_episodes: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """Evaluate a trained policy."""
    total_reward = 0.0
    success_count = 0
    
    for episode in range(n_episodes):
        reset_result = await env.reset()
        state = reset_result[0]
        episode_reward = 0.0
        done = False
        
        while not done:
            # Get valid actions
            valid_actions = await env.get_valid_actions()
            
            # Convert state to tensor
            state_tensor = state_to_tensor(state, device)
            
            # Select action
            with torch.no_grad():
                action_probs, _ = policy(state_tensor, valid_actions)
                action_dist = Categorical(action_probs)
                action_idx = action_dist.sample()
            
            # Convert action index to action dictionary
            action = valid_actions[action_idx.item()]
            
            # Take action
            step_result = await env.step(action)
            state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
        if info.get("task_success", False):
            success_count += 1
            
        print(f"Episode {episode} finished with reward {episode_reward}")
    
    return {
        "mean_reward": total_reward / n_episodes,
        "success_rate": success_count / n_episodes
    }

async def main():
    """Main training loop."""
    # Initialize wandb with new directory
    os.environ["WANDB_DIR"] = os.path.join(os.path.dirname(__file__), "wandb")
    wandb.init(
        project="browser-rl",
        dir=os.environ["WANDB_DIR"]
    )
    
    # Create environment
    env = BrowserEnv(
        task="Navigate to google.com and search for 'reinforcement learning'",
        browser_config=BrowserConfig(headless=False),  # Set to False to see the browser
        context_config=BrowserContextConfig()
    )
    
    # Create policy and agent
    state_dim = 9  # Number of features in our state tensor
    policy = BrowserPolicy(
        state_dim=state_dim,
        hidden_dim=256,
        action_space=env.action_space,
        llm=None  # Disabled for now
    )
    
    # Create checkpoints directory
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    agent = PPOAgent(
        policy=policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Train agent
    await train(
        env=env,
        agent=agent,
        num_episodes=100,
        steps_per_episode=50,
        update_epochs=10,
        batch_size=64,
        wandb_log=True
    )
    
    # Save final model
    torch.save(
        agent.policy.state_dict(),
        os.path.join(checkpoints_dir, "final_policy.pt")
    )
    
    await env.close()
    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main()) 