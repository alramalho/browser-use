from typing import Dict, Any, Tuple, Optional, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from openai import OpenAI
import base64
import os

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.agent.views import ActionResult

from .state import RLStateTracker, RLState
from .action import RLActionSpace
from .reward import RewardFunction, RewardConfig, StateMetrics

class BrowserEnv(gym.Env):
    """
    Gymnasium environment for browser interaction using RL.
    Integrates browser-use components with RL abstractions.
    """
    def __init__(
        self,
        task: str,
        reward_config: Optional[RewardConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
        context_config: Optional[BrowserContextConfig] = None
    ):
        super().__init__()
        self.task = task
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize browser components
        self.browser_config = browser_config or BrowserConfig()
        self.context_config = context_config or BrowserContextConfig()
        self.browser = None
        self.browser_context = None
        self.action_space = None
        
        # Initialize RL components
        self.reward_function = RewardFunction(config=reward_config)
        self.state_tracker = None
        
        # Define observation and action spaces
        # Note: These are placeholder spaces - they will be properly initialized in reset()
        self.observation_space = spaces.Dict({
            "url": spaces.Text(max_length=2048),
            "page_title": spaces.Text(max_length=256),
            "step_count": spaces.Discrete(100),
            "tokens_used": spaces.Discrete(10000),
            "time_elapsed": spaces.Box(low=0, high=float("inf"), shape=(1,), dtype=np.float32),
            "last_action_success": spaces.Discrete(2),
            "errors_encountered": spaces.Discrete(100),
            "task_progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

    async def _setup(self):
        """Initialize browser and context if not already done."""
        if self.browser is None:
            self.browser = Browser(config=self.browser_config)
            
        if self.browser_context is None:
            self.browser_context = await self.browser.new_context(self.context_config)
            # Create initial page and navigate to about:blank
            page = await self.browser_context.get_current_page()
            await page.goto("https://www.google.com")
            
        if self.action_space is None:
            self.action_space = RLActionSpace(self.browser_context)
            
        if self.state_tracker is None:
            self.state_tracker = RLStateTracker(
                browser_context=self.browser_context,
                task=self.task
            )

    async def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[RLState, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        Returns:
            state: Initial state
            info: Additional information
        """
        if seed is not None:
            super().reset(seed=seed)
        
        # Setup/reset components
        await self._setup()
        
        if self.reward_function:
            self.reward_function.reset()
            
        if self.state_tracker:
            self.state_tracker.reset()
            initial_state = await self.state_tracker.get_state()
        else:
            raise RuntimeError("State tracker not initialized")
        
        return initial_state, {"task": self.task}

    async def estimate_task_alignment(self, metrics: StateMetrics) -> float:
        """
        Estimate task alignment using OpenAI's Vision model to analyze the current screenshot.
        Returns a value between 0 and 1 indicating progress towards task completion.
        """
        if not metrics.navigation_history or not self.state_tracker or not self.browser_context:
            return 0.0
            
        # Get the current screenshot
        state = await self.browser_context.get_state()
        if not state.screenshot:
            return 0.0
            
        try:
            # Prepare the message for the vision model
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Given the task '{self.task}', analyze this screenshot and estimate the progress towards completing the task. Return ONLY a number between 0 and 1, where 0 means no progress and 1 means task completed. For example: 0.5"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{state.screenshot}"
                                }
                            }
                        ]
                }],
            )
            print("Model visual analysis: ")
            print(response.choices[0].message.content)
            
            # Extract the progress value from the response
            progress_text = response.choices[0].message.content.strip()
            try:
                progress = float(progress_text)
                # Ensure the value is between 0 and 1
                progress = max(0.0, min(1.0, progress))
                return progress
            except ValueError:
                return 0.0
                
        except Exception as e:
            print(f"Error estimating task alignment: {e}")
            return 0.0

    async def step(
        self,
        action: Dict[str, Any]
    ) -> Tuple[RLState, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        Args:
            action: Dictionary with keys 'type' and 'params'
        Returns:
            state: New state after action
            reward: Reward for the action
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        if not self.action_space:
            raise RuntimeError("Action space not initialized")
            
        # Execute action
        action_result = await self.action_space.execute_action(
            action_type=action["type"],
            action_params=action["params"]
        )
        
        if not self.state_tracker:
            raise RuntimeError("State tracker not initialized")
            
        # Update state
        new_state = await self.state_tracker.update_state(
            action_result=action_result.action_result,
            reward=action_result.predicted_reward,
            done=self.action_space.is_terminal_action(action["type"])
        )
        
        # Calculate reward
        metrics = new_state.to_metrics()
        task_alignment = await self.estimate_task_alignment(metrics)
        reward_info = self.reward_function.calculate_reward(
            task_alignment=task_alignment,
            metrics=metrics,
            done=new_state.done
        )
        
        # Check if episode is done
        terminated = new_state.done or task_alignment >= 1.0
        truncated = (
            metrics.step_count >= self.reward_function.config.max_allowed_steps or
            metrics.tokens_used >= self.reward_function.config.max_allowed_tokens or
            metrics.time_elapsed >= self.reward_function.config.max_allowed_time
        )
        
        info = {
            "action_result": action_result,
            "reward_components": reward_info,
            "metrics": metrics,
            "task_alignment": task_alignment
        }
        
        return new_state, reward_info["total"], terminated, truncated, info

    async def close(self):
        """Clean up resources."""
        if self.browser_context:
            await self.browser_context.close()
        if self.browser:
            await self.browser.close()
            
    async def render(self):
        """
        Render the current state.
        For browser environments, this could show the current page screenshot.
        """
        if self.browser_context:
            state = await self.browser_context.get_state()
            return state.screenshot  # Returns base64 encoded screenshot
        return None

    async def get_valid_actions(self) -> List[Dict[str, Any]]:
        """Get the currently valid actions."""
        if not self.action_space:
            return []
        return await self.action_space.get_valid_actions() 