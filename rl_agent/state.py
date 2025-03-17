from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime

from browser_use.browser.views import BrowserState as BaseBrowserState
from browser_use.agent.views import AgentState
from browser_use.browser.context import BrowserContext
from .reward import StateMetrics

@dataclass
class RLState:
    """
    Reinforcement Learning state that wraps and extends the browser-use state.
    Adds RL-specific tracking while maintaining compatibility with browser-use.
    """
    # Core browser state (from browser-use)
    browser_state: BaseBrowserState
    agent_state: AgentState
    
    # RL-specific state
    episode_reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    
    def to_metrics(self) -> StateMetrics:
        """Convert current state to metrics for reward calculation."""
        return StateMetrics(
            step_count=self.agent_state.n_steps,
            tokens_used=self.agent_state.history.total_input_tokens(),
            time_elapsed=self.agent_state.history.total_duration_seconds(),
            last_action_type=self.agent_state.last_action_type if hasattr(self.agent_state, 'last_action_type') else "",
            errors_encountered=len(self.agent_state.history.errors()),
            llm_confidence=self.agent_state.last_llm_confidence if hasattr(self.agent_state, 'last_llm_confidence') else 1.0,
            navigation_history=[h.state.url for h in self.agent_state.history.history]
        )

class RLStateTracker:
    """
    Manages the RL-enhanced browser state, integrating with browser-use's state management.
    """
    def __init__(self, browser_context: BrowserContext, task: str):
        self.browser_context = browser_context
        self.task = task
        self.rl_state = None
        
    async def initialize_state(self) -> RLState:
        """Initialize the RL state with current browser and agent states."""
        browser_state = await self.browser_context.get_state()
        agent_state = AgentState()  # Initialize with default agent state
        
        self.rl_state = RLState(
            browser_state=browser_state,
            agent_state=agent_state
        )
        return self.rl_state
    
    async def update_state(
        self,
        action_result: Dict[str, Any],
        reward: float,
        done: bool = False
    ) -> RLState:
        """
        Update state based on action results and reward.
        Integrates with browser-use's state management.
        """
        if self.rl_state is None:
            self.rl_state = await self.initialize_state()
            
        # Update browser state
        self.rl_state.browser_state = await self.browser_context.get_state()
        
        # Update RL-specific state
        self.rl_state.episode_reward += reward
        self.rl_state.cumulative_reward += reward
        self.rl_state.done = done
        
        return self.rl_state
    
    async def get_state(self) -> RLState:
        """Get current RL state, initializing if necessary."""
        if self.rl_state is None:
            return await self.initialize_state()
        return self.rl_state
    
    def reset(self) -> None:
        """Reset RL state for a new episode."""
        self.rl_state = None  # Will be reinitialized on next get_state/update_state call
