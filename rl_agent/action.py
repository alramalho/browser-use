from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from browser_use.controller.views import (
    SearchGoogleAction,
    GoToUrlAction,
    ClickElementAction,
    InputTextAction,
    DoneAction,
    SwitchTabAction,
    OpenTabAction,
    ScrollAction,
    SendKeysAction,
    ExtractPageContentAction,
    NoParamsAction
)
from browser_use.controller.registry.views import ActionModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller
from browser_use.dom.views import DOMElementNode

@dataclass
class RLActionResult:
    """
    Wrapper around browser-use's ActionResult with RL-specific information.
    """
    action_result: ActionResult
    action_type: str
    action_params: Dict[str, Any]
    predicted_success_prob: float
    predicted_reward: float

class RLActionSpace:
    """
    Manages the RL action space by wrapping browser-use's actions.
    Provides methods for action selection, execution, and prediction.
    """
    def __init__(self, browser_context: BrowserContext):
        self.browser_context = browser_context
        self.controller = Controller()
        # Get the ActionModel class that knows about all registered actions
        self.ActionModel = self.controller.registry.create_action_model()
        
    async def execute_action(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        **kwargs
    ) -> RLActionResult:
        """
        Execute an action using browser-use's controller.
        Returns the result wrapped with RL-specific information.
        """
        # Create the action model using browser-use's native format
        action_model = self.ActionModel(**{action_type: action_params})
        
        # Execute the action using browser-use's controller
        result = await self.controller.act(
            action=action_model,
            browser_context=self.browser_context,
            **kwargs
        )
        
        # For now, use simple heuristics for predictions
        # These will be replaced by learned predictions
        predicted_success = 0.8 if not result.error else 0.2
        predicted_reward = 1.0 if not result.error else -0.5
        
        return RLActionResult(
            action_result=result,
            action_type=action_type,
            action_params=action_params,
            predicted_success_prob=predicted_success,
            predicted_reward=predicted_reward
        )
    
    def _get_action_for_element(self, element: DOMElementNode, idx: int) -> List[Dict[str, Any]]:
        """Get all valid actions for a given DOM element."""
        actions = []
        
        
        # Handle input fields
        if element.tag_name.lower() == 'input':
            # For search input fields
            if (element.attributes.get('type') == 'text' or 
                element.attributes.get('type') == 'search' or 
                element.attributes.get('name') == 'q' or  # Google's search input name
                element.attributes.get('title', '').lower().find('search') != -1):
                actions.append({
                    "type": "input_text",
                    "params": InputTextAction(
                        index=idx,
                        text="reinforcement learning"
                    ).model_dump()
                })
        
        # Handle clickable elements
        if element.tag_name.lower() in ['button', 'a'] or element.attributes.get('role') == 'button':
            actions.append({
                "type": "click_element",
                "params": ClickElementAction(index=idx).model_dump()
            })
        
        return actions
    
    async def get_valid_actions(self) -> List[Dict[str, Any]]:
        """
        Get list of currently valid actions based on browser state.
        Uses browser-use's native system for determining interactive elements.
        """
        valid_actions = []
        state = await self.browser_context.get_state()
        
        
        # If not on Google, first action should be to navigate there
        if not state.url or "google.com" not in state.url.lower():
            valid_actions.append({
                "type": "go_to_url",
                "params": GoToUrlAction(url="https://www.google.com").model_dump()
            })
            return valid_actions
            
        # Get interactive elements from the selector map
        if state.selector_map:
            for idx, element in state.selector_map.items():
                # Get all valid actions for this element
                element_actions = self._get_action_for_element(element, idx)
                valid_actions.extend(element_actions)
        
        # Always allow scrolling and extracting content
        valid_actions.append({
            "type": "scroll",
            "params": ScrollAction(amount=300).model_dump()
        })
        
        valid_actions.append({
            "type": "extract_page_content",
            "params": NoParamsAction().model_dump()
        })
        
        print(f"Total valid actions: {len(valid_actions)}")
        return valid_actions
    
    def action_space_size(self) -> int:
        """Get the current size of the action space."""
        return len(self.controller.registry.get_prompt_description().split('\n'))
    
    def is_terminal_action(self, action_type: str) -> bool:
        """Check if an action type is terminal (ends the episode)."""
        return action_type == "done"
