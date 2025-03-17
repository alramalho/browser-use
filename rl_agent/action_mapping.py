from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto

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
    ExtractPageContentAction
)

class ActionType(Enum):
    """Enumeration of all possible action types."""
    SEARCH = auto()
    NAVIGATE = auto()
    CLICK = auto()
    TYPE = auto()
    DONE = auto()
    SWITCH_TAB = auto()
    NEW_TAB = auto()
    SCROLL = auto()
    SEND_KEYS = auto()
    EXTRACT = auto()

@dataclass
class ActionMapping:
    """Maps between action dictionaries and indices."""
    action_type: ActionType
    params: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, action_dict: Dict[str, Any]) -> 'ActionMapping':
        """Convert action dictionary to ActionMapping."""
        action_type = ActionType[action_dict["type"].upper()]
        return cls(
            action_type=action_type,
            params=action_dict["params"]
        )
    
    def to_browser_action(self) -> Dict[str, Any]:
        """Convert to browser-use action format."""
        action_class = self.get_action_class()
        action_instance = action_class(**self.params)
        return {self.action_type.name.lower(): action_instance}
    
    def get_action_class(self):
        """Get the corresponding browser-use action class."""
        action_classes = {
            ActionType.SEARCH: SearchGoogleAction,
            ActionType.NAVIGATE: GoToUrlAction,
            ActionType.CLICK: ClickElementAction,
            ActionType.TYPE: InputTextAction,
            ActionType.DONE: DoneAction,
            ActionType.SWITCH_TAB: SwitchTabAction,
            ActionType.NEW_TAB: OpenTabAction,
            ActionType.SCROLL: ScrollAction,
            ActionType.SEND_KEYS: SendKeysAction,
            ActionType.EXTRACT: ExtractPageContentAction
        }
        return action_classes[self.action_type]

class ActionSpace:
    """
    Manages the mapping between high-level actions and their indices.
    Provides utilities for converting between different action representations.
    """
    def __init__(self):
        self.action_types = list(ActionType)
        self._type_to_idx = {
            action_type: idx 
            for idx, action_type in enumerate(self.action_types)
        }
        self._idx_to_type = {
            idx: action_type 
            for idx, action_type in enumerate(self.action_types)
        }
    
    def action_to_idx(self, action: Dict[str, Any]) -> int:
        """Convert action dictionary to index."""
        action_mapping = ActionMapping.from_dict(action)
        return self._type_to_idx[action_mapping.action_type]
    
    def idx_to_action_type(self, idx: int) -> ActionType:
        """Convert index to action type."""
        return self._idx_to_type[idx]
    
    def create_action(
        self,
        action_type: ActionType,
        **params
    ) -> Dict[str, Any]:
        """Create an action dictionary with the given type and parameters."""
        return {
            "type": action_type.name.lower(),
            "params": params
        }
    
    @property
    def n_actions(self) -> int:
        """Get the total number of action types."""
        return len(self.action_types)
    
    def get_template_params(self, action_type: ActionType) -> Dict[str, Any]:
        """Get template parameters for an action type."""
        templates = {
            ActionType.SEARCH: {"query": ""},
            ActionType.NAVIGATE: {"url": ""},
            ActionType.CLICK: {"index": 0},
            ActionType.TYPE: {"index": 0, "text": ""},
            ActionType.DONE: {"text": "", "success": False},
            ActionType.SWITCH_TAB: {"page_id": 0},
            ActionType.NEW_TAB: {"url": ""},
            ActionType.SCROLL: {"amount": None},
            ActionType.SEND_KEYS: {"keys": ""},
            ActionType.EXTRACT: {"value": ""}
        }
        return templates[action_type]
    
    def create_valid_actions(
        self,
        clickable_indices: List[int] = None,
        has_multiple_tabs: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Create list of valid actions given the current state.
        
        Args:
            clickable_indices: List of valid element indices for click/type actions
            has_multiple_tabs: Whether tab switching is available
        """
        valid_actions = []
        
        # Always valid actions
        valid_actions.extend([
            self.create_action(ActionType.SEARCH, **self.get_template_params(ActionType.SEARCH)),
            self.create_action(ActionType.NAVIGATE, **self.get_template_params(ActionType.NAVIGATE)),
            self.create_action(ActionType.DONE, **self.get_template_params(ActionType.DONE)),
            self.create_action(ActionType.SCROLL, **self.get_template_params(ActionType.SCROLL)),
            self.create_action(ActionType.EXTRACT, **self.get_template_params(ActionType.EXTRACT))
        ])
        
        # Add click/type actions for valid indices
        if clickable_indices:
            for idx in clickable_indices:
                valid_actions.extend([
                    self.create_action(ActionType.CLICK, index=idx),
                    self.create_action(ActionType.TYPE, index=idx, text="")
                ])
        
        # Add tab actions if multiple tabs exist
        if has_multiple_tabs:
            valid_actions.append(
                self.create_action(ActionType.SWITCH_TAB, **self.get_template_params(ActionType.SWITCH_TAB))
            )
        
        return valid_actions 