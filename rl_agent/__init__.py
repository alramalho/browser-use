"""
Reinforcement Learning agent for browser automation.
"""

from .train import train, evaluate
from .model import BrowserPolicy, PPOAgent
from .environment import BrowserEnv
from .reward import RewardConfig
from .experience import ExperienceTracker

__all__ = [
    'train',
    'evaluate',
    'BrowserPolicy',
    'PPOAgent',
    'BrowserEnv',
    'RewardConfig',
    'ExperienceTracker'
]
