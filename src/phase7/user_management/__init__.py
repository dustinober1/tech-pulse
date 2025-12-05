"""
Phase 7.2 User Management Module
Provides comprehensive user personalization and profile management features
"""

from .database import UserDatabase
from .user_profile import UserProfile
from .recommendations import PersonalizedRecommendations, Recommendation
from .ui_components import UIComponents

__all__ = [
    'UserDatabase',
    'UserProfile',
    'PersonalizedRecommendations',
    'Recommendation',
    'UIComponents'
]

__version__ = '1.0.0'