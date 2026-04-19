"""Research trend tracking and automatic topic generation."""

from research.trends.daily_digest import DailyDigest
from research.trends.trend_analyzer import TrendAnalyzer
from research.trends.opportunity_finder import OpportunityFinder
from research.trends.auto_topic import AutoTopicGenerator
from research.trends.feeds import FeedManager

__all__ = [
    "AutoTopicGenerator",
    "DailyDigest",
    "FeedManager",
    "OpportunityFinder",
    "TrendAnalyzer",
]
