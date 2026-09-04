"""Retail Analytics Computer Vision, Tracking, and Shopper Analytics Package."""

from src.detector import Detection, DetectionBatch, PersonDetector
from src.tracker import (
    PersonTracker,
    TrackedPerson,
    TrackingBatch,
    TrajectoryHistory,
    TrajectoryManager,
)
from src.tracker_botsort import BotSortTracker, BotSortTrackerConfig
from src.shopper_analytics import (
    EntryExitCounter,
    CrossingEvent,
    FootfallMetrics,
    ShopperDwellRecord,
)
from src.heatmap import MovementHeatmap
from src.queue_analytics import CongestionLevel, QueueAnalytics, QueueMetrics
from src.database import AnalyticsSnapshot, EdgeDatabase
from src.visualizer import Visualizer
from src.pipeline import VideoPipeline, PipelineMetrics

__all__ = [
    "Detection",
    "DetectionBatch",
    "PersonDetector",
    "TrackedPerson",
    "TrackingBatch",
    "TrajectoryHistory",
    "TrajectoryManager",
    "PersonTracker",
    "BotSortTracker",
    "BotSortTrackerConfig",
    "EntryExitCounter",
    "CrossingEvent",
    "FootfallMetrics",
    "ShopperDwellRecord",
    "MovementHeatmap",
    "CongestionLevel",
    "QueueAnalytics",
    "QueueMetrics",
    "AnalyticsSnapshot",
    "EdgeDatabase",
    "Visualizer",
    "VideoPipeline",
    "PipelineMetrics",
]
