"""Configuration parameters for the Intelligent Retail Analytics System (SIH 179).

Provides centralized configuration for detection, tracking, video capture,
and visual display overlays.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class DetectorConfig:
    """YOLO detector configuration options."""

    # Default lightweight YOLO model for CPU inference
    model_path: str = "yolo11n.pt"

    # Inference hardware target (CPU explicitly set for Intel i5-1334U / Iris Xe)
    device: str = "cpu"

    # Default confidence threshold for person detection
    confidence_threshold: float = 0.40

    # Non-Maximum Suppression (IoU) threshold
    iou_threshold: float = 0.45

    # Input image resolution for inference (640 is standard; 480 or 320 for extra speed)
    imgsz: int = 640

    # COCO class index for 'person' is 0
    target_classes: List[int] = field(default_factory=lambda: [0])


@dataclass
class TrackerConfig:
    """ByteTrack tracking and trajectory configuration options."""

    # High-confidence threshold for first-stage association
    track_high_thresh: float = 0.35

    # Low-confidence threshold for second-stage association
    track_low_thresh: float = 0.05

    # Minimum threshold to initialize a new track
    new_track_thresh: float = 0.35

    # Number of frames to keep a lost track in buffer before deletion
    track_buffer: int = 60

    # Matching threshold (IoU) for track-detection association
    match_thresh: float = 0.80

    # Whether to fuse detection score with track score
    fuse_score: bool = True

    # Maximum number of past points retained in trajectory trail for visualization
    max_trajectory_points: int = 40

    # Whether to render motion trajectory trails on video frames
    draw_trajectories: bool = True


@dataclass
class VisualizerConfig:
    """Overlay and styling options for video rendering."""

    # Bounding box color in BGR (Emerald green: (80, 200, 120))
    box_color: Tuple[int, int, int] = (80, 200, 120)
    box_thickness: int = 2

    # Label text styling
    label_bg_color: Tuple[int, int, int] = (30, 30, 30)
    label_text_color: Tuple[int, int, int] = (255, 255, 255)
    font_scale: float = 0.55
    font_thickness: int = 1

    # HUD Banner styling
    hud_bg_color: Tuple[int, int, int] = (20, 24, 33)  # Dark slate
    hud_accent_color: Tuple[int, int, int] = (0, 215, 255)  # Cyan accent
    hud_text_color: Tuple[int, int, int] = (240, 240, 240)
    hud_alpha: float = 0.75  # Transparency factor

    # Trajectory trail styling (BGR Gold/Amber: (0, 190, 255))
    trail_color: Tuple[int, int, int] = (0, 190, 255)
    trail_thickness: int = 2
    trail_circle_radius: int = 3


@dataclass
class AnalyticsConfig:
    """Configuration for retail footfall and occupancy analytics."""

    # Y-coordinate (pixel) of the virtual entrance boundary line
    entrance_line_y: int = 300

    # Direction representing entry ('down': crossing from y < line_y to y > line_y is ENTRY, 'up': opposite)
    entry_direction: str = "down"

    # Line styling
    line_color: Tuple[int, int, int] = (0, 165, 255)  # Orange / Amber (BGR)
    line_thickness: int = 2
    show_entrance_line: bool = True

    # Spatial crossing debounce parameters
    buffer_margin: int = 4  # pixel dead-zone around line to prevent flicker
    debounce_frames: int = 15  # min frames between opposite crossings for same track


@dataclass
class HeatmapConfig:
    """Configuration for movement heatmap accumulation and visualization."""

    enabled: bool = False
    alpha: float = 0.50
    blur_kernel: int = 31  # Must be an odd positive integer


@dataclass
class QueueConfig:
    """Configuration for retail queue zone analytics and congestion monitoring."""

    enabled: bool = False
    # Rectangular zone bounding box (x1, y1, x2, y2) in pixel coordinates
    zone_bbox: Tuple[int, int, int, int] = (300, 200, 600, 400)
    # Congestion count thresholds
    medium_threshold: int = 3
    high_threshold: int = 6
    # Frames before a disappeared track is considered terminated from queue
    disappear_buffer_frames: int = 60
    # Status colors in BGR format
    low_color: Tuple[int, int, int] = (80, 220, 100)      # Green
    medium_color: Tuple[int, int, int] = (0, 190, 255)    # Amber / Orange
    high_color: Tuple[int, int, int] = (50, 50, 240)      # Red


@dataclass
class DatabaseConfig:
    """Configuration for Edge Computing local SQLite database."""

    enabled: bool = True
    db_path: str = "data/retail_edge.db"
    store_id: str = "store_001"
    device_id: str = "edge_device_01"
    snapshot_interval_seconds: float = 5.0


@dataclass
class ZoneConfig:
    """Configuration for an individual retail store zone."""

    id: str
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    expected_staff: int = 1
    max_shopper_capacity: int = 15


@dataclass
class RetailIntelligenceConfig:
    """Configuration for Phase 8 Retail Intelligence Engine."""

    enabled: bool = False
    prediction_minutes: float = 3.0
    queue_high_threshold: int = 6
    growth_rate_threshold: float = 1.0  # people per minute
    max_shoppers_per_staff: float = 4.0  # load threshold
    spike_percentage_threshold: float = 0.50  # 50% increase over baseline
    min_occupancy_for_spike: int = 4
    baseline_window_size: int = 10
    cooldown_seconds: float = 45.0
    zones: List[ZoneConfig] = field(default_factory=list)


@dataclass
class SimulationClockConfig:
    """Configuration for Demo Simulation Store Clock."""

    # Simulated start time of the store day (e.g. 17:00:00 for 5:00 PM)
    demo_start_time: str = "17:00:00"
    # Time acceleration factor (1.0 = real-time, 60.0 = 1 sec video represents 1 min store time)
    time_scale: float = 1.0


@dataclass
class CameraStreamConfig:
    """Configuration for an individual camera stream in multi-camera setup."""

    camera_id: str
    zone_id: str
    video_source: str
    name: str = ""
    role: str = "department"  # 'department', 'checkout', 'entrance'
    expected_staff: int = 1
    is_simulation: bool = True


@dataclass
class FourCameraSetupConfig:
    """Standard 4-camera recorded video store layout."""

    cameras: List[CameraStreamConfig] = field(
        default_factory=lambda: [
            CameraStreamConfig(
                camera_id="CAM_01",
                name="Food",
                zone_id="food",
                video_source="videos/food/food.mp4",
                role="department",
                expected_staff=2,
                is_simulation=True,
            ),
            CameraStreamConfig(
                camera_id="CAM_02",
                name="Electronics",
                zone_id="electronics",
                video_source="videos/electronics/electronics.mp4",
                role="department",
                expected_staff=1,
                is_simulation=True,
            ),
            CameraStreamConfig(
                camera_id="CAM_03",
                name="Grocery",
                zone_id="grocery",
                video_source="videos/grocery/grocery.mp4",
                role="department",
                expected_staff=2,
                is_simulation=True,
            ),
            CameraStreamConfig(
                camera_id="CAM_04",
                name="Checkout",
                zone_id="checkout",
                video_source="videos/checkout/checkout.mp4",
                role="checkout",
                expected_staff=2,
                is_simulation=True,
            ),
        ]
    )
    clock: SimulationClockConfig = field(default_factory=SimulationClockConfig)


@dataclass
class PipelineConfig:
    """Default runtime pipeline configuration."""

    default_video_path: str = str(Path("videos") / "test.mp4")
    window_title: str = "Intelligent Retail Analytics - Footfall & Tracking (SIH 179)"
    enable_tracking: bool = True
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    intelligence: RetailIntelligenceConfig = field(default_factory=RetailIntelligenceConfig)
    four_camera: FourCameraSetupConfig = field(default_factory=FourCameraSetupConfig)

