"""Movement Heatmap Module for Retail Analytics.

Provides spatial movement and occupancy density heatmaps by accumulating
anonymous foot ground-contact positions (TrackedPerson.bottom_center) over time,
applying Gaussian smoothing, and rendering a translucent colormap overlay
while keeping the original video visible underneath.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.tracker import TrackingBatch


class MovementHeatmap:
    """Accumulates tracked person foot positions and renders spatial heatmaps.

    Guarantees user privacy: only anonymous 2D ground coordinates are retained,
    with zero facial or biometric data stored.
    """

    def __init__(
        self,
        frame_shape: Tuple[int, int],
        blur_kernel: int = 31,
        alpha: float = 0.50,
        colormap: int = cv2.COLORMAP_JET,
    ):
        """Initialize the heatmap accumulator.

        Args:
            frame_shape: (height, width) of the video frame.
            blur_kernel: Odd integer kernel size for Gaussian smoothing.
            alpha: Transparency factor in [0.0, 1.0] for blending over video.
            colormap: OpenCV colormap enum (defaults to cv2.COLORMAP_JET).
        """
        self.height, self.width = frame_shape[:2]
        # Ensure blur_kernel is positive and odd
        k = max(3, int(blur_kernel))
        self.blur_kernel = k if k % 2 != 0 else k + 1
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.colormap = colormap
        self.accumulator = np.zeros((self.height, self.width), dtype=np.float32)

    def update(self, batch: TrackingBatch) -> None:
        """Accumulate foot ground-contact points from the current frame tracks.

        Args:
            batch: TrackingBatch containing active TrackedPerson objects.
        """
        for track in batch.tracks:
            cx, cy = track.bottom_center
            if 0 <= cy < self.height and 0 <= cx < self.width:
                self.accumulator[cy, cx] += 1.0

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render and blend the spatial heatmap onto the provided frame.

        Unvisited regions (zero or negligible heat) remain 100% untouched so the
        original video is fully visible underneath.

        Args:
            frame: Original BGR video frame.

        Returns:
            Frame with overlaid movement heatmap.
        """
        # Adapt accumulator if frame dimensions differ
        h, w = frame.shape[:2]
        if (h, w) != (self.height, self.width):
            self.height, self.width = h, w
            self.accumulator = cv2.resize(
                self.accumulator, (w, h), interpolation=cv2.INTER_LINEAR
            )

        max_val = float(np.max(self.accumulator))
        if max_val <= 0.0:
            return frame

        # Apply Gaussian smoothing to spread discrete accumulation points
        blurred = cv2.GaussianBlur(
            self.accumulator, (self.blur_kernel, self.blur_kernel), 0
        )

        blur_max = float(np.max(blurred))
        if blur_max <= 0.0:
            return frame

        # Normalize density to 8-bit scale [0, 255]
        norm_map = np.clip((blurred / blur_max) * 255.0, 0, 255).astype(np.uint8)

        # Generate false-color density visualization
        colored_heatmap = cv2.applyColorMap(norm_map, self.colormap)

        # Mask out cold/unvisited areas (threshold = 5) to keep video fully visible
        mask = norm_map > 5
        if not np.any(mask):
            return frame

        output = frame.copy()
        blended = cv2.addWeighted(
            colored_heatmap, self.alpha, frame, 1.0 - self.alpha, 0
        )
        output[mask] = blended[mask]
        return output

    def reset(self, frame_shape: Optional[Tuple[int, int]] = None) -> None:
        """Clear the accumulation grid."""
        if frame_shape is not None:
            self.height, self.width = frame_shape[:2]
        self.accumulator = np.zeros((self.height, self.width), dtype=np.float32)
