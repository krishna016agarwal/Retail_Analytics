"""Step 31: Visual Evidence & Inventory Event Contact Sheet Generator.

Generates representative visual evidence across all Step 31 event lifecycle dimensions:
- Panel 1: Normal Stocked Shelf (0 active alerts, clean state)
- Panel 2: Transient Gap Under Observation (Persistence < 10 frames, no alert emitted)
- Panel 3: Active Vacancy Confirmed Event (Frame 11, EVT-RUN-001 emitted)
- Panel 4: Replenishment Recommendation Alert (Confidence >= 70%, Action Required)
- Panel 5: Shopper Occlusion Safety (Event freeze during human interaction)
- Panel 6: Shelf Restored Lifecycle (Gap refilled, event RESOLVED, SHELF_RESTORED logged)
- Panel 7: Operator Acknowledged State (Status ACKNOWLEDGED, audit trail timestamped)
- Panel 8: Multi-Tier Simultaneous Events (Independent tracking on T1 and T2)
- Panel 9: Audit Trail Table HUD (Visual representation of persistent event log)

Consolidates all 9 panels into output/step31_evidence/step31_contact_sheet.jpg (3x3 grid).
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inventory.shelf_events import (
    ShelfEventManager,
    ShelfEventType,
    ShelfEventStatus,
    ShelfInventoryEvent,
)
from inventory.shelf_vacancy import (
    ShelfVacancyEngine,
    ShelfVacancyTracker,
    ShelfVacancySnapshot,
    annotate_vacancy_frame,
)

OUT_DIR = ROOT_DIR / "output" / "step31_evidence"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_event_hud_badge(
    img: np.ndarray,
    event: ShelfInventoryEvent,
    title: str,
    action_note: str = "",
) -> np.ndarray:
    """Renders a high-visibility event lifecycle badge directly on top of the image."""
    canvas = img.copy()
    h, w = canvas.shape[:2]

    # Semi-transparent overlay banner at top
    overlay = canvas.copy()
    cv2.rectangle(overlay, (20, 20), (w - 20, 110), (15, 18, 25), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
    border_color = (
        (60, 60, 230)
        if event.status == ShelfEventStatus.ACTIVE.value
        else (60, 200, 100)
        if event.status == ShelfEventStatus.RESOLVED.value
        else (240, 180, 50)
    )
    cv2.rectangle(canvas, (20, 20), (w - 20, 110), border_color, 2)

    # Title and Event Type
    cv2.putText(canvas, title, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    ev_type_str = f"EVENT: {event.event_type} | STATUS: {event.status}"
    cv2.putText(canvas, ev_type_str, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, border_color, 2)

    # Right side: Conf & ID
    info_str = f"ID: {event.event_id} | Conf: {int(event.visual_vacancy_confidence * 100)}% | Dur: {event.duration_sec:.1f}s"
    cv2.putText(canvas, info_str, (w - 480, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 240), 1)

    if action_note:
        cv2.putText(canvas, action_note, (w - 480, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 240, 255), 1)

    return canvas


def create_contact_sheet(
    image_paths: List[pathlib.Path],
    output_path: pathlib.Path,
    grid_cols: int = 3,
    thumb_w: int = 640,
    thumb_h: int = 480,
) -> None:
    """Creates a consolidated multi-panel contact sheet."""
    images = []
    for p in image_paths:
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                images.append(resized)

    if not images:
        print("No images found for contact sheet.")
        return

    num_images = len(images)
    grid_rows = int(np.ceil(num_images / grid_cols))
    sheet_h = grid_rows * thumb_h
    sheet_w = grid_cols * thumb_w

    contact_sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r = idx // grid_cols
        c = idx % grid_cols
        y1 = r * thumb_h
        y2 = y1 + thumb_h
        x1 = c * thumb_w
        x2 = x1 + thumb_w
        contact_sheet[y1:y2, x1:x2] = img

    cv2.imwrite(str(output_path), contact_sheet)
    print(f"Step 31 Contact sheet saved: {output_path} ({sheet_w}x{sheet_h})")


def run_step31_visual_evidence():
    print("=" * 80)
    print("STEP 31: INVENTORY EVENT & REPLENISHMENT ACTION LAYER VISUAL EVIDENCE")
    print("=" * 80)

    frame_w, frame_h = 1280, 720
    shelf_roi = (0.05, 0.15, 0.95, 0.88)
    tiers = [
        {"tier_id": "SHELF-01-T1", "name": "Tier 1 — Top Rack", "roi": (0.05, 0.18, 0.95, 0.48)},
        {"tier_id": "SHELF-01-T2", "name": "Tier 2 — Bottom Rack", "roi": (0.05, 0.52, 0.95, 0.82)},
    ]

    engine = ShelfVacancyEngine(
        shelf_id="SHELF-01",
        roi=shelf_roi,
        tiers=tiers,
        min_gap_multiplier=1.75,
        min_absolute_gap_px=45,
    )
    tracker = ShelfVacancyTracker(
        min_consecutive_frames=10,
        recovery_frames=5,
        min_replenishment_confidence=0.70,
    )

    mgr = ShelfEventManager(
        run_id="DEMO-STEP31",
        camera_id="CAM-01",
        storage_path=OUT_DIR / "evidence_events.json",
        dashboard_sync_path=None,
    )

    def generate_shelf_canvas(product_boxes: List[Tuple[int, int, int, int]], person_boxes: List[Tuple[int, int, int, int]] = None):
        img = np.full((frame_h, frame_w, 3), 28, dtype=np.uint8)
        # Shelf backgrounds
        cv2.rectangle(img, (int(frame_w*0.05), int(frame_h*0.18)), (int(frame_w*0.95), int(frame_h*0.48)), (42, 42, 42), -1)
        cv2.rectangle(img, (int(frame_w*0.05), int(frame_h*0.52)), (int(frame_w*0.95), int(frame_h*0.82)), (42, 42, 42), -1)
        for b in product_boxes:
            bx1, by1, bx2, by2 = b
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (180, 140, 60), -1)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (220, 180, 90), 2)
        if person_boxes:
            for p in person_boxes:
                px1, py1, px2, py2 = p
                overlay = img.copy()
                cv2.rectangle(overlay, (px1, py1), (px2, py2), (50, 50, 200), -1)
                cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
                cv2.rectangle(img, (px1, py1), (px2, py2), (80, 80, 255), 2)
                cv2.putText(img, "SHOPPER OCCLUSION", (px1 + 10, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def build_boxes(tier_y1=145, tier_y2=330, item_w=75, spacing=15, skip_indices=()):
        boxes = []
        x_start = int(frame_w * 0.06)
        x_end = int(frame_w * 0.94)
        curr_x = x_start
        idx = 0
        while curr_x + item_w <= x_end:
            if idx not in skip_indices:
                boxes.append((curr_x, tier_y1, curr_x + item_w, tier_y2))
            curr_x += item_w + spacing
            idx += 1
        return boxes

    panels: List[pathlib.Path] = []

    # --- PANEL 1: Normal Stocked Shelf (0 events) ---
    t1_norm = build_boxes(tier_y1=145, tier_y2=330, spacing=12)
    t2_norm = build_boxes(tier_y1=390, tier_y2=575, spacing=12)
    tracker.reset()
    mgr.clear()
    snap1 = tracker.update(engine.analyze_frame(t1_norm + t2_norm, frame_w, frame_h))
    mgr.process_frame(snap1, frame_index=1, timestamp_sec=0.1, fps=10.0)
    canvas1 = annotate_vacancy_frame(generate_shelf_canvas(t1_norm + t2_norm), snap1, shelf_roi, False)
    cv2.putText(canvas1, "PANEL 1: NORMAL STOCKED SHELF (0 EVENTS)", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 100), 2)
    p1 = OUT_DIR / "panel1_normal_stocked.jpg"
    cv2.imwrite(str(p1), canvas1)
    panels.append(p1)

    # --- PANEL 2: Temporary Gap Under Observation (Persistence < 10) ---
    t1_gap = build_boxes(tier_y1=145, tier_y2=330, skip_indices=(4, 5))
    all_gap = t1_gap + t2_norm
    tracker.reset()
    for f in range(3):
        snap2 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h))
        mgr.process_frame(snap2, frame_index=f, timestamp_sec=f * 0.1, fps=10.0)
    canvas2 = annotate_vacancy_frame(generate_shelf_canvas(all_gap), snap2, shelf_roi, False)
    cv2.putText(canvas2, "PANEL 2: TRANSIENT GAP (OBSERVING - NO EVENT YET)", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 220, 255), 2)
    p2 = OUT_DIR / "panel2_transient_observing.jpg"
    cv2.imwrite(str(p2), canvas2)
    panels.append(p2)

    # --- PANEL 3: Active Vacancy Confirmed Event (Frame 11) ---
    for f in range(3, 11):
        snap3 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h))
        mgr.process_frame(snap3, frame_index=f, timestamp_sec=f * 0.1, fps=10.0)
    active_evts = mgr.get_active_events()
    ev3 = mgr._events[0]
    canvas3 = annotate_vacancy_frame(generate_shelf_canvas(all_gap), snap3, shelf_roi, True)
    canvas3 = draw_event_hud_badge(canvas3, ev3, "PANEL 3: VACANCY CONFIRMED EVENT", "Lifecycle: State -> CONFIRMED")
    p3 = OUT_DIR / "panel3_vacancy_confirmed.jpg"
    cv2.imwrite(str(p3), canvas3)
    panels.append(p3)

    # --- PANEL 4: Replenishment Recommendation Alert ---
    for f in range(11, 15):
        snap4 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h))
        mgr.process_frame(snap4, frame_index=f, timestamp_sec=f * 0.1, fps=10.0)
    ev4 = mgr._events[0]
    canvas4 = annotate_vacancy_frame(generate_shelf_canvas(all_gap), snap4, shelf_roi, True)
    canvas4 = draw_event_hud_badge(canvas4, ev4, "PANEL 4: REPLENISHMENT RECOMMENDED", "HIGH SEVERITY ALERT EMITTED")
    p4 = OUT_DIR / "panel4_replenishment_recommended.jpg"
    cv2.imwrite(str(p4), canvas4)
    panels.append(p4)

    # --- PANEL 5: Shopper Occlusion Safety Freeze ---
    shopper_box = [(int(frame_w * 0.32), int(frame_h * 0.12), int(frame_w * 0.65), int(frame_h * 0.90))]
    snap5 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h, person_boxes=shopper_box))
    mgr.process_frame(snap5, frame_index=15, timestamp_sec=1.5, fps=10.0)
    canvas5 = annotate_vacancy_frame(generate_shelf_canvas(all_gap, shopper_box), snap5, shelf_roi, False)
    cv2.putText(canvas5, "PANEL 5: OCCLUSION SAFETY FREEZE (NO FALSE RESOLVE)", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 180, 50), 2)
    p5 = OUT_DIR / "panel5_occlusion_freeze.jpg"
    cv2.imwrite(str(p5), canvas5)
    panels.append(p5)

    # --- PANEL 6: Shelf Restored Lifecycle Event ---
    tracker.reset()
    snap6 = tracker.update(engine.analyze_frame(t1_norm + t2_norm, frame_w, frame_h))
    mgr.process_frame(snap6, frame_index=16, timestamp_sec=1.6, fps=10.0)
    restored_evt = mgr._events[-1]
    canvas6 = annotate_vacancy_frame(generate_shelf_canvas(t1_norm + t2_norm), snap6, shelf_roi, False)
    canvas6 = draw_event_hud_badge(canvas6, restored_evt, "PANEL 6: SHELF RESTORED (EVENT RESOLVED)", "Audit: Restored in 1.6s")
    p6 = OUT_DIR / "panel6_shelf_restored.jpg"
    cv2.imwrite(str(p6), canvas6)
    panels.append(p6)

    # --- PANEL 7: Operator Acknowledged State ---
    # Re-trigger gap on T1 and acknowledge it
    for f in range(12):
        snap7 = tracker.update(engine.analyze_frame(all_gap, frame_w, frame_h))
        mgr.process_frame(snap7, frame_index=20 + f, timestamp_sec=2.0 + f * 0.1, fps=10.0)
    active7 = mgr.get_active_events()[0]
    mgr.acknowledge_event(active7["event_id"])
    ack_evt = [e for e in mgr._events if e.event_id == active7["event_id"]][0]
    canvas7 = annotate_vacancy_frame(generate_shelf_canvas(all_gap), snap7, shelf_roi, True)
    canvas7 = draw_event_hud_badge(canvas7, ack_evt, "PANEL 7: OPERATOR ACKNOWLEDGED", "Operator Action: In Progress")
    p7 = OUT_DIR / "panel7_operator_acknowledged.jpg"
    cv2.imwrite(str(p7), canvas7)
    panels.append(p7)

    # --- PANEL 8: Multi-Tier Simultaneous Events ---
    t2_gap = build_boxes(tier_y1=390, tier_y2=575, skip_indices=(7, 8))
    multi_gap_boxes = t1_gap + t2_gap
    tracker.reset()
    for f in range(12):
        snap8 = tracker.update(engine.analyze_frame(multi_gap_boxes, frame_w, frame_h))
        mgr.process_frame(snap8, frame_index=40 + f, timestamp_sec=4.0 + f * 0.1, fps=10.0)
    canvas8 = annotate_vacancy_frame(generate_shelf_canvas(multi_gap_boxes), snap8, shelf_roi, True)
    cv2.putText(canvas8, "PANEL 8: MULTI-TIER CONCURRENT EVENTS (T1 & T2)", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 100, 200), 2)
    p8 = OUT_DIR / "panel8_multitier_events.jpg"
    cv2.imwrite(str(p8), canvas8)
    panels.append(p8)

    # --- PANEL 9: Audit Trail HUD Table ---
    hud = np.full((frame_h, frame_w, 3), 18, dtype=np.uint8)
    cv2.rectangle(hud, (30, 30), (frame_w - 30, frame_h - 30), (35, 38, 48), -1)
    cv2.rectangle(hud, (30, 30), (frame_w - 30, frame_h - 30), (70, 80, 100), 2)

    cv2.putText(hud, "PANEL 9: PERSISTENT INVENTORY EVENT AUDIT TRAIL", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(hud, "Live query: /inventory/events/history (JSON backed storage)", (60, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 170, 200), 1)

    headers = ["EVENT ID", "TIER", "TYPE", "STATUS", "CONF", "DURATION"]
    xs = [60, 330, 480, 800, 960, 1080]
    y_table = 160
    for idx, (h_title, x_pos) in enumerate(zip(headers, xs)):
        cv2.putText(hud, h_title, (x_pos, y_table), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 2)
    cv2.line(hud, (50, y_table + 12), (frame_w - 50, y_table + 12), (80, 90, 110), 1)

    events_to_show = mgr.get_history()[:7]
    row_y = y_table + 45
    for ev in events_to_show:
        col = (
            (80, 80, 255)
            if ev["status"] == "ACTIVE"
            else (100, 240, 120)
            if ev["status"] == "RESOLVED"
            else (255, 200, 80)
        )
        cv2.putText(hud, ev["event_id"][:22], (xs[0], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        cv2.putText(hud, ev["tier_id"][:12], (xs[1], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 210, 255), 1)
        cv2.putText(hud, ev["event_type"][:22], (xs[2], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(hud, ev["status"], (xs[3], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)
        cv2.putText(hud, f"{int(ev['visual_vacancy_confidence']*100)}%", (xs[4], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 220, 100), 1)
        cv2.putText(hud, f"{ev['duration_sec']:.1f}s", (xs[5], row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        row_y += 40

    p9 = OUT_DIR / "panel9_audit_trail_table.jpg"
    cv2.imwrite(str(p9), hud)
    panels.append(p9)

    # Consolidate 3x3 contact sheet
    sheet_path = OUT_DIR / "step31_contact_sheet.jpg"
    create_contact_sheet(panels, sheet_path, grid_cols=3)
    print("Step 31 visual evidence generated successfully!")


if __name__ == "__main__":
    run_step31_visual_evidence()
