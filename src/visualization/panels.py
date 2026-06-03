"""Render individual visualization panels as BGR images."""

from __future__ import annotations

import cv2
import numpy as np

from src.masks import overlay_mask_on_image
from src.visualization._common import CLASS_COLORS_BGR
from src.visualization.context import ImageInspectionContext
from src.visualization.legacy import draw_polygon_annotations


def render_rgb(ctx: ImageInspectionContext) -> np.ndarray:
    return ctx.image_bgr.copy()


def render_mask(ctx: ImageInspectionContext) -> np.ndarray:
    return overlay_mask_on_image(ctx.image_bgr, ctx.fish_mask, color=(0, 255, 0), alpha=0.45)


def render_skeleton(ctx: ImageInspectionContext) -> np.ndarray:
    vis = ctx.image_bgr.copy()
    skel = ctx.skeleton
    vis[skel > 0] = (255, 0, 255)
    if ctx.skeleton_path:
        for x, y in ctx.skeleton_path:
            cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)
    return vis


def render_pca(ctx: ImageInspectionContext) -> np.ndarray:
    vis = ctx.image_bgr.copy()
    half = ctx.pca_length_px / 2.0
    p0 = ctx.pca_center - ctx.pca_axis * half
    p1 = ctx.pca_center + ctx.pca_axis * half
    cv2.line(vis, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 0, 255), 2)
    cv2.circle(vis, tuple(ctx.pca_center.astype(int)), 4, (255, 0, 0), -1)
    return vis


def render_calibration(ctx: ImageInspectionContext) -> np.ndarray:
    vis = draw_polygon_annotations(ctx.sample, image_bgr=ctx.image_bgr)
    for ann in ctx.sample.annotations:
        if ann.class_name not in ("blue", "yellow"):
            continue
        if ann.coords_pixels is None or len(ann.coords_pixels) < 3:
            continue
        rect = cv2.minAreaRect(ann.coords_pixels.astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.int32)
        color = CLASS_COLORS_BGR[ann.class_name]
        cv2.polylines(vis, [box], True, color, 2)
    ppm = ctx.calibration.pixels_per_mm
    cv2.putText(
        vis,
        f"scale: {ppm:.3f} px/mm ({ctx.scale_source})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def _hough_line_segments(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        gray_small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray

    edges = cv2.Canny(gray_small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=15,
    )
    segments: list[tuple[int, int, int, int]] = []
    if lines is not None:
        inv = 1.0 / scale if scale != 1.0 else 1.0
        for x1, y1, x2, y2 in lines[:, 0]:
            segments.append(
                (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))
            )
    return segments


def render_grid(ctx: ImageInspectionContext) -> np.ndarray:
    vis = ctx.image_bgr.copy()
    for x1, y1, x2, y2 in _hough_line_segments(ctx.image_bgr):
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if angle < 15 or angle > 165:
            color = (0, 255, 255)
        elif 75 < angle < 105:
            color = (255, 128, 0)
        else:
            color = (128, 128, 128)
        cv2.line(vis, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    grid = ctx.grid
    if grid is not None and grid.success:
        text = (
            f"grid: {grid.pixels_per_grid_square:.1f} px/sq, "
            f"{grid.pixels_per_mm:.3f} px/mm (H={grid.n_horizontal_lines}, V={grid.n_vertical_lines})"
        )
    else:
        text = "grid: detection failed"
    cv2.putText(
        vis,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def render_perspective(ctx: ImageInspectionContext) -> np.ndarray:
    if ctx.rectified_bgr is not None:
        h, w = ctx.original_bgr.shape[:2]
        rect_small = cv2.resize(ctx.rectified_bgr, (w // 2, h // 2))
        orig_small = cv2.resize(ctx.original_bgr, (w // 2, h // 2))
        return np.hstack([orig_small, rect_small])
    vis = ctx.original_bgr.copy()
    cv2.putText(
        vis,
        "Perspective correction disabled",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def render_depth(ctx: ImageInspectionContext) -> np.ndarray:
    if ctx.depth_map is None:
        vis = ctx.image_bgr.copy()
        cv2.putText(
            vis,
            "Depth map not available",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    depth = ctx.depth_map.astype(np.float64)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        vis = ctx.image_bgr.copy()
        cv2.putText(vis, "Depth map empty", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis

    lo, hi = np.percentile(finite, [2, 98])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    norm_u8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)
    if ctx.fish_mask.max() > 0:
        mask_bool = ctx.fish_mask > 0
        overlay = cv2.addWeighted(colored, 0.65, ctx.image_bgr, 0.35, 0)
        colored = np.where(mask_bool[:, :, None], overlay, colored)
    return colored


def render_skeleton3d(ctx: ImageInspectionContext) -> np.ndarray:
    vis = ctx.image_bgr.copy()
    path = ctx.skeleton_path
    if not path or ctx.depth_map is None:
        cv2.putText(
            vis,
            "3D skeleton: need depth + skeleton path",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    depth = ctx.depth_map
    zs = []
    for x, y in path:
        z = float(depth[y, x]) if np.isfinite(depth[y, x]) else 0.0
        zs.append(z * ctx.depth_scale)
    zs_arr = np.array(zs, dtype=np.float64)
    z_min, z_max = zs_arr.min(), zs_arr.max()
    z_range = max(z_max - z_min, 1e-6)

    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        t = (zs_arr[i] - z_min) / z_range
        color = (int(255 * (1 - t)), 0, int(255 * t))
        cv2.line(vis, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)

    if ctx.length_3d_mm is not None:
        cv2.putText(
            vis,
            f"3D arc: {ctx.length_3d_mm:.1f} mm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def render_measurement(ctx: ImageInspectionContext) -> np.ndarray:
    vis = overlay_mask_on_image(ctx.image_bgr, ctx.fish_mask, color=(0, 255, 0), alpha=0.25)
    method = ctx.method

    if method == "bbox":
        ys, xs = np.where(ctx.fish_mask > 0)
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 128, 0), 2)
            cv2.line(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
    elif method == "pca":
        vis = render_pca(ctx)
        vis = overlay_mask_on_image(vis, ctx.fish_mask, color=(0, 255, 0), alpha=0.2)
    else:
        vis = render_skeleton(ctx)

    pred = ctx.predicted_mm
    gt = ctx.ground_truth_mm
    lines = [f"method: {method}", f"predicted: {pred:.1f} mm" if pred else "predicted: n/a"]
    if gt is not None:
        lines.append(f"ground truth: {gt:.1f} mm")
    if ctx.abs_error_mm is not None:
        lines.append(f"abs error: {ctx.abs_error_mm:.1f} mm")

    y = 25
    for line in lines:
        cv2.putText(
            vis,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return vis


PANEL_RENDERERS = {
    "rgb": render_rgb,
    "mask": render_mask,
    "skeleton": render_skeleton,
    "pca": render_pca,
    "calibration": render_calibration,
    "grid": render_grid,
    "perspective": render_perspective,
    "depth": render_depth,
    "skeleton3d": render_skeleton3d,
    "measurement": render_measurement,
}

PANEL_TITLES = {
    "rgb": "Original RGB",
    "mask": "Fish mask",
    "skeleton": "Skeleton",
    "pca": "PCA major axis",
    "calibration": "Calibration landmarks",
    "grid": "Grid detections",
    "perspective": "Perspective correction",
    "depth": "Depth map",
    "skeleton3d": "3D skeleton projection",
    "measurement": "Final measurement",
}


def render_panel(ctx: ImageInspectionContext, panel: str) -> np.ndarray:
    if panel not in PANEL_RENDERERS:
        raise ValueError(f"Unknown panel {panel!r}; choose from {list(PANEL_RENDERERS)}")
    return PANEL_RENDERERS[panel](ctx)
