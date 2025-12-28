
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F

from .journal import EventJournal
from .models import BrainModels
from .schemas import DeltaFeatures, FramePacket, LayoutHint, OCRSpan, PointerCandidate, RegionProposal
from .utils import bbox_center, iou


def _cluster_components(
    embs: list[list[float]],
    bboxes: list[tuple[float, float, float, float]],
    H: int,
    W: int,
    similarity_threshold: float = 0.75,
) -> list[int]:
    """
    Cluster detected components by embedding similarity.

    Components that look alike (same visual appearance in the encoder's
    embedding space) get the same cluster ID.  This is much more
    generalizable than a fixed role classifier — visually similar
    components naturally share interaction semantics.

    Returns a cluster_id for each component.
    """
    n = len(embs)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Compute pairwise cosine similarity matrix
    emb_t = torch.tensor(embs, dtype=torch.float32)
    norms = emb_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    emb_normed = emb_t / norms
    sim_matrix = emb_normed @ emb_normed.T  # (N, N)

    # Also factor in geometric similarity (aspect ratio, relative size)
    geo_features = []
    for bb in bboxes:
        x1, y1, x2, y2 = bb
        bw, bh = x2 - x1, y2 - y1
        geo_features.append([
            bw / max(bh, 1),          # aspect ratio
            (bw * bh) / (W * H),      # area fraction
            bh / H,                    # relative height
        ])
    geo_t = torch.tensor(geo_features, dtype=torch.float32)
    geo_norms = geo_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    geo_normed = geo_t / geo_norms
    geo_sim = geo_normed @ geo_normed.T

    # Combined similarity: 70% visual, 30% geometric
    combined = 0.7 * sim_matrix + 0.3 * geo_sim

    # Simple greedy clustering (union-find style)
    cluster_ids = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if combined[i, j] > similarity_threshold:
                # Merge j's cluster into i's
                old_id = cluster_ids[j]
                new_id = cluster_ids[i]
                for k in range(n):
                    if cluster_ids[k] == old_id:
                        cluster_ids[k] = new_id

    # Re-index clusters to be contiguous
    unique = {}
    for i, cid in enumerate(cluster_ids):
        if cid not in unique:
            unique[cid] = len(unique)
        cluster_ids[i] = unique[cid]

    return cluster_ids


def _infer_cluster_role(
    cluster_bboxes: list[tuple[float, float, float, float]],
    cluster_embs: list[list[float]],
    H: int,
    W: int,
) -> str:
    """
    Infer the likely interaction role of a cluster from its collective
    geometric and visual properties.

    This is not a classifier — it reads emergent properties of the
    cluster (count, layout pattern, size distribution) to determine
    what kind of UI component family this cluster represents.
    """
    count = len(cluster_bboxes)
    if count == 0:
        return "button"

    # Aggregate geometric stats
    ars = []
    areas = []
    ys = []
    xs = []
    widths = []
    heights = []
    for bb in cluster_bboxes:
        x1, y1, x2, y2 = bb
        bw, bh = x2 - x1, y2 - y1
        ars.append(bw / max(bh, 1))
        areas.append((bw * bh) / (W * H))
        ys.append(y1 / H)
        xs.append(x1 / W)
        widths.append(bw / W)
        heights.append(bh / H)

    mean_ar = sum(ars) / count
    mean_area = sum(areas) / count
    mean_y = sum(ys) / count
    mean_h = sum(heights) / count
    mean_w = sum(widths) / count

    # Many similar items stacked vertically → list_item
    if count >= 3:
        y_sorted = sorted(ys)
        if len(y_sorted) >= 3:
            spacings = [y_sorted[i + 1] - y_sorted[i] for i in range(len(y_sorted) - 1)]
            mean_spacing = sum(spacings) / len(spacings)
            spacing_var = sum((s - mean_spacing) ** 2 for s in spacings) / len(spacings)
            # Regular vertical spacing → list
            if spacing_var < 0.002 and mean_spacing < 0.1:
                return "list_item"

    # Many items in a horizontal row → toolbar or nav
    if count >= 3:
        x_sorted = sorted(xs)
        if mean_h < 0.06 and mean_y < 0.1:
            return "toolbar"

    # Single wide element at top → menubar / nav_bar
    if count <= 2 and mean_ar > 5.0 and mean_y < 0.06 and mean_w > 0.7:
        return "menubar"

    # Single wide element at bottom → tab_bar
    if count <= 2 and mean_ar > 4.0 and mean_y > 0.85 and mean_w > 0.6:
        return "tab_bar"

    # Tall element on the left → sidebar
    mean_x = sum(xs) / count
    if count <= 2 and mean_h > 0.3 and mean_x < 0.2:
        return "sidebar"

    # Small uniform-colored rectangles → button
    if mean_area < 0.03 and mean_ar < 4.0:
        return "button"

    # Thin horizontal strips with high contrast → text_field
    if mean_h < 0.05 and mean_w > 0.1 and mean_ar > 3.0:
        return "text_field"

    # Large areas → container / label
    if mean_area > 0.05:
        return "label"

    return "button"


class PerceptionWorker:
    def __init__(
        self,
        models: BrainModels,
        use_metadata_hints: bool = True,
        grid_size: int = 4,
        desktop_mode: bool = False,
        body: Optional[object] = None,
    ) -> None:
        self.models = models
        self.use_metadata_hints = use_metadata_hints
        self.grid_size = grid_size if not desktop_mode else 8
        self.desktop_mode = desktop_mode
        self.body = body  # optional reference for direct cursor readout
        self.prev_frame: Optional[FramePacket] = None

    def process_frame(
        self,
        frame_event,
        frame_packet: FramePacket,
        journal: EventJournal,
    ) -> list:
        img = frame_packet.image
        H, W = img.shape[-2:]
        events = []

        if self.use_metadata_hints and "elements" in frame_packet.metadata:
            spans = []
            proposals = []
            layout_hints = []
            for el in frame_packet.metadata["elements"]:
                bbox = tuple(float(x) for x in el["bbox"])
                text = el.get("text", "")
                if text:
                    spans.append(OCRSpan(text=text, bbox=bbox, confidence=0.99))
                crop = self._crop(img, bbox)
                emb = self.models.encode_crops(crop.unsqueeze(0))[0].cpu().tolist()
                proposals.append(
                    RegionProposal(
                        bbox=bbox,
                        score=0.95,
                        label_hint=el.get("role", "unknown"),
                        patch_embedding=emb,
                        metadata=el,
                    )
                )
                if el.get("role") in {"list", "toolbar", "modal", "menubar", "sidebar", "dialog", "statusbar"}:
                    kind = el["role"]
                    layout_hints.append(LayoutHint(kind=kind, bbox=bbox, confidence=0.95, metadata=el))

            if "screen_id" in frame_packet.metadata:
                screen_id = frame_packet.metadata["screen_id"]
                layout_hints.append(LayoutHint(kind="nav_bar", bbox=(0, 0, W, 80), confidence=0.8, metadata={"screen_id": screen_id}))

            # ── pointer candidates ──
            ptrs = self._resolve_pointer_candidates(frame_packet)

            delta = self._compute_delta(frame_packet, self.prev_frame)

            events.append(journal.make_event("ocr", frame_event.session_id, frame_event.episode_id, {"spans": spans}, [frame_event.id], frame_packet.t_capture_ns))
            events.append(journal.make_event("regions", frame_event.session_id, frame_event.episode_id, {"proposals": proposals}, [frame_event.id], frame_packet.t_capture_ns))
            events.append(journal.make_event("layout", frame_event.session_id, frame_event.episode_id, {"layout_hints": layout_hints}, [frame_event.id], frame_packet.t_capture_ns))
            events.append(journal.make_event("pointer_candidates", frame_event.session_id, frame_event.episode_id, {"pointer_candidates": ptrs}, [frame_event.id], frame_packet.t_capture_ns))
            events.append(journal.make_event("delta", frame_event.session_id, frame_event.episode_id, {"delta": delta}, [frame_event.id], frame_packet.t_capture_ns))
            self.prev_frame = frame_packet
            return events

        # Fallback path without metadata hints — vision-only element detection.
        proposals = self._edge_based_proposals(img)
        delta = self._compute_delta(frame_packet, self.prev_frame)
        ptrs = self._resolve_pointer_candidates(frame_packet)
        layout_hints = self._layout_hints_from_frame(img)

        events.append(journal.make_event("ocr", frame_event.session_id, frame_event.episode_id, {"spans": []}, [frame_event.id], frame_packet.t_capture_ns))
        events.append(journal.make_event("regions", frame_event.session_id, frame_event.episode_id, {"proposals": proposals}, [frame_event.id], frame_packet.t_capture_ns))
        events.append(journal.make_event("layout", frame_event.session_id, frame_event.episode_id, {"layout_hints": layout_hints}, [frame_event.id], frame_packet.t_capture_ns))
        events.append(journal.make_event("pointer_candidates", frame_event.session_id, frame_event.episode_id, {"pointer_candidates": ptrs}, [frame_event.id], frame_packet.t_capture_ns))
        events.append(journal.make_event("delta", frame_event.session_id, frame_event.episode_id, {"delta": delta}, [frame_event.id], frame_packet.t_capture_ns))
        self.prev_frame = frame_packet
        return events

    def _resolve_pointer_candidates(self, frame_packet: FramePacket) -> list[PointerCandidate]:
        """Determine pointer candidates using the best available source."""
        # 1. Direct cursor from body (desktop / simulator)
        if self.body is not None and hasattr(self.body, "has_direct_cursor") and self.body.has_direct_cursor:
            pos = self.body.get_cursor_position()
            if pos is not None:
                return [PointerCandidate(float(pos[0]), float(pos[1]), 1.0, [1.0, 0.0, 0.0, 0.0])]

        # 2. Metadata hint
        if "pointer_hint" in frame_packet.metadata:
            px, py = frame_packet.metadata["pointer_hint"]
            return [PointerCandidate(float(px), float(py), 0.99, [1.0, 0.0, 0.0, 0.0])]

        # 3. Pixel-based fallback
        return self._pointer_candidates_from_pixels(frame_packet.image)

    def _crop(self, img: torch.Tensor, bbox) -> torch.Tensor:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, img.shape[-1] - 1))
        x2 = max(x1 + 1, min(x2, img.shape[-1]))
        y1 = max(0, min(y1, img.shape[-2] - 1))
        y2 = max(y1 + 1, min(y2, img.shape[-2]))
        crop = img[:, y1:y2, x1:x2]
        crop = F.interpolate(crop.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)[0]
        return crop

    def _grid_proposals(self, img: torch.Tensor) -> list[RegionProposal]:
        """Coarse grid fallback — used when edge detection yields too few regions."""
        _, H, W = img.shape
        rows = cols = self.grid_size
        crops = []
        bboxes = []
        for r in range(rows):
            for c in range(cols):
                x1 = int(c * W / cols)
                x2 = int((c + 1) * W / cols)
                y1 = int(r * H / rows)
                y2 = int((r + 1) * H / rows)
                bboxes.append((float(x1), float(y1), float(x2), float(y2)))
                crops.append(self._crop(img, bboxes[-1]))
        embs = self.models.encode_crops(torch.stack(crops, dim=0)).cpu().tolist()
        out = []
        for bbox, emb in zip(bboxes, embs):
            out.append(RegionProposal(bbox=bbox, score=0.5, label_hint=None, patch_embedding=emb))
        return out

    def _edge_based_proposals(self, img: torch.Tensor) -> list[RegionProposal]:
        """
        Vision-only element detection via edge-based segmentation.

        Pipeline:
          1. Grayscale → Sobel edges (horizontal + vertical)
          2. Threshold → binary edge mask
          3. Morphological close (dilate then erode) to connect nearby edges
          4. Connected-component bounding boxes via row/column projection
          5. Filter by aspect ratio / size to keep UI-element-like regions
          6. Encode each region with TinyVisionEncoder

        This is the pluggable point where SAM or a learned segmentation
        model would replace steps 1-5.
        """
        _, H, W = img.shape
        gray = img.mean(dim=0)  # (H, W)

        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        g = gray.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        ex = F.conv2d(g, sobel_x, padding=1).squeeze()
        ey = F.conv2d(g, sobel_y, padding=1).squeeze()
        edges = (ex.abs() + ey.abs())
        threshold = float(edges.mean().item()) + 1.5 * float(edges.std().item())
        mask = (edges > max(threshold, 0.05)).float()

        # Morphological close: dilate then erode (3x3 kernel)
        kernel = torch.ones(1, 1, 5, 5)
        dilated = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=2)
        mask_closed = (dilated > 0).float().squeeze()

        # Find bounding boxes via connected components (simple scan-line approach)
        bboxes = self._extract_component_bboxes(mask_closed, min_area=200, max_components=40)

        if len(bboxes) < 3:
            # Edge detection didn't find enough — fall back to grid
            return self._grid_proposals(img)

        # Encode each region
        crops = [self._crop(img, bb) for bb in bboxes]
        if not crops:
            return self._grid_proposals(img)
        embs = self.models.encode_crops(torch.stack(crops, dim=0)).cpu().tolist()

        # ── embedding similarity clustering ──
        # Instead of a fixed classifier, cluster components by visual
        # embedding similarity.  Components that look alike share roles.
        # Then infer the role of each cluster from its collective geometry.
        cluster_ids = _cluster_components(embs, bboxes, H, W, similarity_threshold=0.75)

        # Group by cluster
        clusters: dict[int, list[int]] = {}
        for i, cid in enumerate(cluster_ids):
            clusters.setdefault(cid, []).append(i)

        # Infer role per cluster
        cluster_roles: dict[int, str] = {}
        for cid, members in clusters.items():
            c_bboxes = [bboxes[i] for i in members]
            c_embs = [embs[i] for i in members]
            cluster_roles[cid] = _infer_cluster_role(c_bboxes, c_embs, H, W)

        proposals = []
        for i, (bb, emb) in enumerate(zip(bboxes, embs)):
            cid = cluster_ids[i]
            hint = cluster_roles[cid]
            cluster_size = len(clusters[cid])
            # Score: larger clusters → more consistent → higher confidence
            score = 0.55 + 0.15 * min(cluster_size, 5) / 5.0

            proposals.append(RegionProposal(
                bbox=bb,
                score=score,
                label_hint=hint,
                patch_embedding=emb,
                metadata={
                    "cluster_id": cid,
                    "cluster_size": cluster_size,
                    "cluster_role": hint,
                },
            ))

        return proposals

    def _extract_component_bboxes(
        self, mask: torch.Tensor, min_area: int = 200, max_components: int = 40
    ) -> list[tuple[float, float, float, float]]:
        """Extract bounding boxes from a binary mask using horizontal run-length scanning."""
        H, W = mask.shape
        visited = torch.zeros_like(mask, dtype=torch.bool)
        bboxes: list[tuple[float, float, float, float]] = []

        # Downsample for speed on large images
        step = max(1, min(H, W) // 200)
        for y in range(0, H, step):
            for x in range(0, W, step):
                if mask[y, x] < 0.5 or visited[y, x]:
                    continue
                # Flood-fill to find component bounds
                x_min, x_max, y_min, y_max = x, x, y, y
                stack = [(y, x)]
                count = 0
                while stack and count < 5000:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= H or cx < 0 or cx >= W:
                        continue
                    if visited[cy, cx] or mask[cy, cx] < 0.5:
                        continue
                    visited[cy, cx] = True
                    count += 1
                    x_min = min(x_min, cx)
                    x_max = max(x_max, cx)
                    y_min = min(y_min, cy)
                    y_max = max(y_max, cy)
                    for dy, dx in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                        stack.append((cy + dy, cx + dx))

                area = (x_max - x_min) * (y_max - y_min)
                if area < min_area:
                    continue
                # Skip components that span the entire width (background)
                if (x_max - x_min) > W * 0.95 and (y_max - y_min) > H * 0.95:
                    continue
                bboxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))
                if len(bboxes) >= max_components:
                    return bboxes

        return bboxes

    def _pointer_candidates_from_pixels(self, img: torch.Tensor) -> list[PointerCandidate]:
        """Detect pointer from pixels.  Handles both white-dot (mock) and red-circle (servo) cursors."""
        _, H, W = img.shape

        # 1. Red cursor detection (servo overlay: high R, low G, low B)
        r, g, b = img[0], img[1], img[2]
        red_mask = (r > 0.7) & (g < 0.4) & (b < 0.4)
        red_coords = torch.nonzero(red_mask)
        if red_coords.numel() > 6:
            cy = float(red_coords[:, 0].float().mean().item())
            cx = float(red_coords[:, 1].float().mean().item())
            return [PointerCandidate(x=cx, y=cy, confidence=0.85, signature=[1.0, 0.0, 0.0, 0.0])]

        # 2. Bright white cursor (mock phone)
        bright = img.mean(dim=0)
        yx = torch.nonzero(bright > 0.96)
        if yx.numel() > 0:
            y = float(yx[:, 0].float().mean().item())
            x = float(yx[:, 1].float().mean().item())
            return [PointerCandidate(x=x, y=y, confidence=0.5, signature=[1.0, 1.0, 1.0, 0.0])]

        return []

    def _layout_hints_from_frame(self, img: torch.Tensor) -> list[LayoutHint]:
        _, H, W = img.shape
        hints = [
            LayoutHint(kind="nav_bar", bbox=(0.0, 0.0, float(W), 80.0), confidence=0.3),
        ]
        if self.desktop_mode:
            hints.append(LayoutHint(kind="menubar", bbox=(0.0, 0.0, float(W), 30.0), confidence=0.4))
        else:
            hints.append(LayoutHint(kind="tab_bar", bbox=(0.0, float(H - 80), float(W), float(H)), confidence=0.2))
        return hints

    def _compute_delta(self, cur: FramePacket, prev: Optional[FramePacket]) -> DeltaFeatures:
        if prev is None:
            H, W = cur.image.shape[-2:]
            return DeltaFeatures(changed_regions=[(0.0, 0.0, float(W), float(H))], global_change_score=1.0, dominant_motion=(0.0, 0.0))
        d = (cur.image - prev.image).abs().mean(dim=0)
        score = float(d.mean().item())
        mask = d > max(0.02, float(d.mean().item() * 1.5))
        coords = torch.nonzero(mask)
        changed = []
        if coords.numel() > 0:
            y1 = float(coords[:, 0].min().item())
            y2 = float(coords[:, 0].max().item())
            x1 = float(coords[:, 1].min().item())
            x2 = float(coords[:, 1].max().item())
            changed.append((x1, y1, x2, y2))
        else:
            changed.append((0.0, 0.0, 1.0, 1.0))
        return DeltaFeatures(changed_regions=changed, global_change_score=score, dominant_motion=(0.0, 0.0))
