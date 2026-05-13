"""Radial soma extraction + Sholl analysis from single-cell binary masks.

Both metrics share the same scan:

1. Center = EDT peak of the cell mask (deepest inscribed point — by
   construction inside the soma for any normal microglia).
2. For every angle θ on a dense angular grid, cast a ray outward and
   record ``r_out(θ)`` = distance from center to the first background
   pixel along that ray.
3. Walk r outward and inspect the foreground arcs on each ring of radius
   r. The smallest r at which the largest background gap on the ring
   exceeds ``gap_tol_deg`` is the **critical radius** r₀ — the first
   radius where the soma stops being convex. The arcs at r₀ + ε are the
   primary processes; their angular ranges are the "process angles".
4. Soma boundary at non-process angles = ``r_out(θ)`` directly.
   At process angles, ``r_soma(θ)`` is linearly interpolated between
   the nearest non-process angles on either side.
5. Rasterize the resulting polygon → soma mask. Sholl intersections vs r
   are counted for r > r₀, which is where they're physically meaningful.

The one hyperparameter is ``gap_tol_deg`` (default 20°): how big a
background gap on a ring counts as "the soma ended". Smaller values
fire earlier on shape noise; larger values can absorb thin processes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.draw import polygon as draw_polygon


@dataclass
class RadialResult:
    """Per-cell output of :func:`analyze_radial`."""

    # Geometry
    soma_mask: np.ndarray          # 2D bool, same shape as input cell mask
    soma_polygon: np.ndarray       # (N, 2) array of (y, x) vertices
    center_yx: tuple[float, float] # EDT-peak center used for the scan

    # Scalar metrics
    soma_area: float
    soma_perimeter: float
    soma_circularity: float
    soma_to_cell_area_ratio: float
    primary_process_count: int
    critical_radius: float         # r₀, in pixels
    max_process_extent: float      # max r_out, in pixels
    max_intersections: int         # peak of the Sholl profile
    ramification_index: float      # max_intersections / primary_process_count

    # Curves for plotting / debug
    r_out: np.ndarray              # (n_angles,) ray-cast distances
    angles_deg: np.ndarray         # (n_angles,) θ in degrees, 0..360
    process_angle_mask: np.ndarray # (n_angles,) bool — angles inside processes
    sholl_radii: np.ndarray        # (n_radii,) r values
    sholl_intersections: np.ndarray  # (n_radii,) arc count at each r


def analyze_radial(
    cell_mask: np.ndarray,
    *,
    n_angles: int = 360,
    gap_tol_deg: float = 20.0,
    sholl_step: float = 1.0,
    center_yx: tuple[float, float] | None = None,
) -> RadialResult:
    """Run the unified soma + Sholl radial scan on one cell.

    Args:
        cell_mask: 2D boolean / uint8 array, foreground = cell.
        n_angles: angular resolution of the ray fan (360 = 1° steps).
        gap_tol_deg: minimum background gap on a ring (in degrees) that
            counts as "the soma ended". 15–25° is the useful range.
        sholl_step: radial spacing of Sholl rings, in pixels.
        center_yx: optional override for the scan center. Useful for
            DAPI-seeded analysis where the nucleus centroid is the
            biologically grounded "true center" of the soma. If None,
            falls back to the EDT peak (deepest inscribed point) of the
            cell mask. If the supplied center is outside the cell mask,
            it's snapped to the nearest in-mask pixel along the EDT;
            this guards against minor channel mis-registration.

    Raises:
        ValueError on empty mask.
    """
    mask = np.asarray(cell_mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    if not mask.any():
        raise ValueError("Empty mask — no foreground pixels.")

    filled = binary_fill_holes(mask)
    h, w = filled.shape
    edt = distance_transform_edt(filled)

    # ── Center: supplied (snapped if needed) or EDT peak ─────────────
    if center_yx is None:
        cy, cx = np.unravel_index(int(np.argmax(edt)), edt.shape)
        cy_f, cx_f = float(cy), float(cx)
    else:
        cy_f, cx_f = float(center_yx[0]), float(center_yx[1])
        cy_i = int(round(cy_f)); cx_i = int(round(cx_f))
        in_bounds = 0 <= cy_i < h and 0 <= cx_i < w
        if not in_bounds or not filled[cy_i, cx_i]:
            # Snap to the in-mask pixel with the largest EDT value near
            # the supplied point. We just pick the global EDT max here —
            # for any reasonable supplied center, this is essentially
            # the soma's inscribed center.
            cy, cx = np.unravel_index(int(np.argmax(edt)), edt.shape)
            cy_f, cx_f = float(cy), float(cx)

    # ── Per-angle ray-cast ───────────────────────────────────────────
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    angles_deg = np.degrees(angles)

    # Step along each ray in 0.5-pixel increments until we exit the mask
    # or leave the image. r_out(θ) is the last in-mask distance reached.
    max_reach = float(np.hypot(h, w))
    step = 0.5
    n_steps = int(max_reach / step) + 1
    rs = np.arange(n_steps, dtype=float) * step  # (n_steps,)
    # (n_angles, n_steps) coordinates
    ys = cy_f + np.outer(np.sin(angles), rs)
    xs = cx_f + np.outer(np.cos(angles), rs)
    yi = np.round(ys).astype(int)
    xi = np.round(xs).astype(int)
    in_bounds = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
    sampled = np.zeros_like(yi, dtype=bool)
    sampled[in_bounds] = filled[yi[in_bounds], xi[in_bounds]]

    # First background hit along each ray = r_out. argmax on a boolean
    # finds the first True; we want the first False, so invert.
    not_fg = ~sampled
    first_bg = np.argmax(not_fg, axis=1)
    # If a ray never exits, argmax returns 0 — guard by setting to n_steps.
    never_exits = ~not_fg.any(axis=1)
    first_bg[never_exits] = n_steps
    r_out = rs[np.clip(first_bg - 1, 0, n_steps - 1)]
    # Rays whose very first sample is background (center on a thin
    # spur) get r_out = 0; those don't carry useful soma info.
    r_out = np.where(first_bg == 0, 0.0, r_out)

    # ── Critical radius r₀: walk r outward, find first big gap ──────
    # At each ring of radius r, the foreground angles are those where
    # the ray hasn't exited yet (r_out(θ) > r). Find the largest
    # circular gap in that boolean array; if > gap_tol, that's r₀.
    gap_tol_steps = gap_tol_deg / 360.0 * n_angles
    r0 = None
    r0_search = np.arange(1.0, float(r_out.max()) + 1.0, sholl_step)
    for r in r0_search:
        on_ring = r_out > r
        if not on_ring.any():
            r0 = r
            break
        gap = _largest_circular_gap(on_ring)
        if gap > gap_tol_steps:
            r0 = float(r)
            break
    if r0 is None:
        # Ameboid: ring never fragments. Soma = whole cell.
        r0 = float(r_out.max())

    # ── Identify process angles at r₀ + ε ────────────────────────────
    eps_r = r0 + max(1.0, sholl_step)
    on_ring_eps = r_out > eps_r
    if on_ring_eps.any():
        # Connected arcs (circular). Each arc = one primary process.
        arcs = _circular_runs(on_ring_eps)
        primary_process_count = len(arcs)
        process_angle_mask = on_ring_eps.copy()
    else:
        # No processes survived past r₀ — ameboid-ish cell.
        primary_process_count = 0
        process_angle_mask = np.zeros(n_angles, dtype=bool)

    # ── Soma boundary: r_out at non-process angles, interpolated across ──
    r_soma = r_out.copy()
    if process_angle_mask.any() and not process_angle_mask.all():
        # For each process angle, interpolate between the two nearest
        # non-process angles (circular interpolation).
        non_proc_idx = np.flatnonzero(~process_angle_mask)
        non_proc_vals = r_soma[non_proc_idx]
        # Build a fine grid then interpolate periodically.
        all_idx = np.arange(n_angles)
        # Extend with wrap-around so np.interp handles the seam.
        ext_idx = np.concatenate([non_proc_idx - n_angles,
                                  non_proc_idx,
                                  non_proc_idx + n_angles])
        ext_vals = np.concatenate([non_proc_vals, non_proc_vals, non_proc_vals])
        r_soma_interp = np.interp(all_idx, ext_idx, ext_vals)
        r_soma[process_angle_mask] = r_soma_interp[process_angle_mask]

    # ── Rasterize the soma polygon to a binary mask ──────────────────
    poly_y = cy_f + r_soma * np.sin(angles)
    poly_x = cx_f + r_soma * np.cos(angles)
    soma_mask = np.zeros_like(filled, dtype=bool)
    rr, cc = draw_polygon(poly_y, poly_x, shape=filled.shape)
    soma_mask[rr, cc] = True
    # Intersect with the cell mask so the soma never extends past the cell.
    soma_mask &= filled

    # ── Soma scalar features ─────────────────────────────────────────
    soma_area = float(soma_mask.sum())
    cell_area = float(filled.sum())
    soma_to_cell = soma_area / cell_area if cell_area > 0 else 0.0
    soma_perimeter = _mask_perimeter(soma_mask)
    soma_circ = (4.0 * np.pi * soma_area / (soma_perimeter ** 2)
                 if soma_perimeter > 0 else 0.0)

    # ── Sholl profile (intersections vs r), starting at r₀ ───────────
    sholl_radii = np.arange(r0, float(r_out.max()) + sholl_step, sholl_step)
    sholl_intersections = np.array([
        len(_circular_runs(r_out > r)) for r in sholl_radii
    ], dtype=int)
    max_intersections = int(sholl_intersections.max()) if sholl_intersections.size else 0
    ramification = (max_intersections / primary_process_count
                    if primary_process_count > 0 else 0.0)

    polygon_yx = np.column_stack([poly_y, poly_x])

    return RadialResult(
        soma_mask=soma_mask,
        soma_polygon=polygon_yx,
        center_yx=(cy_f, cx_f),
        soma_area=soma_area,
        soma_perimeter=soma_perimeter,
        soma_circularity=soma_circ,
        soma_to_cell_area_ratio=soma_to_cell,
        primary_process_count=primary_process_count,
        critical_radius=r0,
        max_process_extent=float(r_out.max()),
        max_intersections=max_intersections,
        ramification_index=ramification,
        r_out=r_out,
        angles_deg=angles_deg,
        process_angle_mask=process_angle_mask,
        sholl_radii=sholl_radii,
        sholl_intersections=sholl_intersections,
    )


# ── Internal helpers ────────────────────────────────────────────────


def _circular_runs(b: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) index pairs for True runs in a
    circular boolean array. Indices are inclusive; a run that wraps the
    seam is reported as a single run with start > end."""
    n = len(b)
    if not b.any():
        return []
    if b.all():
        return [(0, n - 1)]
    # Find rising/falling edges on the doubled array, then collapse.
    diff = np.diff(b.astype(np.int8))
    starts = list(np.flatnonzero(diff == 1) + 1)
    ends = list(np.flatnonzero(diff == -1))
    if b[0]:
        starts = [0] + starts
    if b[-1]:
        ends = ends + [n - 1]
    # Merge wrap-around: if first run starts at 0 and last ends at n-1,
    # treat them as one circular run.
    if b[0] and b[-1] and starts[0] == 0 and ends[-1] == n - 1 and len(starts) > 1:
        starts_merged = starts[1:]
        ends_merged = ends[:-1]
        # The wrap run: from starts[-1] (last start) around to ends[0].
        wrap = (starts[-1], ends[0])
        runs = list(zip(starts_merged[:-1], ends_merged))
        runs.append(wrap)
        return runs
    return list(zip(starts, ends))


def _largest_circular_gap(on_ring: np.ndarray) -> float:
    """Largest contiguous False run in a circular boolean array, in
    array-index units (= angular bins)."""
    runs = _circular_runs(~on_ring)
    if not runs:
        return 0.0
    n = len(on_ring)
    sizes = []
    for s, e in runs:
        if s <= e:
            sizes.append(e - s + 1)
        else:
            sizes.append((n - s) + (e + 1))
    return float(max(sizes))


def dapi_centroid(
    dapi_image: np.ndarray, cell_mask: np.ndarray | None = None,
    *, min_nucleus_area: int = 20,
) -> tuple[float, float] | None:
    """Centroid (y, x) of the brightest connected nucleus in ``dapi_image``.

    The expected input is a **binary** mask, produced by a global Otsu
    threshold on the full DAPI image at segment time (see
    :func:`glia.segment.extract_single_cells`). We take the largest
    connected component that overlaps ``cell_mask`` and return its
    centroid.

    For backwards compatibility we also accept a grayscale crop — if
    the input has more than two unique values, an Otsu threshold is
    applied here. That path is the per-crop variant and is noise-prone
    on faint cells; prefer the segment-time global threshold.

    ``min_nucleus_area`` rejects single-pixel noise blobs. Defaults to
    20 px², which is well below any real nucleus at 20× or 60×.

    Returns ``None`` if no real nucleus survives — caller falls back to
    the EDT peak. This is also the "no DAPI inside this cell" signal,
    which is biologically meaningful (debris vs cell).
    """
    from skimage.measure import label, regionprops

    img = np.asarray(dapi_image)
    if img.ndim != 2 or img.size == 0:
        return None

    # Detect whether input is already binary. uint8 with values {0, 255}
    # or bool both qualify. If grayscale, fall back to per-crop Otsu.
    uniq = np.unique(img)
    if img.dtype == bool or len(uniq) <= 2:
        bw = img > 0
    else:
        from skimage.filters import threshold_otsu
        try:
            t = float(threshold_otsu(img))
        except Exception:
            return None
        bw = img > t

    if cell_mask is not None:
        bw = bw & np.asarray(cell_mask).astype(bool)
    if not bw.any():
        return None

    labels = label(bw, connectivity=2)
    regions = [r for r in regionprops(labels)
               if r.area >= min_nucleus_area]
    if not regions:
        return None
    biggest = max(regions, key=lambda r: r.area)
    cy, cx = biggest.centroid
    return float(cy), float(cx)


def _mask_perimeter(mask: np.ndarray) -> float:
    """4-connected boundary perimeter of a binary mask, in pixels."""
    if not mask.any():
        return 0.0
    # Count edges between foreground and background / image border.
    p = 0
    p += int((mask[:, :-1] != mask[:, 1:]).sum())  # vertical edges
    p += int((mask[:-1, :] != mask[1:, :]).sum())  # horizontal edges
    p += int(mask[:, 0].sum() + mask[:, -1].sum()
             + mask[0, :].sum() + mask[-1, :].sum())  # image border
    return float(p)
