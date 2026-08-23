#!/usr/bin/env python3
"""
eclipse_stabilize.py - lock the Sun in the centre of a partial-eclipse timelapse.

Why not a normal video stabilizer?
  A featureless disc on black has no trackable texture, and during an eclipse the
  *brightness centroid* of the crescent walks away from the true solar centre.
  This script instead fits a circle of known radius to the SOLAR LIMB only, using
  a gradient-oriented Hough vote, so the Moon's edge cannot pull the fit.

Usage:
  # Pass 1 - measure the Sun centre in every frame -> track.json (+ preview overlay)
  python eclipse_stabilize.py analyze input.avi --track track.json --preview check.mp4

  # (optional) inspect check.mp4 / edit track.json

  # Pass 2 - render the stabilized result
  python eclipse_stabilize.py render input.avi --track track.json -o stabilized.mov \
      --crop 1200

  # Pass 3 - measure what is left, then re-cut if the tail is ruined
  python eclipse_stabilize.py verify stabilized.mov
  python eclipse_stabilize.py render input.avi --track track.json -o cut.mov \
      --crop 1200 --end 2707

Requires: python3, opencv-python, numpy, and ffmpeg on PATH.
  pip install -r requirements.txt
"""

import argparse
import json
import subprocess
import sys

import cv2
import numpy as np


# ------------------------------------------------------------------ debayer --

# WARNING: these keys are OpenCV code names, NOT sensor patterns. OpenCV names its
# Bayer constants after the 2x2 block at row 1, col 1 - one pixel diagonally in from
# the pattern the manufacturer quotes. Translate before passing:
#
#     sensor RGGB -> --bayer bggr        sensor GRBG -> --bayer gbrg
#     sensor BGGR -> --bayer rggb        sensor GBRG -> --bayer grbg
#
# Sanity check on the preview: a blue or magenta disc means R/B are swapped (use the
# partner of the pair); a fine checkerboard or maze texture means the phase is off by
# one pixel (switch pairs entirely).
_BAYER = {
    "rggb": getattr(cv2, "COLOR_BayerRG2BGR_EA", cv2.COLOR_BayerRG2BGR),
    "bggr": getattr(cv2, "COLOR_BayerBG2BGR_EA", cv2.COLOR_BayerBG2BGR),
    "grbg": getattr(cv2, "COLOR_BayerGR2BGR_EA", cv2.COLOR_BayerGR2BGR),
    "gbrg": getattr(cv2, "COLOR_BayerGB2BGR_EA", cv2.COLOR_BayerGB2BGR),
}


def demosaic(frame, pattern):
    """Debayer a raw frame. Must happen BEFORE any geometric transform."""
    if not pattern or pattern == "none":
        return frame
    if frame.ndim == 3:
        # OpenCV often expands a mono/raw stream into 3 identical channels.
        s = frame[::16, ::16]
        if np.array_equal(s[..., 0], s[..., 1]) and np.array_equal(s[..., 1], s[..., 2]):
            frame = frame[..., 0]
        else:
            return frame  # genuine colour already, nothing to do
    return cv2.cvtColor(np.ascontiguousarray(frame), _BAYER[pattern])


# ----------------------------------------------------------------- detection --

def _sobel_scale(ksize=5):
    """cv2.Sobel is unnormalised; find its gain on a unit-slope ramp."""
    ramp = np.tile(np.arange(32, dtype=np.float32), (32, 1))
    return float(cv2.Sobel(ramp, cv2.CV_32F, 1, 0, ksize=ksize)[16, 16])


_SOBEL_SCALE = _sobel_scale()


def half_max_threshold(blur):
    """Threshold at half the disc's own brightness, so auto-exposure and haze
    changes don't move the apparent limb from frame to frame."""
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hi, lo = blur[blur > t], blur[blur <= t]
    if hi.size < 50 or lo.size < 50:
        return float(t)
    sky, disc = float(np.median(lo)), float(np.median(hi))
    return sky + 0.5 * (disc - sky)


def edge_points(gray, thresh=None):
    """Boundary pixels of the bright region, with outward-pointing normals."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if thresh is None:
        thresh = half_max_threshold(blur)
    mask = (blur > thresh).astype(np.uint8)
    if mask.sum() < 50:
        return None, None

    # Sobel gradients: point from bright to dark (i.e. outward on the solar limb)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)

    # Boundary = mask minus its erosion
    edge = mask - cv2.erode(mask, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(edge)
    if len(xs) < 30:
        return None, None

    # Subsample deterministically - a random subset differs frame to frame and
    # injects its own jitter into the fit.
    if len(xs) > 6000:
        step = len(xs) // 6000 + 1
        xs, ys = xs[::step], ys[::step]

    nx, ny = -gx[ys, xs], -gy[ys, xs]          # bright -> dark
    norm = np.hypot(nx, ny) + 1e-6
    n = np.stack([nx / norm, ny / norm], 1)

    # Sub-pixel: slide each boundary pixel along its normal to where the intensity
    # actually crosses the threshold. Integer-snapped limb points are the single
    # biggest source of residual jitter once the big shakes are gone.
    p = np.stack([xs, ys], 1).astype(np.float64)
    inten = blur.astype(np.float64)[ys, xs]
    grad = norm / _SOBEL_SCALE                  # intensity units per pixel
    shift = np.clip((inten - thresh) / np.maximum(grad, 1e-3), -2.0, 2.0)
    return p + n * shift[:, None], n


def vote_centre(pts, normals, radius, shape, coarse=2.0):
    """Each limb pixel votes for a centre one radius back along its inward normal.
    Moon-limb pixels vote incoherently, so the only sharp peak is the Sun."""
    votes = pts - normals * radius
    h, w = shape
    acc = np.zeros((int(h / coarse) + 2, int(w / coarse) + 2), np.float32)
    vx = np.clip((votes[:, 0] / coarse).astype(int), 0, acc.shape[1] - 1)
    vy = np.clip((votes[:, 1] / coarse).astype(int), 0, acc.shape[0] - 1)
    np.add.at(acc, (vy, vx), 1.0)
    acc = cv2.GaussianBlur(acc, (0, 0), 1.5)
    py, px = np.unravel_index(np.argmax(acc), acc.shape)
    return np.array([px * coarse, py * coarse], np.float64), float(acc[py, px])


def kasa(p):
    """Algebraic circle fit with the radius free. Only used to measure R once."""
    A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    b = p[:, 0] ** 2 + p[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = np.array([sol[0], sol[1]], float)
    return c, float(np.sqrt(max(sol[2] + c @ c, 1.0)))


def refine(pts, centre, radius, tol=3.0, iters=8):
    """Circle fit with the radius LOCKED to the known solar radius.

    Letting r float is what makes deep partial phases wobble: the visible arc is
    short, so centre and radius trade off against each other and the fitted centre
    slides along the arc's axis. With r fixed the only free parameter is position.
    """
    c = np.asarray(centre, float).copy()
    R = float(radius)
    inl = np.ones(len(pts), bool)
    for _ in range(iters):
        v = pts - c
        d = np.hypot(v[:, 0], v[:, 1]) + 1e-9
        inl = np.abs(d - R) < tol
        if inl.sum() < 20:
            break
        u = v[inl] / d[inl, None]              # unit vectors centre -> limb point
        step, *_ = np.linalg.lstsq(u, d[inl] - R, rcond=None)
        c += step
        if np.hypot(step[0], step[1]) < 1e-4:
            break
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    rms = float(np.sqrt((((d - R)[inl]) ** 2).mean())) if inl.sum() else 99.0
    return c[0], c[1], R, int(inl.sum()), rms


def scan_radius(pts, nrm, shape, rmin=8, rmax=None):
    """Find R by scanning: the oriented vote only peaks sharply when the trial
    radius matches the true one. Works even if the disc is never fully visible,
    unlike sqrt(area/pi), which assumes an uneclipsed frame exists."""
    h, w = shape
    rmax = rmax or int(min(h, w) * 0.55)
    rs = np.arange(rmin, rmax, 1.0)
    scores = np.array([vote_centre(pts, nrm, r, shape)[1] for r in rs])
    k = int(np.argmax(scores))
    # parabolic interpolation around the peak for sub-pixel radius
    if 0 < k < len(rs) - 1:
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            return float(rs[k] + 0.5 * (y0 - y2) / denom)
    return float(rs[k])


def _radius_in_frame(cap, idx, thresh, bayer, rmin=8, rmax=None):
    """Apparent solar radius in ONE frame, plus the fraction of boundary pixels
    that ended up on it. That fraction is the quality flag: when the disc is
    broken up by branches or thick cloud most of the boundary is not solar limb."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, f = cap.read()
    if not ok:
        return None
    f = demosaic(f, bayer)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
    pts, nrm = edge_points(g, thresh)
    if pts is None:
        return None
    r = scan_radius(pts, nrm, g.shape, rmin=rmin, rmax=rmax)
    c, _ = vote_centre(pts, nrm, r, g.shape)
    frac = 0.0
    for _ in range(4):                       # polish with radius free
        d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
        inl = np.abs(d - r) < 4.0
        if inl.sum() < 20:
            break
        frac = float(inl.mean())
        c, r = kasa(pts[inl])
    return float(r), frac


def radius_curve(cap, thresh, bayer, total, n=120):
    """Apparent solar radius for EVERY frame, as a slow curve.

    R still has to be locked inside each frame's fit - letting it float trades
    centre against radius and slides the centre along a crescent's axis. But R is
    not constant across a session. As the Sun drops toward the horizon, extinction
    and haze pull the half-max limb inward: measured on the reference clip it runs
    255.8 px at the start, 260 at deepest eclipse, 228 by frame 2700 - an 11%
    swing.

    Locking the frame-0 value makes the fit BISTABLE late in the sequence. With a
    circle ~15 px too big, hugging the top arc and hugging the bottom arc are both
    stable optima with similar inlier counts and RMS, so neither fit-quality gate
    can tell them apart, and the centre flips by twice the radius error (~30 px)
    from frame to frame. So measure R on a sample of frames, smooth it, and lock
    each frame to its own value.
    """
    idx = np.unique(np.linspace(0, total - 1, min(n, total)).astype(int))
    ks, rs, run = [], [], None
    for i in idx:
        # Only the first sample gets a full scan. That costs one Hough vote per
        # trial radius over the whole plausible range; repeating it 120 times is
        # far slower than the analysis pass itself. R cannot jump between samples,
        # so afterwards search a narrow window around the running estimate.
        got = (_radius_in_frame(cap, i, thresh, bayer) if run is None else
               _radius_in_frame(cap, i, thresh, bayer, max(8, run - 25), run + 25))
        if got is None:
            continue
        r, frac = got
        if frac < 0.25:            # limb mostly missing - cloud, branches, dropout
            continue
        ks.append(i)
        rs.append(r)
        run = float(np.median(rs[-5:]))
    if len(rs) < 3:
        raise SystemExit("Could not measure the solar radius.")
    ks, rs = np.array(ks, float), np.array(rs, float)

    # Robust clean-up. A sample taken through branches reads far off even when the
    # inlier fraction looks acceptable; reject against a rolling median of the
    # samples. Then smooth: the real curve is a slow atmospheric trend, and any
    # per-sample jitter left in R re-enters the centre as jitter.
    if len(rs) >= 7:
        resid = np.abs(rs - rolling_median(rs, 7))
        s = float(np.median(resid)) * 1.4826
        keep = resid < max(3.0, 4 * s)
        if keep.sum() >= 3:
            ks, rs = ks[keep], rs[keep]
        rs = rolling_median(rs, 5)
    return np.interp(np.arange(total), ks, rs)


# ------------------------------------------------------------------ exposure --

# Fractional-radius bins for the limb-darkening profile. Comparing frames at
# matched rho is what makes this work during deep partial phases: only the outer
# limb is still visible then, and the limb is intrinsically darker than disc
# centre, so comparing a crescent against a whole-disc average would over-brighten
# it badly.
_RHO = np.linspace(0.0, 0.99, 34)
_RHO_MID = (_RHO[:-1] + _RHO[1:]) / 2


def _photosphere(bgr, blur, thresh, cx, cy, R, grid, erode_px=8):
    """Visible photosphere pixels as (rho, BGR), stepped back from BOTH limbs.

    The step-back matters far more than it looks. The Moon's edge is not hard at
    this scale - pixels within several px of it are partially covered, so they read
    low. At deepest eclipse almost the entire crescent lies within that band, and
    including it collapses the measured level: the first version of this read 1.95
    at frame 1500 against 4.72 at frame 1400, which is physically impossible for a
    monotonic auto-exposure ramp.
    """
    k = 2 * erode_px + 1
    mask = cv2.erode((blur > thresh).astype(np.uint8), np.ones((k, k), np.uint8))
    st, X, Y = grid
    rho = np.hypot(X - cx, Y - cy) / R
    vis = (mask[::st, ::st] > 0) & (rho < 0.99)
    return rho[vis], bgr[::st, ::st][vis].astype(np.float64)


def _profile(rho, vals, black):
    """Median BGR against fractional radius - the limb-darkening curve."""
    prof = np.full((len(_RHO_MID), 3), np.nan)
    for j, (a, b) in enumerate(zip(_RHO[:-1], _RHO[1:])):
        m = (rho >= a) & (rho < b)
        if m.sum() >= 40:
            prof[j] = np.median(vals[m], axis=0) - black
    return prof


def _level(rho, vals, black, ref):
    """How much brighter this frame's photosphere reads than the reference, per
    channel, comparing like with like at every fractional radius.

    Deliberately counts SATURATED pixels. Dividing by the true exposure ratio is
    wrong once the sensor clips: at deepest eclipse the Seestar's auto-exposure has
    ramped ~12.9x, but the crescent is pinned at 255 (99.7% of it in R, 93% in G),
    so dividing by 12.9 would render it near black. Matching the level actually
    observed degrades gracefully - it is the exact photometric correction while
    there is headroom (agrees with a clipped-pixels-excluded estimate to 1-2%), and
    it lands a blown-out crescent at the right average brightness once there is
    not. Detail inside the clipped region is gone either way; it was never captured.
    """
    out = np.full(3, np.nan)
    for c in range(3):
        ok = np.isfinite(ref[:, c]) & (ref[:, c] > 8)
        if ok.sum() < 3:
            continue
        pred = np.interp(rho, _RHO_MID[ok], ref[ok, c], left=np.nan, right=np.nan)
        q = (vals[:, c] - black) / np.maximum(pred, 1e-6)
        q = q[np.isfinite(q) & (pred > 8)]
        if q.size >= 200:
            out[c] = float(np.median(q))
    return out


def exposure_reference(cap, thresh, bayer, radii, total, grid, n=24):
    """Pick the least-eclipsed sampled frame and take its limb-darkening profile
    and black level as the target every other frame is matched to.

    Least-eclipsed, not frame 0, so a clip that opens at maximum still gets a
    reference with the full radial range covered - a crescent-only reference has no
    data below rho ~ 0.9 to compare the early frames against.
    """
    best = None
    for i in np.unique(np.linspace(0, total - 1, min(n, total)).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        f = demosaic(f, bayer)
        bgr = f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(g, (5, 5), 0)
        th = half_max_threshold(blur) if thresh is None else float(thresh)
        pts, nrm = edge_points(g, th)
        if pts is None:
            continue
        R = float(radii[min(int(i), len(radii) - 1)])
        c, _ = vote_centre(pts, nrm, R, g.shape)
        cx, cy, _, inl, _r = refine(pts, c, R)
        area = float((blur > th).sum())
        if best is None or area > best[0]:
            best = (area, int(i), bgr, blur, th, cx, cy, R, g)
    if best is None:
        return None, 0.0, -1
    _, idx, bgr, blur, th, cx, cy, R, g = best
    black = float(np.percentile(g, 1.0))
    rho, vals = _photosphere(bgr, blur, th, cx, cy, R, grid)
    if rho.size < 200:
        return None, 0.0, -1
    return _profile(rho, vals, black), black, idx


# --------------------------------------------------------------------- codecs --

# QuickTime plays none of FFV1. It is an FFmpeg-native codec and Apple never
# shipped a decoder, so an .mov wrapping one opens to a black window or an error -
# and FFV1 in MOV is an odd pairing anyway; it belongs in MKV. For something that
# both plays natively on macOS and holds up as an intermediate, use ProRes 4444:
# 4:4:4 so there is no chroma subsampling to band the low-contrast limb, and
# 10-bit so there is headroom the 8-bit source cannot even fill. It is not
# mathematically lossless the way FFV1 is - use FFV1 in .mkv when that matters.
_CODECS = {                     # name: (encoder args, pixel format, containers)
    "h264":   (["-c:v", "libx264", "-preset", "slow"], "yuv420p",
               (".mp4", ".mov", ".mkv")),
    "prores": (["-c:v", "prores_ks", "-profile:v", "4444"], "yuv444p10le",
               (".mov", ".mkv")),
    "ffv1":   (["-c:v", "ffv1", "-level", "3"], "gbrp", (".mkv",)),
}


# -------------------------------------------------------------------- passes --

def analyze(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    radii = radius_curve(cap, args.threshold, args.bayer, total, args.radius_samples)
    # radius_curve seeks; AVI seeks are often inexact, so reopen rather than
    # trusting a rewind. If the analysis loop starts at frame k instead of 0 the
    # whole track is offset against the video and the render is worse than useless.
    cap.release()
    cap = cv2.VideoCapture(args.input)
    print(f"{w}x{h} @ {fps:.3f} fps, {total} frames")
    print(f"solar radius = {radii.min():.2f} .. {radii.max():.2f} px "
          f"({radii[0]:.2f} at frame 0, {radii[-1]:.2f} at the end)")

    # Photometry grid, built once: rho is needed for every frame and the pixel
    # coordinates never move. Subsampled - these are medians over 10^4-10^5 points,
    # and every 4th pixel is far more than enough.
    st = 4
    gy, gx = np.mgrid[0:h:st, 0:w:st]
    grid = (st, gx.astype(np.float32), gy.astype(np.float32))
    expo_ref, black, ridx = exposure_reference(cap, args.threshold, args.bayer,
                                               radii, total, grid)
    cap.release()
    cap = cv2.VideoCapture(args.input)
    if expo_ref is None:
        print("  no exposure reference found; --normalize will be unavailable")
    else:
        print(f"exposure reference = frame {ridx}, black level {black:.1f}")

    writer = None
    if args.preview:
        writer = cv2.VideoWriter(args.preview, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))

    track, levels, n = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = demosaic(frame, args.bayer)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        # Threshold once here rather than inside edge_points, so the photometry
        # below can reuse it - recomputing it is 10 ms, the rest of this is ~5.
        blur = cv2.GaussianBlur(g, (5, 5), 0)
        th = half_max_threshold(blur) if args.threshold is None else float(args.threshold)
        pts, nrm = edge_points(g, th)
        radius = float(radii[min(n, len(radii) - 1)])
        if pts is None:
            track.append(None)
        else:
            c, score = vote_centre(pts, nrm, radius, g.shape)
            cx, cy, _, inl, rms = refine(pts, c, radius)
            track.append([cx, cy, inl, rms] if inl >= args.min_inliers else None)
        if expo_ref is None or track[-1] is None:
            levels.append([None, None, None])
        else:
            bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            rho, vals = _photosphere(bgr, blur, th, track[-1][0], track[-1][1],
                                     radius, grid)
            lv = _level(rho, vals, black, expo_ref)
            levels.append([None if not np.isfinite(x) else float(x) for x in lv])
        if writer is not None:
            vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if track[-1]:
                cv2.circle(vis, (int(track[-1][0]), int(track[-1][1])),
                           int(radius), (0, 0, 255), 1)
                cv2.drawMarker(vis, (int(track[-1][0]), int(track[-1][1])),
                               (0, 255, 0), cv2.MARKER_CROSS, 25, 1)
            writer.write(vis)
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{total}", end="\r", flush=True)
    cap.release()
    if writer is not None:
        writer.release()

    found = sum(1 for t in track if t)
    print(f"\ndetected in {found}/{len(track)} frames "
          f"({100 * found / max(1, len(track)):.1f}%)")
    if found < 0.9 * len(track):
        print("  -> try adjusting --threshold (look at the preview video)")

    if expo_ref is not None:
        lv = np.array([[np.nan if x is None else x for x in e] for e in levels])
        fin = np.isfinite(lv[:, 1])
        if fin.any():
            print(f"exposure level = {np.nanmin(lv[fin, 1]):.2f} .. "
                  f"{np.nanmax(lv[fin, 1]):.2f}x reference (green channel), "
                  f"measured in {int(fin.sum())}/{len(lv)} frames")

    json.dump({"width": w, "height": h, "fps": fps,
               "radius": float(np.median(radii)), "radii": radii.tolist(),
               "threshold": args.threshold, "bayer": args.bayer,
               "black": black, "exposure": levels, "track": track},
              open(args.track, "w"))
    print(f"wrote {args.track}" + (f" and {args.preview}" if args.preview else ""))


def rolling_median(a, win):
    win = win if win % 2 else win + 1
    pad = np.pad(a, win // 2, "edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, win), axis=-1)


def _fill(a, bad):
    """Linearly interpolate across flagged samples."""
    out = a.copy()
    idx = np.arange(len(a))
    good = ~bad & np.isfinite(a)
    if good.sum() < 2:
        raise SystemExit("Too few good detections to stabilize.")
    out[~good] = np.interp(idx[~good], idx[good], a[good])
    return out


def _curvature_flags(a, k=8.0, floor=0.3, win=151):
    """Flag samples whose second difference is a robust outlier.

    Real mount motion is smooth in position, however fast it is, so its curvature
    stays small. A single mis-fit frame produces a large one-frame kink. Judging by
    distance from a rolling median instead fails here: when the mount is really
    moving, the median lags and the tolerance inflates until it hides the spikes.

    The scale is measured LOCALLY, over `win` frames. This mount does not shake
    uniformly: it is quiet for long stretches and then lurches, and phase
    correlation on the source confirms excursions of 50-230 px in a single frame
    are real image motion, not bad fits. A single global MAD is set by the quiet
    stretches, so a violent one reads as a long spike train and gets interpolated
    away - which is exactly what left the Sun 300 px off centre in the render. A
    rolling MAD stays tight where the mount is quiet and loosens where it is not.
    """
    n = len(a)
    d2 = np.zeros(n)
    d2[1:-1] = a[2:] - 2 * a[1:-1] + a[:-2]
    core = d2[1:-1]        # the two endpoints are padding, not real curvature -
                           # leaving them in drags the rolling median to zero at
                           # both ends and flags the whole first and last window
    win = max(3, min(win, len(core) if len(core) % 2 else len(core) - 1))
    med = rolling_median(core, win)
    mad = rolling_median(np.abs(core - med), win) * 1.4826
    flag = np.zeros(n, bool)
    flag[1:-1] = np.abs(core - med) > k * np.maximum(mad, floor)
    return flag


def clean_track(track, despike=9, max_jump=0.0, smooth=0):
    """Reject bad fits and interpolate across them. Never smooths the good data."""
    n = len(track)
    xs = np.array([t[0] if t else np.nan for t in track], float)
    ys = np.array([t[1] if t else np.nan for t in track], float)
    inl = np.array([t[2] if t and len(t) > 2 else np.nan for t in track], float)
    rms = np.array([t[3] if t and len(t) > 3 else np.nan for t in track], float)

    bad = ~np.isfinite(xs)
    n_missing = int(bad.sum())

    # Fit-quality gates: a garbage fit usually has few inliers or a poor residual.
    # `weak` is the softer version of the same test - see the geometric gate below.
    weak = np.zeros(n, bool)
    if np.isfinite(inl).any():
        med_inl = float(np.nanmedian(inl))
        bad |= inl < 0.35 * med_inl
        weak |= inl < 0.60 * med_inl
    if np.isfinite(rms).any():
        m = float(np.nanmedian(rms))
        s = float(np.nanmedian(np.abs(rms - m))) * 1.4826
        bad |= rms > m + 6 * max(s, 0.05)
        weak |= rms > m + 3 * max(s, 0.05)
    weak |= bad | ~np.isfinite(xs)
    n_quality = int(bad.sum()) - n_missing

    # Geometric gates, iterated so neighbours of a spike get caught too.
    if despike:
        for _ in range(3):
            fx, fy = _fill(xs, bad), _fill(ys, bad)
            # Curvature alone cannot tell a real lurch from a mis-fit, so it only
            # gets to condemn a frame whose fit is ALSO below par. This mount
            # really does jump 229 px in one frame - phase correlation on the
            # source confirms it, and those frames fit beautifully (inliers and
            # RMS both normal). Rejecting them on curvature interpolated straight
            # through the lurch and left the Sun 300 px off centre: the single
            # worst shake in the output was produced by the despiker, not by the
            # mount. A frame that fits well is data, not a spike.
            flag = (_curvature_flags(fx) | _curvature_flags(fy)) & weak
            if max_jump:
                step = np.hypot(np.gradient(fx), np.gradient(fy))
                flag |= step > max_jump
            if not (flag & ~bad).any():
                break
            bad |= flag
            if bad.mean() > 0.30:
                print("  >30% flagged - gates too aggressive, keeping raw track")
                bad = ~np.isfinite(xs)
                break

    xs, ys = _fill(xs, bad), _fill(ys, bad)
    step = np.hypot(np.diff(xs), np.diff(ys))
    print(f"track: {n} frames, motion {np.median(step):.2f} px/frame median, "
          f"{step.max():.2f} max")
    print(f"  rejected {int(bad.sum())} ({100 * bad.mean():.1f}%): "
          f"{n_missing} undetected, {n_quality} poor fit, "
          f"{int(bad.sum()) - n_missing - n_quality} spikes")

    if smooth and smooth > 1:      # only for deliberate slow drift, not locking
        w = smooth if smooth % 2 else smooth + 1
        k = np.ones(w) / w
        xs = np.convolve(np.pad(xs, w // 2, "edge"), k, "valid")
        ys = np.convolve(np.pad(ys, w // 2, "edge"), k, "valid")
    return xs, ys, bad


def render(args):
    meta = json.load(open(args.track))
    W, H, fps = meta["width"], meta["height"], meta["fps"]
    xs, ys, bad = clean_track(meta["track"], args.despike, args.max_jump, args.smooth)

    gains, black = None, float(meta.get("black", 0.0))
    if args.normalize:
        lv = np.array([[np.nan if x is None else x for x in e]
                       for e in meta.get("exposure") or []], float)
        if lv.shape[0] != len(xs):
            raise SystemExit("--normalize needs an exposure record from analyze; "
                             "re-run analyze to add one to the track file.")
        # A frame whose fit was rejected has an unreliable mask too - the
        # photosphere is selected using that frame's own centre and radius - so
        # drop the same frames here and interpolate.
        for c in range(3):
            lv[:, c] = _fill(lv[:, c], bad | ~np.isfinite(lv[:, c]))
        # No temporal smoothing. It is tempting - the auto-exposure holds a level
        # then steps, so a rolling median looks like a free denoise - but measured
        # on the real clip the per-frame estimate is already stable to 0.17%
        # frame to frame, and a median-of-5 left its stdev unchanged. It is not
        # free either: it assumes exposure is piecewise-smooth, and on a source
        # that really does vary frame to frame it destroys the correction outright
        # (the synthetic clip, which jitters exposure every frame, came out no
        # flatter than uncorrected). Measure it per frame and use it per frame.
        gains = 1.0 / np.maximum(lv, 1e-6)
        print(f"  normalize: gain {gains.min():.2f}..{gains.max():.2f} "
              f"(per channel, black level {black:.1f})")

    # Trim is a render-time decision, so the track still covers the whole clip and
    # you can re-cut without re-running analyze. The gates above also still see the
    # whole track, which keeps their medians the same as the ones `verify` reports.
    start = max(0, args.start)
    end = len(xs) if args.end is None else min(int(args.end), len(xs))
    if end <= start:
        raise SystemExit(f"--start {start} --end {end}: no frames left to render.")
    if start or end < len(xs):
        print(f"  rendering frames {start}-{end - 1}, "
              f"trimming {len(xs) - (end - start)} of {len(xs)}")
    speed = max(1, int(args.speed))
    if speed > 1:
        n_out = -(-(end - start) // speed)
        print(f"  speed {speed}x ({args.speed_mode}): {end - start} -> {n_out} "
              f"frames, {n_out / fps:.1f}s")

    if args.crop:
        ow = oh = int(args.crop)
    else:
        ow, oh = W, H
    tx, ty = ow / 2.0, oh / 2.0

    codec = args.codec
    enc, pix, containers = _CODECS[codec]
    if not args.output.lower().endswith(containers):
        print(f"  warning: {codec} does not belong in "
              f"{args.output.rsplit('.', 1)[-1]}; prefer {containers[0]}")
    # -pix_fmt has to be chosen PER CODEC. Setting yuv420p for everything, as this
    # used to, silently made the "lossless" path lossy: it converted RGB to 8-bit
    # 4:2:0 and threw away three quarters of the chroma BEFORE handing the frames
    # to FFV1, so the archive was a bit-exact copy of already-damaged data. Measured
    # against the frames the pipeline actually produced: max error 22/255, 85% of
    # pixels wrong. Exactly the banding the lossless intermediate exists to avoid.
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{ow}x{oh}", "-r", f"{fps}", "-i", "-", "-an"]
    cmd += enc + ["-pix_fmt", pix]
    if codec == "h264":
        cmd += ["-crf", str(args.crf)]
    cmd.append(args.output)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    bayer = args.bayer or meta.get("bayer")
    cap = cv2.VideoCapture(args.input)
    i = written = 0
    acc, n_acc = None, 0

    def emit(img):
        nonlocal written
        proc.stdin.write(img.tobytes())
        written += 1
        if written % 200 == 0:
            print(f"  {written}", end="\r", flush=True)

    while i < end:
        if i < start:
            # Skip by grabbing, never by seeking: the track is indexed by source
            # frame number, and one silently inexact seek offsets the whole thing.
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        frame = demosaic(frame, bayer)   # debayer FIRST, then warp
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if gains is not None:
            # Before the warp, not after: warpAffine pads with real zeros, and
            # scaling those afterwards would lift the border to black*(1-gain) and
            # ring the frame in grey.
            frame = np.clip((frame.astype(np.float32) - black) * gains[i] + black,
                            0, 255).astype(np.uint8)
        M = np.float32([[1, 0, tx - xs[i]], [0, 1, ty - ys[i]]])
        out = cv2.warpAffine(frame, M, (ow, oh), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if speed == 1:
            emit(out)
        elif args.speed_mode == "drop":
            if (i - start) % speed == 0:
                emit(out)
        else:
            # Stack AFTER the warp - averaging is only meaningful once the frames
            # are aligned, which is the whole point of having locked the Sun first.
            acc = out.astype(np.float32) if acc is None else acc + out
            n_acc += 1
            if n_acc == speed:
                emit((acc / n_acc).astype(np.uint8))
                acc, n_acc = None, 0
        i += 1
    if acc is not None and n_acc:
        emit((acc / n_acc).astype(np.uint8))       # partial group at the tail
    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"\nwrote {args.output} ({written} frames, "
          f"{written / fps:.1f}s at {fps:g} fps)")


def verify(args):
    """Re-detect the Sun in a rendered file and report how much it still moves.

    This re-detects with the same code as `analyze`, so it needs the same
    per-frame radius: measuring one radius for the whole clip makes the fit
    bistable late on and verify then reports its own flip-flopping as spikes.
    Spike numbers below are real frame numbers, not indices into the frames that
    happened to yield a detection.
    """
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    radii = radius_curve(cap, None, args.bayer, total)
    cap.release()
    cap = cv2.VideoCapture(args.input)
    cs, at, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = demosaic(frame, args.bayer)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        pts, nrm = edge_points(g)
        radius = float(radii[min(i, len(radii) - 1)])
        if pts is not None:
            c, _ = vote_centre(pts, nrm, radius, g.shape)
            cx, cy, _, inl, _r = refine(pts, c, radius)
            if inl >= 40:
                cs.append((cx, cy))
                at.append(i)
        i += 1
    cap.release()
    if len(cs) < 10:
        raise SystemExit("Too few detections to verify.")
    a = np.array(cs)
    at = np.array(at)
    d = a - np.median(a, axis=0)
    dist = np.hypot(d[:, 0], d[:, 1])
    step = np.hypot(*np.diff(a, axis=0).T)
    adjacent = np.diff(at) == 1          # only compare genuinely consecutive frames
    print(f"frames measured  : {len(a)} of {i}")
    print(f"radius           : {radii.min():.2f} .. {radii.max():.2f} px")
    print(f"drift about median: {np.median(dist):.2f} px median, "
          f"p99 {np.percentile(dist, 99):.2f}, peak {dist.max():.2f}")
    print(f"frame-to-frame   : {np.median(step):.2f} px median, "
          f"p99 {np.percentile(step, 99):.2f}, max {step.max():.2f}")
    spikes = at[1:][adjacent & (step > max(1.0, 20 * np.median(step)))]
    if len(spikes):
        print(f"spike frames ({len(spikes)}): "
              f"{', '.join(str(int(f)) for f in spikes[:15])}"
              f"{' ...' if len(spikes) > 15 else ''}")
    if np.median(step) < 0.3 and not len(spikes):
        print("-> locked.")
    elif np.median(step) < 0.3:
        print(f"-> locked apart from {len(spikes)} bad frames; these are the flicker. "
              f"Inspect those frames before touching the gates - if the Sun is "
              f"obscured there (cloud, branches) no alignment fix applies.")
    else:
        print("-> still moving; track is probably misaligned against the video.")


# --------------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("input")
    a.add_argument("--track", default="track.json")
    a.add_argument("--preview", default=None,
                   help="write an overlay video to visually check the fit")
    a.add_argument("--threshold", type=int, default=None,
                   help="fixed 8-bit grey level for the limb. Default: auto "
                        "half-max per frame, which is immune to exposure changes")
    a.add_argument("--min-inliers", type=int, default=40)
    a.add_argument("--radius-samples", type=int, default=120,
                   help="frames sampled to build the apparent-radius curve. The "
                        "radius drifts with altitude/haze; one value for the whole "
                        "clip makes the fit bistable late on")
    a.add_argument("--bayer", choices=["none", "rggb", "bggr", "grbg", "gbrg"],
                   default="none",
                   help="demosaic raw frames. OpenCV code name, NOT the sensor "
                        "pattern: RGGB->bggr, BGGR->rggb, GRBG->gbrg, GBRG->grbg")
    a.set_defaults(func=analyze)

    r = sub.add_parser("render")
    r.add_argument("input")
    r.add_argument("--track", default="track.json")
    r.add_argument("-o", "--output", default="stabilized.mp4")
    r.add_argument("--crop", type=int, default=0,
                   help="output square size in px, e.g. 1200 (0 = keep original)")
    r.add_argument("--start", type=int, default=0,
                   help="first source frame to render (default 0)")
    r.add_argument("--end", type=int, default=None,
                   help="stop BEFORE this source frame, like a Python slice. Use "
                        "it to cut a ruined tail - if the Sun sets behind trees or "
                        "cloud, no alignment fix applies and the frames are best "
                        "dropped. `verify` prints the first bad frame number, so "
                        "--end 2707 drops everything from 2707 on")
    r.add_argument("--despike", type=int, default=1,
                   help="1 = reject bad fits (default), 0 = disable all gating")
    r.add_argument("--max-jump", type=float, default=0.0,
                   help="hard px/frame limit on top of the automatic spike gates. "
                        "0 = auto only. This one is unconditional, so it will cut "
                        "genuine fast motion too - check the fit quality first")
    r.add_argument("--smooth", type=int, default=0,
                   help="LEAVE AT 0 to lock the Sun. Non-zero follows a smoothed "
                        "path, which leaves residual wobble in the output")
    r.add_argument("--speed", type=int, default=1,
                   help="speed the timelapse up by this whole-number factor: 2 "
                        "makes the output half as long. Frame rate is unchanged")
    r.add_argument("--speed-mode", choices=["drop", "stack"], default="drop",
                   help="how --speed discards time. drop (default) keeps the first "
                        "frame of each group. stack averages the group instead, "
                        "which is valid here because the Sun is locked to ~0.04 px, "
                        "and cuts noise ~1.5x at 4x - but that gain is invisible at "
                        "8 bit, and anything fixed to the GROUND smears badly: "
                        "foreground trees sweep ~8 px per source frame while the Sun "
                        "is held still, so 4x stacking blurs them by ~31 px. Use "
                        "stack only for a clip with no foreground")
    r.add_argument("--normalize", action="store_true",
                   help="undo the camera's auto-exposure ramp so the photosphere "
                        "holds a constant brightness. Per channel, which also "
                        "restores the colour - R saturates well before G and B, so "
                        "an uncorrected deep partial phase drifts white")
    r.add_argument("--crf", type=int, default=16, help="H.264 quality, lower = better")
    r.add_argument("--codec", choices=["h264", "prores", "ffv1"], default="h264",
                   help="h264 (default) for sharing. prores writes ProRes 4444 in "
                        ".mov - 10-bit 4:4:4, plays natively in QuickTime, and is "
                        "the one to use for an editable master. ffv1 is "
                        "mathematically lossless but QuickTime CANNOT play it; put "
                        "it in .mkv and treat it as an archive, not a viewing copy")
    r.add_argument("--bayer", choices=["none", "rggb", "bggr", "grbg", "gbrg"],
                   default=None, help="override the pattern stored in track.json")
    r.set_defaults(func=render)

    v = sub.add_parser("verify")
    v.add_argument("input", help="a rendered/stabilized video to measure")
    v.add_argument("--bayer", choices=["none", "rggb", "bggr", "grbg", "gbrg"],
                   default="none")
    v.set_defaults(func=verify)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()