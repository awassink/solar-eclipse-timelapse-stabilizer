# Solar eclipse timelapse stabilisation

Locking the Sun to a fixed pixel in a partial-eclipse (~90%) timelapse shot on a
Seestar S30 Pro. Single script: `eclipse_stabilize.py` (stdlib + opencv-python +
numpy, ffmpeg on PATH). Two passes plus a checker.

```
python eclipse_stabilize.py analyze input.avi --bayer gbrg --track track.json [--preview check.mp4]
python eclipse_stabilize.py render  input.avi --track track.json -o stabilized.mov --crop 1200 --codec prores
python eclipse_stabilize.py verify  stabilized.mov
# re-cut a ruined tail; trimming is render-time, so no re-analyze
python eclipse_stabilize.py render  input.avi --track track.json -o cut.mov --crop 1200 --end 2707
# undo the camera's auto-exposure ramp as well
python eclipse_stabilize.py render  input.avi --track track.json -o cut.mov --crop 1200 --end 2707 --normalize
# 2x / 4x faster (same fps, shorter clip)
python eclipse_stabilize.py render  input.avi --track track.json -o fast.mov --crop 1200 --end 2707 --normalize --speed 4
```

## Source data

- ~6 GB raw AVI, 2959 frames.
- Raw Bayer. Sensor pattern is GRBG, but OpenCV's Bayer constants are named for
  the 2x2 block at row 1 col 1, so the flag to pass is `--bayer gbrg`. Mapping:
  sensor RGGB -> `bggr`, BGGR -> `rggb`, GRBG -> `gbrg`, GBRG -> `grbg`.
- Alt-az mount, so the field rotates over the sequence.
- Auto-exposure is ON and cannot be turned off after the fact. It ramps ~12.9x as
  the Moon covers the disc, and the crescent goes into hard saturation from roughly
  frame 1350 to 1900: at frame 1483, 99.7% of the visible photosphere is clipped in
  R, 93% in G, 18% in B. R clips at 255, G at 249. Nothing recovers detail there.
- Channels do not clip together. R saturates first (it is already 8.7% clipped at
  frame 0), so as the exposure climbs the recorded hue drifts from orange to white.

## Approach

Generic stabilisers (Warp Stabilizer, vidstab) fail: a smooth disc on black has no
trackable texture. Brightness-centroid centring also fails: the centroid walks away
from the true centre as the Moon covers the disc. Instead fit a circle to the solar
limb geometrically.

1. Threshold at half-max relative to the disc's own brightness, take boundary
   pixels, localise each sub-pixel along its intensity gradient.
2. Gradient-oriented Hough vote: each limb pixel votes one radius back along its
   inward normal. Moon-limb pixels vote incoherently, so the only sharp peak is the
   Sun. Radius is found by scanning trial radii for the sharpest peak.
3. Gauss-Newton circle fit with the radius LOCKED, solving only for position —
   locked to *this frame's* radius, taken off a smooth curve measured across the
   whole clip (see the radius-drift pitfall below).
4. Reject bad frames, interpolate across them, warp with sub-pixel Lanczos.

## Pitfalls — every one of these was a live bug, and all look like reasonable code

**Do not smooth the track.** Shifting each frame so a *smoothed* path lands at
centre is the camera-stabilisation recipe; it leaves `measured - smoothed` in the
output as visible wobble. Here there is no intentional motion to preserve — shift by
the measured position itself. `--smooth` exists but must stay 0.

**Do not let the radius float.** A free-radius (Kasa) fit trades centre against
radius. On a thin crescent the visible arc is short, so the fitted centre slides
along the arc's axis, worst at deepest eclipse. Measure R, then lock it — but
measure it per frame, not once for the clip. See the next-but-one pitfall.

**Do not estimate R from disc area.** `sqrt(area/pi)` assumes some frame shows an
uneclipsed disc. It read 44.6 px against a true 60 px on eclipsed-only input.
Use the radius scan.

**Lock the radius per frame, not per clip.** R must not float *inside* a frame's
fit, but it is not constant *across* a session. On the reference clip the apparent
half-max radius runs 255.8 px at frame 0, 260 at deepest eclipse, then falls to
216 by the end — an 11% swing, as the Sun drops toward the horizon and extinction
and haze pull the half-max contour inward. Lock the frame-0 value for the whole
clip and the fit goes BISTABLE late on: with a circle ~15 px too big, hugging the
top arc and hugging the bottom arc are both stable optima with comparable inlier
counts and RMS — the wrong one sometimes has *more* inliers — so no fit-quality
gate can separate them and the centre flips by twice the radius error (~30 px)
between neighbouring frames. Seeding the fit from the previous frame does not help;
it just picks whichever basin it started in. `radius_curve` samples ~120 frames,
rejects samples taken through cloud or branches, smooths, and interpolates.
Symptom to watch for: a centre that flips between two values a fixed distance
apart, along the crescent's axis, while the threshold and edge-point count stay
perfectly steady. That steadiness is what rules out the threshold as the cause.

**Do not detect outliers by distance from a rolling median.** With several px/frame
of real motion the median lags, the auto-tolerance inflates to cover the lag, and
229 px spikes sail through. Worse, a fixed tolerance flags nearly every frame and
overwrites good data with the median — silently reintroducing smoothing. Use the
second difference (curvature): real motion is smooth however fast it is, a bad fit
is a one-frame kink. Plus fit-quality gates on inlier count and fit RMS.

**Curvature alone must not be allowed to condemn a frame.** This mount really does
lurch: frames 117-119 and 131-135 move 50-229 px in a single frame, and phase
correlation on the source confirms every one of them as real image motion. Those
frames fit beautifully — inliers and RMS both normal — so a 229 px jump is not, on
its own, evidence of anything. Rejecting them interpolated straight through the
lurch and left the Sun up to 302 px off centre in the render: the single worst
shake in the output was manufactured by the despiker, not by the mount. So the
curvature gate now only fires on frames whose fit is *also* below par
(`flag & weak`), and its MAD is measured over a rolling window rather than the
whole clip. A frame that fits well is data. Note the rolling MAD must skip the
artificial zeros at `d2[0]` and `d2[-1]`, or the median collapses to zero there and
the first and last window get flagged wholesale.

**Phase correlation is the arbiter.** When the track disagrees with your intuition
about whether a frame really moved, cross-correlate the two source frames directly
(`cv2.phaseCorrelate` on a Hann-windowed crop). It is independent of the circle fit,
so it settles "real motion or bad fit?" in one step — and its response value tells
you when to distrust it (it drops below ~0.5 on the dim, low-contrast frames near
maximum eclipse).

**Reopen the capture after seeking.** `radius_curve` seeks; AVI seeks are often
inexact, so `cap.set(POS_FRAMES, 0)` cannot be trusted. If analysis starts at frame
k the whole track is offset against the video and render is worse than the input.
This one is invisible in `--preview`, which is drawn inside the analysis loop and so
stays self-consistent at any offset. Verify against the rendered file, not the
preview.

**Debayer before warping, never after.** Sub-pixel interpolation on a mosaiced frame
destroys the pattern irrecoverably.

## Exposure pitfalls — `--normalize`, and four more ways to get it wrong

**Step back from the Moon's limb before measuring brightness.** The lunar edge is
not hard at this scale; pixels within several px of it are partly covered and read
low. At deepest eclipse nearly the whole crescent sits in that band, so including
it collapses the measured level — the first version read 1.95 at frame 1500 against
4.72 at frame 1400, impossible for a monotonic exposure ramp. Erode the visible
mask by ~8 px. The result is then insensitive to the exact amount (8 px and 14 px
agree to 2%).

**Compare at matched fractional radius.** During deep partial phases only the outer
limb is visible, and the limb is intrinsically darker than disc centre. Comparing a
crescent's average against a whole-disc average therefore over-brightens it badly.
Build a limb-darkening profile I(rho) from the least-eclipsed frame and compare
like with like.

**Match the level you observe, not the true exposure.** Correcting by the real
exposure ratio is right only while the sensor has headroom. Once the crescent is
pinned at 255, dividing by the true ~12.9x renders it near black. Including the
saturated pixels in the estimate makes it degrade gracefully: it is the exact
photometric correction while unclipped (agrees with a clipped-excluded estimate to
1-2%), and lands a blown crescent at the right average brightness once not.

**Correct per channel, not by one scalar.** Because R saturates long before G and
B, a single luminance gain preserves the drifted hue and merely darkens it — the
crescent comes out grey at maximum and green either side of it. Matching each
channel to its own reference profile restores the colour as well as the level.

**Do not smooth the exposure track.** Same trap as `--smooth` on the position
track, one layer down. A rolling median looks free, but the per-frame estimate is
already stable to 0.17% frame-to-frame on the real clip and a median-of-5 left its
stdev unchanged — while on a source whose exposure really does vary per frame it
destroys the correction outright (the synthetic clip came out no flatter than
uncorrected). Measure per frame, apply per frame.

## Speeding it up — why `--speed-mode drop` is the default

Averaging each group of N frames instead of throwing N-1 away looks like the
obvious win: the Sun is locked to ~0.04 px, so a stack is properly registered, and
it is free stacking. Measured, it does work — high-frequency noise on the disc drops
1.30x at 2x and 1.54x at 4x.

It is still the wrong default, for a reason specific to this footage. Stabilising on
the Sun pins the Sun and sets **everything attached to the ground moving**. Measured
by phase correlation on the stabilised output, the foreground treeline sweeps
**~8 px per source frame**, against **~0.2 px per frame for the Moon's limb** — a 40x
difference. So a 4x stack blurs the Moon by an invisible ~0.8 px and the trees by
~31 px, which turns a crisp treeline into mush. Meanwhile the noise it buys back is
sub-LSB at 8 bit and invisible side by side.

So: `drop` by default, `stack` only for a clip with no foreground. If a future clip
is noisy and framed on sky alone, `stack` is the better choice and the code is
already there.

Frame rate is never changed - `--speed` shortens the clip, it does not slow the
playback. Stacking happens AFTER the warp; averaging unregistered frames would just
be motion blur.

## Output codecs — QuickTime will not play FFV1

`--codec ffv1` writes FFV1, which **QuickTime cannot decode**.
It is an FFmpeg-native codec and Apple never shipped a decoder; AVFoundation reports
`isDecodable = false` on the track, so the file opens to an error. FFV1 in a `.mov`
is an odd pairing regardless — it belongs in `.mkv`.

Worse, the `--lossless` flag this replaced was not lossless. `-pix_fmt yuv420p` was
set once for
every codec, before the encoder was chosen, so RGB was converted to 8-bit 4:2:0 and
three quarters of the chroma was thrown away *before* FFV1 saw it. The archive was a
bit-exact copy of already-damaged data. Measured against the frames the pipeline
actually produced, over 60 frames:

| codec | max error | mean | pixels differing | QuickTime |
|---|---|---|---|---|
| `ffv1` (fixed, `gbrp`) | **0** | 0.0000 | 0.0% | no |
| `prores` 4444 (`yuv444p12le`) | 4 | 0.0506 | 5.1% | **yes** |
| `h264` crf 16 | 39 | 1.4906 | 84.5% | yes |
| `ffv1` + `yuv420p` (old) | 22 | 1.4670 | 85.1% | no |

That path was *less* faithful than ProRes 4444 and unplayable as well. `--lossless`
is gone; `--codec` replaces it.

So: `--codec prores` for a viewable, editable master (ProRes 4444, tag `ap4h`,
10-bit 4:4:4 — no chroma subsampling to band the low-contrast limb), `--codec ffv1`
in `.mkv` when bit-exactness genuinely matters, `h264` for sharing. Pixel format is
now chosen per codec; never set one globally again.

## Status

Working. `verify` on the real render (2959 frames, `--crop 800`):

| | before | after |
|---|---|---|
| drift about median | 0.18 px med, peak 301.6 | 0.03 px med, peak 83.0 |
| frame-to-frame | 0.05 px med, max 226.6 | 0.03 px med, max 81.4 |
| spike frames | 459, spread through the clip | 139, **all at frame 2707+** |

Cut to `--end 2707` the same render measures 0.08 px median drift, **peak 2.58 px**,
frame-to-frame max 1.90 px, and 3 spike frames — all of them sub-2 px. Everything
that was still flagged is the Sun setting behind trees, where the limb is chopped
into fragments and no alignment fix applies. Two changes did it: the per-frame
radius curve, and stopping the despiker from rejecting well-fitted frames.

Delivered cut is `--end 2800 --normalize` (2800 frames, 93.3 s): ends on a real
treeline, and no frame in 0-2799 is overridden by the despiker. Plus `--speed 2`
and `--speed 4` versions at 46.7 s and 23.3 s.

With `--normalize` on top, photosphere luminance goes from 98-250 (2.55x spread,
stdev 51.0) to 92-117 (**1.27x, stdev 7.2**), and the hue holds — R 182-234,
G 60-75, B 22-29, against R 208-254, G 55-249, B 27-248 uncorrected. The residual
dip at maximum is real: only the intrinsically darker extreme limb is visible then.

`verify` needs the radius curve too. Run it against a single radius and its own
re-detection goes bistable, so it reports its own flip-flopping as spikes — that was
a large part of the original 459. It re-detects on the rendered file, so it measures
content quality as well as alignment and will always fail on ruined frames. It also
reads a `--normalize`d file worse than the plain one (29 spikes vs 3) purely because
the rescaling moves its half-max contour; the warp is provably identical, so trust
a direct phase correlation between the two renders (0.02 px) over verify there.

## Open items

1. ~~Decide what to do about the tree frames (2707-2958, ~8 s).~~ Done: cut with
   `--end 2707`. `--start`/`--end` slice at render time only, so the track still
   covers the whole clip and you can re-cut without re-running analyze. They skip
   with `cap.grab()` rather than seeking, because the track is indexed by source
   frame number and one inexact seek offsets the lot.
2. Field rotation is unsolved. Once the disc is locked the Moon's bite will swing
   around over the sequence. Needs a rotation solve — either the parallactic angle
   from time and pointing, or tracking a sunspot — not more translation work.
3. Colour: the disc reads orange after debayer. That is white balance, not a debayer
   error, and `--normalize` does not fix it — it pins every frame to the *reference
   frame's* balance, so it removes the drift but keeps the cast. Fix downstream;
   keep the stabilised master at `--codec prores` (or `ffv1` in `.mkv`) so the
   low-contrast limb does not band.
4. `--normalize` gains exceed 1 near the end of the clip (up to 1.44 in G, where the
   low Sun is heavily reddened) and that introduces *new* clipping — at frame 2700,
   R goes from 0.06% of frame clipped to 2.0%. Acceptable here, but if a clip ends
   dimmer than its reference, consider capping the gain or picking a later reference.
5. `--normalize` needs an `exposure` record in the track file. Track files written
   before it existed will refuse with a message telling you to re-run analyze.

## Testing

No real-data fixtures in repo. Validation so far is synthetic: render a disc plus an
occulting disc at 4x and downsample, add per-frame exposure variation and noise, then
compare recovered centres against known truth. Worth keeping — three of the bugs
above were caught this way and only one was caught by eye. Detection currently
scatters ~0.01 px under exposure variation and drifts <0.06 px across a Moon sweep.
