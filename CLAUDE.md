# Working notes — solar eclipse timelapse stabilizer

Locks the Sun to a fixed pixel in a partial-eclipse (~90%) timelapse shot on a
Seestar S30 Pro. One script, `eclipse_stabilize.py` — stdlib + `opencv-python` +
`numpy`, with `ffmpeg` on `PATH` — and three subcommands: `analyze`, `render`,
`verify`.

**`README.md` is the documentation.** Usage, every flag, the `--bayer` mapping,
how the detection works, the reasoning behind each design decision, results and
limitations all live there. When behaviour changes, update `README.md`. Add
something here only if it is guidance for working *on* the code rather than
*with* it — this file is deliberately short, and duplicating the README into it
just gives the two files room to disagree.

The second place to read is the script itself: the non-obvious choices are
explained in docstrings at the point of use (`radius_curve`, `_curvature_flags`,
`clean_track`, `_level`, `verify`). Fix the comment when you change the code.

## Layout

| | |
|---|---|
| `eclipse_stabilize.py` | the entire tool — ~900 lines, no package, no modules |
| `README.md`, `docs/*.jpg` | user documentation; the figures are generated from real output |
| `requirements.txt` | numpy + opencv-python only. Keep it ASCII/LF — it was UTF-16 once and git treated it as binary |

## Reference clip

`~/Dropbox (Personal)/Seestar/Solar_video/2026-08-12-191843-Solar-timelapse-RAW.avi`
— ~6 GB raw Bayer, 1080x1920, 2959 frames, alt-az mount, auto-exposure on.
`analyze` takes ~100 s over it. The delivered render is
`--crop 800 --end 2800 --normalize` (plus `--speed 2` / `--speed 4`).

No track file or rendered output survives in the working tree, so anything that
needs measurements starts with a fresh `analyze`.

## Invariants — do not "clean these up"

Every one of these was a live bug, and each looked like reasonable code. The
README explains the *why* for the ones a user can see; these are the ones that
only bite someone editing the source.

- **The track is indexed by source frame number.** `radius_curve` and
  `exposure_reference` both seek, and AVI seeks are often inexact, so `analyze`
  reopens the capture after each rather than rewinding. `render --start` steps
  over frames with `cap.grab()`, never a seek. An offset track makes the render
  worse than the input — and is *invisible* in `--preview`, which is drawn inside
  the analysis loop and so stays self-consistent at any offset. Check against the
  rendered file.
- **Debayer before warping**, in both `render` and `verify`. Sub-pixel
  interpolation on a mosaiced frame destroys the pattern irrecoverably.
- **Nothing gets smoothed** — not the position track (`--smooth` stays 0), not
  the exposure track, not the radius *inside* a fit. The radius curve across the
  clip is the one deliberate exception.
- **`_curvature_flags` measures its MAD over `d2[1:-1]`.** The two endpoints are
  artificial zeros; leaving them in collapses the rolling median at both ends and
  flags the whole first and last window.
- **Curvature never condemns a frame on its own** — `clean_track` requires
  `flag & weak`, i.e. a below-par fit as well. A well-fitted frame is data, even
  when it jumps 229 px.
- **`verify` must build its own radius curve.** Re-detecting against a single
  radius goes bistable and reports the detector's own flip-flopping as spikes.
- **Pixel format is chosen per codec**, inside the codec branch. Setting one
  globally is what silently made the old lossless path lossy.
- **Boundary subsampling is deterministic** (every *n*-th point). A random subset
  differs frame to frame and injects its own jitter.

## Debugging

`cv2.phaseCorrelate` on a Hann-windowed crop of two source frames is the arbiter:
it is independent of the circle fit, so it settles "real motion or bad fit?" in
one step. Its response value tells you when to distrust it — it drops below ~0.5
on the dim, low-contrast frames near maximum eclipse.

Signature of a bad radius: the centre flips between two values a fixed distance
apart, along the crescent's axis, while the threshold and edge-point count stay
perfectly steady. That steadiness is what rules out the threshold as the cause.

## Testing

No real-data fixtures in the repo; validation is synthetic (see README →
Testing). Worth keeping — most of the invariants above were caught that way and
only one was caught by eye.

## Next work

Field rotation is the only substantial feature still missing: on an alt-az mount
the Moon's bite swings around once the disc is locked. It needs a rotation solve
— parallactic angle from time and pointing, or tracking a sunspot — not more
translation work. The other known gaps (white balance, blown highlights,
`--normalize` gains above 1) are documented under README → Limitations and are
accepted as they stand.
