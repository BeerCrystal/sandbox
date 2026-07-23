# Apple Watch "Pocket Carry" Case — 1st-gen (Series 0)

A 3D-printable case that turns an original Apple Watch into a standalone,
carryable pocket gadget — the *"put the phone down"* fob you saw in the
Instagram ad. Those cases are built for the **Apple Watch Ultra**; this design
re-creates the same idea (bolted front/back sandwich, exposed screen, side
controls, back charging window, bottom lanyard) but is **re-dimensioned for the
1st-generation Apple Watch (Series 0)** in both **38 mm** and **42 mm** sizes.

Everything is a single parametric OpenSCAD file, so you can tune every dimension
to your exact watch and your printer.

![overview](img/overview.png)

---

## How it works

Two printed parts clamp the watch like a sandwich:

| Part | Role |
|------|------|
| **Bezel** (front) | The watch drops in **face-first**. A lip around the screen window holds it from falling forward. The right wall has an opening for the **Digital Crown** (with a finger scallop) and a slot for the **side button**. A transverse **lanyard hole** runs through the bottom block. |
| **Backplate** | Bolts onto the back with 4 hex bolts and clamps the watch. A **central round window** exposes the sensor/back so the **magnetic charger still reaches it** — you can charge without removing the watch. |

**Hardware you supply:** 4× **M3×10 hex-head bolts** (DIN 933, 5.5 mm across
flats — hardware-store standard) and a length of **paracord** for the lanyard.
Prefer machine threads over self-tapping? Swap `screw_pilot_d` to 4.0 and use
M3 heat-set inserts.

### The bolt detail

The bolt heads are **not hidden** — each head drops into a shallow hex-shaped
seat on the backplate and stands **~1.2 mm proud** of the surface:

- **Grip** — four raised metal studs under your fingers, exactly the tactile
  hardware look of the reference design.
- **Security** — once seated, the hex pocket keys the head against rotation, so
  the bolts cannot vibrate loose in a pocket. To remove them you back the case
  apart deliberately; they never walk out on their own.
- Tighten with a 5.5 mm nut driver / socket until the head sinks into its seat,
  then a final snug by hand. Don't overtighten — printed threads strip.

---

## Ergonomics — will it disappear into a small/medium hand?

Finished dimensions (42 mm watch): **42.4 × 65.3 × 15.7 mm**. For scale:

| Object | Size (mm) |
|--------|-----------|
| Zippo lighter | 38 × 57 × 13 |
| **This case (42 mm watch)** | **42.4 × 65.3 × 15.7** |
| **This case (38 mm watch)** | **39.3 × 61.4 × 15.7** |
| AirPods Pro case | 45 × 61 × 22 |
| Car key fob (typical) | ~40 × 75 × 18 |

So it sits between a Zippo and an AirPods case — and it's **thinner than
both** feel in the pocket. A small-to-medium adult palm is roughly 70–85 mm
across; at ~42 mm wide the case spans about half the palm, and at ~65 mm long
it tucks fully inside curled fingers. Closed-fist carry works; pocket carry
disappears.

Shape decisions made for hand feel:

- **All outer edges filleted** (1.2 mm front part, 1.0 mm back) — no sharp
  corner anywhere; it feels like a river pebble, not a project box.
- **5 mm corner radius** on the brick outline.
- **Slimmed end blocks** vs. a naive design — only as much material above and
  below the watch as the bolts and lanyard actually need.
- **Finger scallop** around the Digital Crown opening so your fingertip can
  reach the crown through a 2.6 mm wall without the opening being oversized.
- The parting line between the two parts reads as a subtle V-groove seam that
  runs around the middle — a design line, and a tactile locator for which way
  the case is facing in your pocket.

The width and thickness are watch-driven (36.4 mm body + minimum walls;
10.5 mm body + front face + backplate) and can't shrink further without
thinning protection.

---

## ⚠️ Read this first — you must confirm two things

I built this from the **published Series 0 body dimensions**, but I could not
measure *your* watch or *your* printer. Two things need your eyes before you
commit to a full print:

1. **Which size is your watch — 38 mm or 42 mm?**
   Measure the aluminum body height (top to bottom, ignore the band):
   - ≈ **38.6 mm** → use `WATCH_SIZE = 38`
   - ≈ **42.5 mm** → use `WATCH_SIZE = 42`
   (Or check the back engraving / Settings → General → About → Model.)

2. **The Crown & side-button positions are estimated.** Their exact height on
   the body is my best estimate. The openings are made generous to be forgiving,
   but **print the test ring first** (below) to verify before printing the whole
   case.

Everything is parametric precisely so you can nudge these. Defaults are sane
starting points, not guarantees.

---

## Files

```
apple_watch_carry_case.scad   ← the parametric source (edit this)
stl/
  bezel_38mm.stl     backplate_38mm.stl
  bezel_42mm.stl     backplate_42mm.stl
img/                            ← reference renders
```

Ready-to-slice STLs are in `stl/` for both sizes. **But please tune tolerances
(below) and print a test piece before the full case** — a 2 h reprint beats a
watch that rattles or won't seat.

---

## Print settings

| Setting | Recommendation |
|---------|----------------|
| Material | **PETG** (tough, slight flex, warm-pocket safe) or PLA to start |
| Layer height | 0.2 mm |
| Walls / perimeters | **4** (this is a protective case — make it solid) |
| Infill | 30–40 % |
| Supports | **None needed.** Print the **bezel face-down** (screen window on the bed) and the **backplate mating-face-down** (hex seats up). Both are authored for support-free printing. |
| Orientation | As exported. |

---

## Tuning the fit (the important part)

Open `apple_watch_carry_case.scad` and adjust the variables at the top. The ones
that matter most:

```scad
WATCH_SIZE = 42;   // 38 or 42
tol        = 0.40; // horizontal gap around the watch — the #1 fit dial
depth_tol  = 0.40; // gap along thickness (how hard the backplate clamps)
win_lip    = 2.2;  // how far the front frame overlaps the watch edge
```

- **Watch rattles / too loose** → lower `tol` toward `0.25`.
- **Watch won't seat / too tight** → raise `tol` toward `0.55`.
- **Backplate won't pull flush** → raise `depth_tol`.
- **Frame covers part of the screen** → lower `win_lip` (but keep ≥ 1.5 mm so it
  still retains the watch).

Crown / button openings, if they don't line up:

```scad
crown_off_y  = 0;   // + moves the crown opening up, - down (mm from mid-height)
button_off_y = 0;   // same for the side button
crown_cut_d  = 11;  // make bigger if the crown is hard to reach
crown_scallop= 2.5; // finger scallop width around the crown (0 = off)
button_cut_h = 14;  // side-button slot length
```

Bolt seats / hardware:

```scad
hex_af   = 5.5;  // bolt head across-flats (M3 DIN 933 = 5.5)
hex_clr  = 0.40; // seat clearance — increase if heads won't drop in
hex_seat = 0.8;  // seat depth; 2 mm head → ~1.2 mm proud. Deeper = flusher.
```

Shape / feel:

```scad
fillet_bez = 1.2; // edge rounding, front part
fillet_bak = 1.0; // edge rounding, backplate
outer_r    = 5.0; // brick corner radius
back_win_d = 0;   // 0 = auto (watch_w − 5). Enlarge if your charger puck is wide.
lanyard_d  = 5.0; // paracord hole Ø (type-III paracord ≈ 4 mm)
```

---

## Print a cheap test first (recommended)

Before the full case, slice **only the bezel** and print just the **first
~4 mm** (stop the print early, or set a low object height in your slicer). Drop
the watch in to check:

1. Does the body **seat fully** with a snug, no-rattle fit? → tune `tol`.
2. Do the **Crown and button openings line up**? → tune `crown_off_y` /
   `button_off_y`.

Re-export and reprint the test until both are right, *then* commit to the full
bezel + backplate.

---

## Assembly

1. Print **bezel** + **backplate**.
2. Drop the watch into the bezel, **screen toward the front window**.
3. Set the **backplate** on the back, aligning the 4 corner holes.
4. Drive the **4 M3×10 hex bolts** through the backplate into the bezel pilot
   holes with a 5.5 mm nut driver until each head settles into its hex seat.
   Snug, don't overtighten (printed threads strip if forced).
5. Thread **paracord** through the bottom hole, knot a loop, add a cord toggle.
6. Charge by resting the magnetic charger on the **back window**.

---

## Re-generating STLs

Requires [OpenSCAD](https://openscad.org). From this folder:

```bash
# one part / one size
openscad -o stl/bezel_42mm.stl -D 'part="bezel"' -D 'WATCH_SIZE=42' apple_watch_carry_case.scad

# all four at once
for SZ in 38 42; do for P in bezel backplate; do
  openscad -o stl/${P}_${SZ}mm.stl -D "part=\"$P\"" -D "WATCH_SIZE=$SZ" apple_watch_carry_case.scad
done; done
```

Or just open the `.scad` in the OpenSCAD GUI and use the **Customizer** panel to
drag the parameters and hit **Render → Export STL**.

---

## Design notes / honest limitations

- **Body dimensions** are the official Series 0 figures (38 mm: 33.3 × 38.6 ×
  10.5 mm; 42 mm: 36.4 × 42.5 × 10.5 mm). **Corner radius, Crown height, and
  button height are estimated** and exposed as parameters for you to correct.
- The Series 0 back is **slightly domed**; the central window gives it clearance
  and lets the charger through. If the fit is loose front-to-back, a thin
  adhesive foam pad on the inside of the backplate takes up the slack nicely.
- **Bolt length matters**: M3×10 engages ~8 mm of printed pilot — right at the
  sweet spot. M3×12 will bottom out just as the head seats; anything longer
  will hit the blind end of the pilot before clamping. Stick to ×10.
- Grip **knurling** from the reference video is intentionally left off (it makes
  slicing slow and prints fuzzier). Add texture in your slicer ("fuzzy skin") if
  you want the look — the proud hex heads already give real grip points.

### Changelog

- **v2** — ergonomic pass: filleted all outer edges, slimmed end blocks
  (~4 mm shorter), swapped countersunk screws for proud hex-head bolts in
  keyed hex seats, added crown finger scallop, **fixed a v1 defect** where the
  lanyard hole left only ~0.5 mm of rim at the bottom edge (now ~2 mm).
- **v1** — initial design.
