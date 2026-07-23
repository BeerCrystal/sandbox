# Apple Watch "Pocket Carry" Case — 1st-gen (Series 0)

A 3D-printable case that turns an original Apple Watch into a standalone,
carryable pocket gadget — the *"put the phone down"* fob you saw in the
Instagram ad. Those cases are built for the **Apple Watch Ultra**; this design
re-creates the same idea (screwed front/back sandwich, exposed screen, side
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
| **Bezel** (front) | The watch drops in **face-first**. A lip around the screen window holds it from falling forward. The right wall has an opening for the **Digital Crown** and a slot for the **side button**. A transverse **lanyard hole** runs through the bottom block. |
| **Backplate** | Screws onto the back with 4 screws and clamps the watch. A **central round window** exposes the sensor/back so the **magnetic charger still reaches it** — you can charge without removing the watch. |

**Hardware you supply:** 4× **M3 self-tapping screws ~16 mm** (or M3 machine
screws + heat-set inserts) and a length of **paracord** for the lanyard.

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

## Finished size

| Watch | Footprint (W × H) | Thickness | Feel |
|-------|-------------------|-----------|------|
| 38 mm | ≈ 39 × 65 mm | ≈ 16 mm | matchbox-in-pocket |
| 42 mm | ≈ 42 × 69 mm | ≈ 16 mm | matchbox-in-pocket |

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
| Supports | **None needed.** Print the **bezel face-down** (screen window on the bed) and the **backplate flat**. Both are authored for support-free printing. |
| Orientation | As exported — bezel front face on the bed, pocket opening up. |

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
button_cut_h = 14;  // side-button slot length
```

Back window / lanyard:

```scad
back_win_d = 0;    // 0 = auto (watch_w − 5). Enlarge if your charger puck is wide.
lanyard_d  = 5.5;  // paracord hole Ø
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
4. Drive the **4 M3 screws** through the backplate counterbores into the bezel
   pilot holes. Snug, don't overtighten (printed threads strip if forced).
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
- Grip **knurling** from the reference video is intentionally left off (it makes
  slicing slow and prints fuzzier). The outer corners are lightly rounded for
  hand feel. Add texture in your slicer ("fuzzy skin") if you want the look.
