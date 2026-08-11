# Chicco Corso handle clip

A parametric handle clip for a Chicco Corso stroller: a clamp that grips the
handle from the side, a hook on top, and a documented interface so attachments
— cup holder, phone mount, bag hook — all share the same mount.

Written from scratch in OpenSCAD. Nothing here is derived from anyone else's
model, so there is no upstream licence to carry.

![assembly](build/assembly.png)

## ⚠ Status: the handle dimensions are guesses

`handle_w`, `handle_h` and `handle_r` in `stroller_clip.scad` are
**placeholders**. No Corso handle has been measured. Print the gauge, measure
your handle, set those three numbers, and only then print the clip.

## Workflow

```sh
make            # all STLs into build/
make check      # clearance checks — both must say "ok"
make preview    # PNG previews
```

**1 — Measure.** Print `build/gauge.stl` (or the ready-made copy in
`print/gauge.stl`, no OpenSCAD needed). Flat on the bed, no supports, ~30 min.

Press the tapered slot onto the handle until it stops and read the tick level
with the handle. Measure twice:

- plate held **vertically** → handle **height** (top to bottom) → `handle_h`
- plate held **horizontally** → handle **width** (front to back) → `handle_w`

For `handle_r`: if the grip is round, set all three to `w/2`. If it is a
flattened oval, start at `handle_r = min(w,h)/2` and reduce if the clip rocks.

**2 — Test the interface.** Set your three numbers, then print
`build/testfit.stl` (~15 min, no supports). It is just the mount and a small
loop. Check it drops onto the hook and does not rattle. Adjust `fit_gap`.

**3 — Print the clip**, then whatever attachment you want.

## Printing

| Part | Orientation | Supports | Notes |
|---|---|---|---|
| `clip` | as exported — standing on end, handle axis vertical | none | the profile is a prism, so nothing overhangs |
| `testfit` | as exported | none | grows upward off the crown |
| `cupholder` | as exported — ring axis vertical | **touching buildplate** | only needed under the tongue |
| `gauge` | flat | none | |

**PETG, not PLA.** The band has to spring open by `handle_h - mouth_w`
(~3.6 mm at the default `mouth_frac`) to pass over the handle. PLA is likely to
crack at that. PETG or ASA will take it.

5 perimeters, 40 % infill. The clip's strength comes almost entirely from
perimeters running around the band.

**Hardware (optional):** one M3 × 30 self-tapping screw for the pinch bolt.
The near ear is clearance, the far ear is a pilot, so tightening pulls the jaws
together. Set `use_bolt = false` for a hardware-free snap-on clip.

The rubber grip is why the bolt is worth having: rubber takes a compression set,
so a snap fit that feels tight today can be loose in a month. The bolt lets you
take that up.

## The attachment interface

The clip's hook is a **throat** — a channel open at the top, between the tower
(inboard) and the upturned tip (outboard). Attachments drop straight down into
it: a tongue into the throat, a crown over the top, a spine down the outboard
face, and two cheeks straddling the clip so nothing slides along the handle.
Lift straight up to remove.

At the default parameters:

| | value |
|---|---|
| throat width | `hook_reach - hook_t` = 11 mm |
| throat depth | `hook_rise` = 16 mm |
| throat floor | y = 13 mm |
| tower face | x = 23.5 mm |
| clip width along handle | `clamp_len` = 30 mm |

Coordinates are as fitted to the stroller: **+X outboard** (the way the hook
points, away from the person pushing), **+Y up**, **+Z along the handle**, with
z = 0 at the middle of the clip.

To build your own attachment, call `hook_mount()` and put your geometry
outboard of `spine_x + spine_t`:

```scad
include <stroller_clip.scad>

module phone_tray() {
    hook_mount(spine_bottom = -30);       // how far the spine runs down
    translate([spine_x + spine_t - 2, -30, 0])
        cube([40, 8, 60], center = false);   // your part goes here
}

upright() phone_tray();
```

Then add it to `checks.scad` and run `make check` before printing — that is
what catches a clearance mistake on screen instead of on the bed.

Because the throat is open at the top and the tip curls up, it also works as a
plain hook: a bag handle or shopping bag loop drops into the channel and the
tip stops it sliding off outboard.

## Tuning

| Symptom | Change |
|---|---|
| Clip will not go on | raise `mouth_frac` |
| Clip rotates on the handle | raise `rubber_bite`, or fit the bolt |
| Clip is loose after a few weeks | fit the bolt; this is rubber creep, not a design fault |
| Clip cracked going on | print in PETG; raise `mouth_frac` |
| Attachment rattles on the hook | lower `fit_gap` |
| Attachment will not seat | raise `fit_gap` |
| Want more load capacity | raise `clamp_len` and `band_wall` |

## Known limitations

- **Unmeasured.** See the status note above.
- **The sample cup holder hangs ~83 mm outboard** of the handle centreline —
  ring radius plus hook reach. That is a long lever arm. It is fine for the
  clip, but the cup sticks out further than a commercial holder. Reducing
  `cup_id` is the quickest way to pull it in.
- **The clip is not load rated.** It was sized against roughly 1.2 kg (a full
  large tumbler); the calculated bearing stress in the throat is trivial, but
  nothing has been physically tested.
- **Don't hang heavy loads on a stroller handle.** Weight on the handle makes a
  stroller tip backwards, which is a hazard independent of how strong this part
  is. Chicco's own manual warns about this.

## Files

| File | |
|---|---|
| `stroller_clip.scad` | parameters, the clip, the attachment interface, sample cup holder |
| `gauge.scad` | handle measuring gauge — print this first |
| `checks.scad` | clearance checks, run via `make check` |
| `print/gauge.stl` | ready to print, no OpenSCAD needed |
