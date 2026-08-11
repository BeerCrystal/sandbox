# Chicco Corso corner caddy

One printed piece. Snaps onto the side arm of the handle, slides down, and
locks under the top arc. Carries a cup inboard, with a swappable attachment
interface.

Written from scratch in OpenSCAD. Nothing is derived from anyone else's model.

![front](build/front.png)

## How it works

**On:** hold it high so the hook clears the arc → snap the claw onto the side
arm → slide down until the hook seats.
**Off:** slide up until the hook lifts clear, then unsnap.

The **hook** is an inverted U resting on top of the arc. It carries the weight
and sets the height — sliding down stops when it bottoms out, which is what
keeps the caddy from slipping down the arm.

The **claw** is a light snap. Because the arm is tilted it slides freely *along*
it, and that is where the travel comes from. It springs **2.9 mm** to click on
and holds nothing but itself.

### Why it locks

Each grip blocks the direction the other releases in:

| | |
|---|---|
| Hook straddles the arc front-to-back | leaves **0.8 mm** of front-to-back play |
| Claw's mouth points **rearward** | into exactly that constraint — it cannot open |
| Claw wraps the arm side-to-side | stops the caddy sliding along the arc |
| Hook rests on the bar | stops it going down |

Up is the only motion left, and that is the intended release.

**The mouth direction *is* the lock.** Point it outboard and the whole caddy
pulls straight off sideways with the hook still seated. `make check` proves
this: the `trapped` check must come out *solid*.

## Measurements

**Done — the bar cross-sections.** One continuous 29 × 19 mm oval tube: bare on
the side arm, wrapped in 2–3 mm of rubber on the top arc.

| | front-to-back | the other axis |
|---|---|---|
| Top arc (rubber) | 23 | 35 top-to-bottom |
| Side arm (metal) | 19 | 29 across the handle plane |

The bend happens *in* the handle plane, so the 19 mm axis stays front-to-back
all the way round while the 29 mm axis rotates from vertical on the arc to
side-to-side on the arm. That is why `arc_h` and `arm_w` are the same number.

**Still needed — the corner.** One piece means the geometry is baked in:

- `nom_run` — horizontal distance, hook spot to claw spot
- `nom_drop` — vertical distance between the same two spots
- `arm_tilt` — degrees the side arm leans out from vertical

Two things make this less fussy than it sounds. **The claw slides along the
arm**, so error *along* the arm just parks the caddy slightly higher or lower —
self-correcting. Only the component *perpendicular* to the arm matters. And the
claw is deliberately short (20 mm), so a 5° angle error is only **0.87 mm**
across the bore, which the clearance absorbs.

## Workflow

```sh
make            # STLs into build/
make check      # eight checks — all must say "ok"
make preview    # PNGs
```

Print `testfit` (~15 min, no supports) to check the attachment interface before
committing to the full part.

## Printing

Exported standing on the **arm axis**. The claw is then a true vertical prism,
the strut and throat are vertical walls, and the throat opens upward. The only
overhang is the hook's top plate bridging the bar channel — a routine 24 mm
bridge, no supports needed.

| Part | Size | Supports |
|---|---|---|
| `caddy` | 89 × 49 × 142 mm | none |
| `cupholder` | 86 × 93 × 104 mm | touching buildplate, under the tongue only |
| `testfit` | 41 × 44 × 31 mm | none |

PLA is fine — the light click is what buys that. 4 perimeters, 30 % infill.
No hardware.

## The attachment interface

A **throat** on the inboard face: a channel open at the top. Attachments drop in
— tongue into the throat, crown over the top, spine down the outboard face,
cheeks straddling the hook. Lift straight up to remove.

| | value |
|---|---|
| throat width | 10 mm |
| throat depth | 15 mm |
| throat root | z = 17.9 mm |

Coordinates, standing behind the stroller: **+X along the arc**, **+Y up**,
**+Z inboard toward the seat**. The hook sits at the origin; the model is drawn
seated.

```scad
include <stroller_clip.scad>

module phone_tray() {
    hook_mount(spine_bottom = -50);
    translate([0, -50, spine_z + spine_t - 2]) cube([60, 40, 8], center = true);
}

phone_tray();
```

Add it to `checks.scad` and run `make check` before printing. The throat doubles
as a plain hook — a bag loop drops in and the tip stops it sliding off.

## Tuning

| Symptom | Change |
|---|---|
| Claw won't click on | raise `claw_frac` |
| Claw falls off before you slide down | lower `claw_frac` |
| Caddy rattles | lower `arm_slop` |
| Won't seat — binds at an angle | raise `arm_slop`, or shorten `claw_len` |
| Hook lifts off over bumps | raise `hook_engage` |
| Sits too high or low on the arm | nothing — it self-corrects along the arm |
| Attachment rattles in the throat | lower `tongue_gap` |

## The checks

Eight, in three kinds — and the distinction is the point:

- **NONEMPTY** (`exists_*`) — the parts render at all.
- **EMPTY** (`arc_bar`, `arm_bar`, `mount`, `cup_bars`, `liftoff`) — clearances.
- **SOLID** (`trapped`) — the hook really does block release.

The existence gate is not padding. Every "must be empty" check passes trivially
against a part that renders to nothing, and during development a silently
dropped body made the entire suite green while the hook did not exist. An
empty-intersection suite is worthless without proof the parts are there.

`stroller_clip.scad` also carries `assert()` guards: OpenSCAD turns an undefined
name into `undef`, and `linear_extrude(undef)` quietly builds something enormous
rather than failing.

## Caveats

- **The corner offsets are still nominal.** See *Measurements*.
- **The hook assumes a straight 34 mm run** of arc. It sits past the bend where
  the arc is roughly level, but the arc is still gently curved — if it rocks,
  drop `hook_len`.
- **The cup hangs ~87 mm inboard.** Mostly ring radius; reduce `cup_id`.
- **Not load rated.** Nothing has been physically tested.
- **Don't hang heavy loads on a stroller handle.** Weight up there makes a
  stroller tip backwards, a hazard independent of this part's strength.

## Files

| File | |
|---|---|
| `stroller_clip.scad` | parameters, the caddy, attachment interface, cup holder |
| `gauge.scad` | bar measuring gauge |
| `checks.scad` | the eight checks, via `make check` |
| `print/gauge.stl` | ready to print, no OpenSCAD needed |
