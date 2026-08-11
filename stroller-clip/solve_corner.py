#!/usr/bin/env python3
"""Turn corner-gauge readings into the caddy's corner parameters.

    python3 solve_corner.py L1 R1 L2 R2

L1/R1 are the left and right edges of the side arm read in WINDOW 1,
L2/R2 the same in WINDOW 2, all in millimetres outboard of the origin
(so they are the numbers printed on the scale, no signs needed).

Two crossings define the arm's axis, which is the whole point: the angle
is what is hard to judge by eye, and the 70 mm baseline between the two
windows is what measures it well.
"""
import sys

D1 = 35.0     # window depths below the arc's centreline, from corner_gauge.scad
D2 = 105.0
ARM_W = 29.0  # known width of the side arm, used as a sanity check

CLAW_DEPTH = 90.0   # where the claw sits down the arm; a design choice,
                    # not a measurement — the claw slides, so this is free


def solve(l1, r1, l2, r2, claw_depth=CLAW_DEPTH):
    a1, a2 = (l1 + r1) / 2.0, (l2 + r2) / 2.0      # centreline at each window
    w1, w2 = abs(r1 - l1), abs(r2 - l2)

    span = D2 - D1
    tilt = __import__("math").degrees(__import__("math").atan2(a2 - a1, span))
    # linear interpolation along the axis to wherever the claw goes
    a_claw = a1 + (a2 - a1) * (claw_depth - D1) / span

    return {
        "arm_tilt": tilt,
        "nom_run": a_claw,
        "nom_drop": claw_depth,
        "widths": (w1, w2),
        "centres": (a1, a2),
    }


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    l1, r1, l2, r2 = (float(v) for v in sys.argv[1:5])
    r = solve(l1, r1, l2, r2)

    print(f"window 1: edges {l1:.0f} / {r1:.0f}   centre {r['centres'][0]:.1f}"
          f"   width {r['widths'][0]:.1f}")
    print(f"window 2: edges {l2:.0f} / {r2:.0f}   centre {r['centres'][1]:.1f}"
          f"   width {r['widths'][1]:.1f}")

    # The measured width should come back close to the known arm width.
    # It is the only independent check on whether the scale was read
    # correctly, so it is worth shouting about.
    for i, w in enumerate(r["widths"], 1):
        if abs(w - ARM_W) > 4:
            print(f"  !! window {i} width is {w:.1f} mm, expected ~{ARM_W:.0f}."
                  f" Re-read it — something is off.")

    print()
    print("Set these in stroller_clip.scad:")
    print(f"    nom_run   = {r['nom_run']:.1f};")
    print(f"    nom_drop  = {r['nom_drop']:.1f};")
    print(f"    arm_tilt  = {r['arm_tilt']:.1f};")
    print()
    print("Then:  make CUP=vendor/<your-cup>.stl check && "
          "make CUP=vendor/<your-cup>.stl")


if __name__ == "__main__":
    main()
