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
import math
import sys

D1 = 35.0     # window depths below the arc's centreline, from corner_gauge.scad
D2 = 105.0
ARM_W = 29.0  # known width of the side arm, used as a sanity check

CLAW_DEPTH = 90.0   # where the claw sits down the arm; a design choice,
                    # not a measurement — the claw slides, so this is free


def solve(l1, r1, l2, r2, claw_depth=CLAW_DEPTH):
    l1, r1 = min(l1, r1), max(l1, r1)              # edge order does not matter
    l2, r2 = min(l2, r2), max(l2, r2)
    a1, a2 = (l1 + r1) / 2.0, (l2 + r2) / 2.0      # centreline at each window
    w1, w2 = abs(r1 - l1), abs(r2 - l2)

    span = D2 - D1
    slope = (a2 - a1) / span
    tilt = math.degrees(math.atan(slope))
    a_claw = a1 + slope * (claw_depth - D1)   # interpolate along the axis

    return {
        "arm_tilt": tilt,
        "nom_run": a_claw,
        "nom_drop": claw_depth,
        "widths": (w1, w2),
        "centres": (a1, a2),
        # Each edge fitted on its own. The spread between them is a free
        # read of how precisely the marks were made.
        "edge_angles": (math.degrees(math.atan2(l2 - l1, span)),
                        math.degrees(math.atan2(r2 - r1, span))),
        # The axis run back to the arc's centreline. The gauge is pushed
        # against the bend, so this SHOULD land near zero -- and nothing
        # in the arithmetic forces it to. It is the one check that says
        # the gauge was held in the right plane.
        "axis_at_zero": a1 - D1 * slope,
        "expect_w": ARM_W / math.cos(math.radians(tilt)),
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

    print(f"edges fitted separately: {r['edge_angles'][0]:.2f} / "
          f"{r['edge_angles'][1]:.2f} deg  (spread "
          f"{abs(r['edge_angles'][0] - r['edge_angles'][1]):.2f})")

    # A bar ARM_W wide crossing a HORIZONTAL scale at angle t reads
    # ARM_W/cos(t), not ARM_W. Comparing against the bare width was
    # wrong: harmless near vertical, but at 45 degrees it cries wolf by
    # 13 mm and would send you back to remeasure a good reading.
    print(f"widths {r['widths'][0]:.0f} / {r['widths'][1]:.0f} mm vs "
          f"{r['expect_w']:.1f} expected at this angle")

    # Overshooting both edges of the bar by the same amount cancels in
    # the centre, so it costs nothing on the angle. Only a LOPSIDED
    # error moves the centre, so that is what is worth warning about.
    lop = abs(r["widths"][0] - r["widths"][1])
    if lop > 8:
        print(f"  !! the two widths disagree by {lop:.0f} mm. Even overshoot"
              f" is harmless, uneven overshoot is not — re-read them.")

    # The axis run back to the arc's centreline. The gauge butts against
    # the bend, so this should land near zero — and no part of the
    # arithmetic forces it to. It is the one check that catches a gauge
    # held out of the handle's plane, which nothing else here would see.
    z = r["axis_at_zero"]
    if abs(z) < 15:
        print(f"axis extrapolated to depth 0: {z:.1f} mm from the origin"
              f"  <- near zero, so the gauge sat in the right plane")
    else:
        print(f"  !! axis extrapolates to {z:.1f} mm at depth 0, and it should"
              f" land near 0 since the gauge butts against the bend."
              f" Suspect it was not held flat against both bars.")

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
