// =====================================================================
//  Fit checks.
//
//    make check
//
//  Three kinds, and the distinction matters:
//
//    EMPTY    — clearance checks. Anything solid is an interference.
//    SOLID    — the interlock check. An empty result means the caddy
//               can be pulled off while still hooked on.
//    NONEMPTY — the parts themselves.
//
//  That last one exists because it has already bitten once. Every
//  "must be empty" check passes trivially against a part that renders
//  to nothing, so a silently dropped body made the whole suite green
//  while the hook did not exist. Never trust an empty-intersection
//  suite without also proving the parts are there.
// =====================================================================

include <stroller_clip.scad>

check = "arc_bar";
// [arc_bar, arm_bar, mount, cup_bars, insert, liftoff, trapped,
//  fused_bars, fused_lift, exists_caddy, exists_cup, exists_fused]

// --- the caddy against the handle ------------------------------------

// The hook rests ON the arc and the claw wraps the arm with clearance,
// so both of these are contact, not overlap.
module check_arc_bar() { intersection() { caddy(); arc_bar(); } }
module check_arm_bar() { intersection() { caddy(); arm_bar(); } }

module check_mount()    { intersection() { caddy(); cupholder(); } }

module check_cup_bars() {
    intersection() { cupholder(); union() { arc_bar(); arm_bar(); } }
}

// --- taking it off ---------------------------------------------------

// Slide up ALONG the arm; the hook must lift clear of the arc. Stepped
// union rather than hull() — these shapes are non-convex, and the
// convex hull of two poses sweeps through material the part never
// actually occupies.
lift_mm   = 34;
lift_step = 1.5;

module check_liftoff() {
    intersection() {
        arc_bar();
        for (d = [0 : lift_step : lift_mm])
            translate([d * sin(arm_tilt), d * cos(arm_tilt), 0]) caddy();
    }
}

// --- the interlock: MUST BE SOLID ------------------------------------
//
//  Releasing the claw means pushing the caddy inboard until the arm
//  clears the mouth. If the hook is doing its job it fouls the arc long
//  before that.

release_push = arm_h / 2 + wall;

module check_trapped() {
    intersection() {
        translate([0, 0, release_push]) caddy();
        arc_bar();
    }
}

// --- the parts exist: MUST BE NONEMPTY -------------------------------

// The fused variant welds the cup on directly, so it has to clear the
// handle on its own terms -- the drop-in geometry is not involved.
module check_fused_bars() {
    intersection() { fused(); union() { arc_bar(); arm_bar(); } }
}

module check_fused_lift() {
    intersection() {
        union() { arc_bar(); arm_bar(); }
        for (d = [0 : lift_step : lift_mm])
            translate([d * sin(arm_tilt), d * cos(arm_tilt), 0]) fused();
    }
}

// --- can a cup actually go IN? ---------------------------------------
//
//  cup_bars only proves the holder does not touch the handle while it
//  sits there. It says nothing about the path a cup takes on the way
//  in, which is straight down into the ring -- and the arc runs right
//  over that. A holder can pass every clearance check here and still be
//  unusable because the handle bar is in the way of your hand.
//
//  So: sweep the ring's bore vertically upward and intersect it with
//  the bars. Anything solid is the handle fouling the insertion path.

insert_h = 220;   // a tall travel mug plus room for fingers

module insert_path() {
    translate([0, ring_top, cup_cz])
        cylinder(h = insert_h, r = cup_id / 2);
}

module check_insert() {
    intersection() { insert_path(); union() { arc_bar(); arm_bar(); } }
}

// Same question for the fused variant, whose cup sits ~30 mm closer in.
module check_fused_insert() {
    intersection() {
        translate(cup_pos) cylinder(h = insert_h, r = cup_id / 2);
        union() { arc_bar(); arm_bar(); }
    }
}

module check_exists_caddy() { caddy(); }
module check_exists_fused() { fused(); }
module check_exists_cup()   { cupholder(); }

if      (check == "arc_bar")      check_arc_bar();
else if (check == "arm_bar")      check_arm_bar();
else if (check == "mount")        check_mount();
else if (check == "cup_bars")     check_cup_bars();
else if (check == "liftoff")      check_liftoff();
else if (check == "trapped")      check_trapped();
else if (check == "exists_caddy") check_exists_caddy();
else if (check == "exists_cup")   check_exists_cup();
else if (check == "exists_fused") check_exists_fused();
else if (check == "fused_bars")   check_fused_bars();
else if (check == "fused_lift")   check_fused_lift();
