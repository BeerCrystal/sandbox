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
// [arc_bar, arm_bar, mount, cup_bars, liftoff, trapped, exists_caddy,
//  exists_cup]

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

module check_exists_caddy() { caddy(); }
module check_exists_cup()   { cupholder(); }

if      (check == "arc_bar")      check_arc_bar();
else if (check == "arm_bar")      check_arm_bar();
else if (check == "mount")        check_mount();
else if (check == "cup_bars")     check_cup_bars();
else if (check == "liftoff")      check_liftoff();
else if (check == "trapped")      check_trapped();
else if (check == "exists_caddy") check_exists_caddy();
else if (check == "exists_cup")   check_exists_cup();
