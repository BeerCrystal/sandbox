// =====================================================================
//  Fit checks. Nothing here is printable — they exist so a clearance
//  mistake shows up on screen rather than on the bed.
//
//    make check
//
//  Every check must come out EMPTY. Anything solid is an interference.
// =====================================================================

include <stroller_clip.scad>

check = "hook_bar";
// [hook_bar, claw_bar, pieces, mount, cup_bars, liftoff]

// --- each piece against its own bar ----------------------------------

// The hook rests ON the arc, so this is contact, not overlap.
module check_hook_bar() { intersection() { hook_part(); arc_bar(); } }

// At rest the arm sits inside the claw's bore with clearance. The click
// only interferes while the bar is passing through the mouth, which is
// a different position, so at rest this must be clean.
module check_claw_bar() { intersection() { claw_placed(); arm_bar(); } }

// --- the two pieces against each other -------------------------------

// They meet face to face at joint_z. Overlap here would mean the pads
// cannot close, and the bolts would just spring the joint apart.
module check_pieces() { intersection() { hook_part(); claw_placed(); } }

module check_mount() { intersection() { hook_part(); cupholder(); } }

// The cup hangs inboard and below; it must miss both bars.
module check_cup_bars() {
    intersection() { cupholder(); union() { arc_bar(); arm_bar(); } }
}

// --- taking it off ---------------------------------------------------

// Slide the caddy up ALONG the arm and the hook has to lift clear of
// the arc. Stepped union rather than hull() — these shapes are
// non-convex, and the convex hull of two poses sweeps through material
// the part never actually occupies.
lift_mm   = 30;
lift_step = 1.5;

module check_liftoff() {
    intersection() {
        arc_bar();
        for (d = [0 : lift_step : lift_mm])
            translate([d * sin(arm_tilt), d * cos(arm_tilt), 0]) hook_part();
    }
}

if      (check == "hook_bar") check_hook_bar();
else if (check == "claw_bar") check_claw_bar();
else if (check == "pieces")   check_pieces();
else if (check == "mount")    check_mount();
else if (check == "cup_bars") check_cup_bars();
else if (check == "liftoff")  check_liftoff();
