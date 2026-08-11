// =====================================================================
//  Fit checks. Nothing here is printable -- they exist so a clearance
//  mistake shows up on screen rather than on the bed.
//
//    make check
//
//  Every check must come out EMPTY. Anything solid is an interference.
// =====================================================================

include <stroller_clip.scad>

check = "seated";   // [seated, raised, release, mount, lift]

// --- the clip against the bar ---------------------------------------

// Seated and locked: the bar sits at the top of the slot, under the
// cap. Contact, but no interference.
module check_seated() {
    intersection() { clip(); handle_bar(); }
}

// Raised by slide_travel: the bar drops to the bottom of the slot and
// lines up with the mouth. If this is not empty the clip cannot be
// lifted far enough to come off.
module check_raised() {
    intersection() {
        translate([0, slide_travel, 0]) clip();
        handle_bar();
    }
}

// Pulling the raised clip off sideways. The bar has to squeeze through
// the mouth, so a band of interference across the mouth is EXPECTED --
// that is the click. What must be empty is the cap: if the cap fouls
// the bar on the way out, the clip is trapped.
//
// So this checks the cap station alone, not the jaw.
module check_release() {
    intersection() {
        for (dx = [0 : 2 : bore_w + wall + 6])
            translate([dx, slide_travel, 0])
                translate([0, 0, cap_z0]) linear_extrude(cap_len) cap_2d();
        handle_bar();
    }
}

// --- the attachment against the clip --------------------------------

module check_mount() { intersection() { clip(); cupholder(); } }

// Stepping the attachment straight up off the hook traces the whole
// removal path. Stepped union rather than hull() -- these shapes are
// non-convex, and the convex hull of two poses sweeps through material
// the part never actually occupies.
lift_mm   = 26;
lift_step = 1;

module check_lift() {
    intersection() {
        clip();
        for (dy = [0 : lift_step : lift_mm])
            translate([0, dy, 0]) cupholder();
    }
}

if      (check == "seated")  check_seated();
else if (check == "raised")  check_raised();
else if (check == "release") check_release();
else if (check == "mount")   check_mount();
else if (check == "lift")    check_lift();
