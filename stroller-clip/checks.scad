// =====================================================================
//  Fit checks. These render nothing you would print -- they exist so a
//  clearance mistake shows up here rather than on the print bed.
//
//    make check
//
//  Both checks must come out EMPTY. Anything solid is an interference.
// =====================================================================

include <stroller_clip.scad>

check = "mount";   // [mount, cheeks]

// Does the attachment foul the clip anywhere?
module check_mount() { intersection() { clip(); cupholder(); } }

// Does the attachment foul the clip on the way down? Stepping it
// straight up off the hook traces the whole removal path; if that is
// clear it also drops on cleanly.
//
// Stepped union rather than hull() -- these shapes are non-convex, and
// the convex hull of two poses sweeps through solid material that the
// part never actually occupies.
lift_mm   = 26;
lift_step = 1;

module check_cheeks() {
    intersection() {
        clip();
        for (dy = [0 : lift_step : lift_mm])
            translate([0, dy, 0]) cupholder();
    }
}

if      (check == "mount")  check_mount();
else if (check == "cheeks") check_cheeks();
