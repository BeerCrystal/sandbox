// =====================================================================
//  Corner gauge — reads the side arm's axis relative to the top arc.
//
//  HOW TO USE
//    1. Slide the top slot onto the top arc, and push the gauge along
//       the arc TOWARD THE CORNER until the closed end of the slot
//       stops against the bend. The bend is the reference: that makes
//       the origin repeatable instead of "wherever I happened to hold
//       it". The hook will sit at that spot.
//    2. Press the gauge flat against the front faces of BOTH bars. That
//       is what puts it parallel to the plane of the handle. (The two
//       bars differ in depth by 2 mm, which tilts the gauge about 1
//       degree over its length — ignorable.)
//    3. Look through each window. Read the tick at the LEFT edge of the
//       arm and at the RIGHT edge, in both windows. Four numbers.
//
//    Read both edges rather than eyeballing the centre: averaging two
//    edge readings is more accurate than judging a midpoint, and the
//    width falling out at ~29 mm confirms you read the right scale.
//
//  WHY TWO WINDOWS
//    Two crossings define a line. From them I get the arm's angle AND
//    its offset, which is everything. A single reading plus an eyeballed
//    angle would be much worse — angle is the hard thing to judge by
//    eye, and it is exactly what the 70 mm baseline between these two
//    windows measures well.
//
//  Then run:  python3 solve_corner.py <L1> <R1> <L2> <R2>
//
//  Prints flat, no supports, ~45 min.
// =====================================================================

$fn = 32;

/* [Bar sizes — from the earlier gauge] */
arc_h    = 35;    // top arc, top-to-bottom: the slot straddles this
slot_gap = 1.5;   // clearance so it slides freely along the arc

/* [Windows] */
d1       = 35;    // depth of the near window below the arc's centreline
d2       = 105;   // depth of the far window
win_h    = 9;     // window height
reach    = 140;   // how far outboard the scales run

/* [Plate] */
right    = 50;    // material to the right of the origin
top      = 34;    // above the arc's centreline
bottom   = 136;   // below it
thick    = 4;
engrave  = 0.6;
tick_step = 2;
label_step = 20;

// --- derived ---------------------------------------------------------

slot_h  = arc_h + slot_gap;
edge    = 8;      // margin outside the scales

module plate_2d() {
    translate([-(reach + edge), -bottom]) square([reach + edge + right,
                                                 bottom + top]);
}

// Slot for the top arc. Open on the right, closed at x = 0 — push the
// gauge left along the arc until the bend stops it there.
module arc_slot_2d() {
    translate([0, -slot_h / 2]) square([right + 1, slot_h]);
    translate([0, 0]) circle(d = slot_h);          // rounded closed end
}

module window_2d(d) {
    translate([-(reach + 2), -d - win_h / 2]) square([reach + 2, win_h]);
}

// Ticks hang below each window so the bar never covers them.
module ticks_2d(d) {
    for (a = [0 : tick_step : reach]) {
        major = (a % label_step == 0);
        len   = major ? 7 : (a % 10 == 0 ? 5 : 3);
        translate([-a - 0.35, -d - win_h / 2 - len]) square([0.7, len]);
    }
}

module labels_2d(d) {
    for (a = [0 : label_step : reach])
        translate([-a, -d - win_h / 2 - 11.5])
            text(str(a), size = 5, halign = "center", valign = "center",
                 font = "Liberation Sans:style=Bold");
}

module titles_2d() {
    translate([-(reach + edge) + 4, -d1 + 9])
        text("CORNER GAUGE — press flat against both bars",
             size = 5, halign = "left", font = "Liberation Sans:style=Bold");
    translate([-(reach + edge) + 4, -d1 - win_h / 2 - 20])
        text(str("WINDOW 1   depth ", d1, " mm"), size = 5,
             halign = "left", font = "Liberation Sans:style=Bold");
    translate([-(reach + edge) + 4, -d2 - win_h / 2 - 20])
        text(str("WINDOW 2   depth ", d2, " mm"), size = 5,
             halign = "left", font = "Liberation Sans:style=Bold");
    translate([-30, top - 9])
        text("◄ OUTBOARD", size = 5.5, halign = "center",
             font = "Liberation Sans:style=Bold");
    // mark the origin, so it is obvious what the scales measure from
    translate([9, -34]) rotate([0, 0, -90])
        text("ORIGIN — hook sits here", size = 4.5, halign = "left",
             font = "Liberation Sans:style=Bold");
}

module origin_line_2d() {
    translate([-0.5, -bottom + 2]) square([1, bottom - slot_h / 2]);
}

module gauge() {
    difference() {
        linear_extrude(thick) difference() {
            plate_2d();
            arc_slot_2d();
            window_2d(d1);
            window_2d(d2);
        }
        translate([0, 0, thick - engrave])
            linear_extrude(engrave + 1) {
                ticks_2d(d1); ticks_2d(d2);
                labels_2d(d1); labels_2d(d2);
                titles_2d();
                origin_line_2d();
            }
    }
}

gauge();
