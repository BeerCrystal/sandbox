// =====================================================================
//  Handle taper gauge
//
//  A flat plate with a wedge-shaped slot cut into one edge, graduated
//  in millimetres. Press the slot onto the stroller handle until it
//  stops; the tick mark level with the handle is the handle's size in
//  that direction.
//
//  Measure twice:
//    - plate held vertically  -> handle HEIGHT  (top to bottom)
//    - plate held horizontally -> handle WIDTH  (front to back)
//
//  Prints flat, no support, ~30 min. Print it before anything else --
//  the clip cannot be sized without these two numbers.
// =====================================================================

// Only the engraved digits have curves, so this can stay low.
$fn = 24;

/* [Range] */
size_max   = 38;   // slot width at the mouth, mm
size_min   = 16;   // slot width at the closed end, mm
label_step = 2;    // number every N mm
tick_step  = 1;    // tick every N mm

/* [Plate] */
slot_len   = 130;  // length of the tapered slot
margin     = 13;   // material either side of the slot
tail       = 16;   // solid material past the closed end (the handle)
thick      = 4;    // plate thickness
engrave    = 0.6;  // depth of ticks and numbers

// --- derived ---------------------------------------------------------

plate_w = size_max + 2 * margin;
plate_l = slot_len + tail;

// Distance from the mouth at which the slot is `d` mm wide.
function pos(d) = (size_max - d) / (size_max - size_min) * slot_len;

module plate_2d() {
    square([plate_l, plate_w], center = false);
}

// The wedge, opening at x = 0.
module slot_2d() {
    polygon([
        [-1,       (plate_w - size_max) / 2],
        [slot_len, (plate_w - size_min) / 2],
        [slot_len, (plate_w + size_min) / 2],
        [-1,       (plate_w + size_max) / 2],
    ]);
}

// Ticks run outward from the slot edge so the handle never covers them.
module ticks_2d() {
    for (d = [size_min : tick_step : size_max]) {
        major = (d % label_step == 0);
        len   = major ? 7 : 4;
        y     = (plate_w - d) / 2;      // lower edge of the slot here
        translate([pos(d) - 0.4, y - len]) square([0.8, len]);
        translate([pos(d) - 0.4, plate_w - y]) square([0.8, len]);
    }
}

module labels_2d() {
    for (d = [size_min : label_step : size_max]) {
        translate([pos(d), (plate_w - d) / 2 - 8.5])
            rotate([0, 0, 90])
                text(str(d), size = 5, halign = "center",
                     valign = "center", font = "Liberation Sans:style=Bold");
    }
}

module title_2d() {
    translate([plate_l - 5, plate_w / 2])
        rotate([0, 0, 180])
            text("HANDLE GAUGE  mm", size = 5.5, halign = "left",
                 valign = "center", font = "Liberation Sans:style=Bold");
}

module gauge() {
    difference() {
        linear_extrude(thick) difference() {
            plate_2d();
            slot_2d();
        }
        translate([0, 0, thick - engrave])
            linear_extrude(engrave + 1) {
                ticks_2d();
                labels_2d();
                title_2d();
            }
    }
}

gauge();
