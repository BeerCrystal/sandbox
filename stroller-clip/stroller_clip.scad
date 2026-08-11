// =====================================================================
//  Chicco Corso — corner handle clip, click-and-drop interlock.
//
//  Coordinate convention, as fitted to the stroller:
//      +X   outboard — the side the body and attachments sit on
//      +Y   up
//      +Z   along the handle bar, z = 0 at the middle of the clip
//
//  The handle bar sits at the origin. The model is drawn in the
//  SEATED (locked) position.
//
//  ---------------------------------------------------------------
//  !! THE HANDLE DIMENSIONS BELOW ARE UNMEASURED PLACEHOLDERS.     !!
//  !! Print gauge.scad, measure the handle in both directions, and !!
//  !! set handle_w / handle_h / handle_r before printing anything. !!
//  ---------------------------------------------------------------
//
//  HOW IT GOES ON
//    1. Hold the clip high, so the cap clears the top of the bar.
//    2. Push it on sideways. The jaw clicks over the bar.
//    3. Slide it down. The cap drops over the top of the bar and the
//       bar rises to the top of the jaw slot.
//    Off is the reverse: lift first, then unclick. Under load it
//    cannot do that by itself.
//
//  WHY IT LOCKS
//    The jaw bore is a vertical SLOT, taller than the bar by
//    slide_travel. The mouth is cut at the bar's insertion height
//    only. Once seated, the bar has risen past the mouth and sits
//    against solid wall, so the jaw cannot release at all until the
//    clip is lifted. The cap over the top is what carries the hanging
//    load, which means the click itself only has to hold the part
//    steady during the slide -- so it can be light. That is the whole
//    point: a snap that has to resist load must be tight, and a tight
//    snap cracks.
//
//  PRINTING
//    Print standing on end, bar axis vertical, exactly as exported.
//    Every face is then a vertical wall: no supports, hoop stress in
//    the jaw runs along the extrusion lines, and the hanging load sits
//    in the layer plane instead of peeling layers apart. The jaw and
//    cap are deliberately adjacent along Z with no gap, so the cap's
//    far leg is printed on top of the jaw's far wall rather than
//    starting in mid-air.
// =====================================================================

part = "clip";   // [clip, cupholder, testfit, assembly, all]

$fn = 96;

// --- handle ----------------------------------------------------------
// Modelled as a rounded rectangle. A round bar is the case where
// handle_w == handle_h == 2 * handle_r.

handle_w   = 30;    // front-to-back, mm   *** PLACEHOLDER ***
handle_h   = 26;    // top-to-bottom, mm   *** PLACEHOLDER ***
handle_r   = 11;    // corner radius, mm   *** PLACEHOLDER ***

fit        = 0.4;   // clearance between clip and bar

// --- the interlock ---------------------------------------------------

slide_travel = 10;  // vertical throw between clicked-on and locked.
                    // Must exceed cap_engage or the cap will not clear
                    // the bar when you lift to remove it.
cap_engage   = 8;   // how far the cap's far leg reaches down past the
                    // top of the bar once seated

// Mouth width as a fraction of handle_h. This is the click. Unlike a
// load-bearing snap it only has to hold the clip steady while you slide
// it down, so it can stay light: 0.94 springs the jaw ~1.5 mm, which
// even PLA will take.
mouth_frac  = 0.94;

// --- structure -------------------------------------------------------

wall        = 4;    // wall thickness around the bar
jaw_len     = 22;   // length of the jaw station along the bar
cap_len     = 24;   // length of the cap station along the bar
backbone_t  = 6;    // thickness of the plate carrying the attachments
fillet_r    = 2;

// --- attachment interface -------------------------------------------
// A throat on the outboard face: a channel open at the top. Attachments
// drop in and lift out, and a bag loop hangs in it directly.

mnt_y       = 0;    // underside of the throat arm
mnt_t       = 5;    // throat stock thickness
mnt_reach   = 15;   // how far the arm reaches outboard
mnt_rise    = 15;   // throat depth

tongue_gap  = 0.35; // clearance per side, tongue to throat
tongue_len  = 32;   // length of the tongue along the bar
crown_t     = 6;
spine_t     = 6;
spine_gap   = 0.6;
cheek_t     = 3;
cheek_gap   = 0.4;
head_up     = 8;

// --- sample cup holder ----------------------------------------------

cup_id      = 80;
ring_wall   = 3;
ring_h      = 45;
ring_top    = -6;   // height of the ring's rim, in stroller coords
base_t      = 3;
drain_r     = 9;

// =====================================================================
//  Derived geometry
// =====================================================================

bore_w   = handle_w + 2 * fit;
bore_h   = handle_h + 2 * fit;
bar_top  = handle_h / 2 + fit;              // where the clip rests on the bar

slot_h   = bore_h + slide_travel;           // the jaw's vertical slot
slot_cy  = bar_top - slot_h / 2;            // seated: bar at the top of it
insert_y = -slide_travel;                   // bar height, in clip coords,
                                            // at the moment you click it on
mouth_h  = handle_h * mouth_frac;

bx0      = bore_w / 2 + wall;               // inboard face of the backbone
bx1      = bx0 + backbone_t;                // outboard face = throat root
far_x    = -(bore_w / 2 + wall);            // outer face of the far wall

cap_top  = bar_top + wall;
cap_bot  = bar_top - cap_engage;

// Adjacent along Z, no gap, so the cap's far leg prints onto the jaw.
body_len = jaw_len + cap_len;
jaw_z0   = -body_len / 2;
cap_z0   = jaw_z0 + jaw_len;

// Attachment interface
tip_inner  = bx1 + mnt_reach - mnt_t;
tip_outer  = bx1 + mnt_reach;
throat_w   = mnt_reach - mnt_t;
throat_y   = mnt_y + mnt_t;
throat_top = throat_y + mnt_rise;

tongue_x0  = bx1 + tongue_gap;
tongue_t   = throat_w - 2 * tongue_gap;
tongue_y0  = throat_y + tongue_gap;
tongue_top = throat_top + head_up;
spine_x    = tip_outer + spine_gap;

body_bot   = slot_cy - slot_h / 2 - wall;
body_top   = max(cap_top, tongue_top);

cheek_z    = body_len / 2 + cheek_gap + cheek_t / 2;
cup_or     = cup_id / 2 + ring_wall;
ring_cx    = spine_x + spine_t + cup_or - 3;

// =====================================================================
//  Helpers
// =====================================================================

module rrect(w, h, r) {
    rr = min(r, w / 2, h / 2);
    hull()
        for (sx = [-1, 1], sy = [-1, 1])
            translate([sx * (w / 2 - rr), sy * (h / 2 - rr)])
                circle(r = rr);
}

// Corners may be given in either order.
module box(x0, y0, x1, y1) {
    translate([min(x0, x1), min(y0, y1)])
        square([abs(x1 - x0), abs(y1 - y0)]);
}

// The bar itself, for fit checks and previews. Not a printed part.
module handle_bar(len = 200) {
    linear_extrude(height = len, center = true)
        rrect(handle_w, handle_h, handle_r);
}

// =====================================================================
//  The clip
// =====================================================================

// The vertical slot the bar rides in. Taller than the bar by
// slide_travel; the bar sits at the top of it once seated.
module slot_2d() {
    translate([0, slot_cy]) rrect(bore_w, slot_h, handle_r + fit);
}

// Cut only at the height the bar occupies while you are clicking it on.
// Once the clip is slid down, the bar is above this and held by solid
// wall -- which is what makes the interlock an interlock.
module mouth_2d() {
    box(far_x - 5, insert_y - mouth_h / 2, 0, insert_y + mouth_h / 2);
}

module jaw_2d() {
    difference() {
        offset(r = -fillet_r) offset(r = fillet_r) {
            translate([0, slot_cy])
                rrect(bore_w + 2 * wall, slot_h + 2 * wall,
                      handle_r + fit + wall);
            backbone_2d();
        }
        slot_2d();
        mouth_2d();
    }
}

// Inverted U straddling the top of the bar. Its far leg reaches down
// past the bar's shoulder by cap_engage, so lifting by slide_travel is
// the only way to get it off.
module cap_2d() {
    difference() {
        offset(r = -fillet_r) offset(r = fillet_r) {
            box(far_x, cap_bot, bx1, cap_top);
            backbone_2d();
        }
        slot_2d();
    }
}

module backbone_2d() { box(bx0, body_bot, bx1, body_top); }

// The attachment throat, running the full length of the backbone.
module throat_2d() {
    box(bx1 - 1, mnt_y,   tip_outer, throat_y);    // arm
    box(tip_inner, throat_y, tip_outer, throat_top); // upturned tip
}

module clip() {
    difference() {
        union() {
            translate([0, 0, jaw_z0]) linear_extrude(jaw_len) jaw_2d();
            translate([0, 0, cap_z0]) linear_extrude(cap_len) cap_2d();
            linear_extrude(height = body_len, center = true)
                offset(r = -fillet_r) offset(r = fillet_r) {
                    backbone_2d();
                    throat_2d();
                }
        }
        // The bar passes through the whole length, so the slot has to
        // be cleared along the backbone too, not just at the stations.
        linear_extrude(height = body_len + 2, center = true) slot_2d();
    }
}

// =====================================================================
//  Attachment interface
//
//  Call hook_mount() and put your geometry outboard of
//  spine_x + spine_t. The cheeks straddle the whole clip, so nothing
//  slides along the bar.
// =====================================================================

module tongue_2d() {
    difference() {
        box(tongue_x0, tongue_y0, tongue_x0 + tongue_t, tongue_top);
        translate([tongue_x0, tongue_y0])
            rotate([0, 0, 45]) square(fillet_r * 1.6, center = true);
        translate([tongue_x0 + tongue_t, tongue_y0])
            rotate([0, 0, 45]) square(fillet_r * 1.6, center = true);
    }
}

module yoke_2d(spine_bottom) {
    box(tongue_x0, tongue_top - crown_t, spine_x + spine_t, tongue_top);
    box(spine_x, spine_bottom, spine_x + spine_t, tongue_top);
}

module cheek_2d() { box(bx1 - 2, mnt_y, spine_x, tongue_top); }

module hook_mount(spine_bottom = ring_top - ring_h) {
    linear_extrude(height = tongue_len, center = true) tongue_2d();
    linear_extrude(height = 2 * cheek_z + cheek_t, center = true)
        yoke_2d(spine_bottom);
    for (s = [-1, 1])
        translate([0, 0, s * cheek_z])
            linear_extrude(height = cheek_t, center = true) cheek_2d();
}

// =====================================================================
//  Parts
// =====================================================================

module testfit() {
    hook_mount(spine_bottom = mnt_y - 14);
    difference() {
        translate([spine_x + spine_t - 2, mnt_y - 4, 0])
            rotate([0, 90, 0]) cylinder(h = 12, r = 9);
        translate([spine_x + spine_t - 3, mnt_y - 4, 0])
            rotate([0, 90, 0]) cylinder(h = 14, r = 4.5);
    }
}

module cup_ring() {
    translate([ring_cx, ring_top - ring_h, 0]) rotate([-90, 0, 0])
        difference() {
            cylinder(h = ring_h, r = cup_or);
            translate([0, 0, base_t]) cylinder(h = ring_h, r = cup_id / 2);
            translate([0, 0, -1]) cylinder(h = base_t + 2, r = drain_r);
        }
}

module cupholder() {
    hook_mount();
    cup_ring();
}

// Fit check only. Not for printing.
//   lift = 0            -> seated and locked
//   lift = slide_travel -> raised, jaw mouth lined up with the bar
module assembly(lift = 0) {
    upright() {
        translate([0, lift, 0]) {
            color("SteelBlue") clip();
            color("Goldenrod") cupholder();
        }
        %handle_bar();
    }
}

// Attachments are not simple extrusions, so each gets the orientation
// that suits it.
//   upright  — in-use +Y up the plate. The cup ring prints axis-vertical
//              and round; costs a little support under the tongue.
//   inverted — grows upward off the crown, no support at all, but only
//              works where there is no downward-facing cavity.
module upright()  { rotate([ 90, 0, 0]) children(); }
module inverted() { rotate([-90, 0, 0]) children(); }

echo(str("jaw springs ", handle_h - mouth_h, " mm to click on"));
echo(str("lift ", slide_travel, " mm to release; cap engages ",
         cap_engage, " mm"));
echo(str("clip is ", body_len, " mm along the bar -- it needs that much "
         , "straight run"));

if      (part == "clip")      clip();
else if (part == "cupholder") upright()  cupholder();
else if (part == "testfit")   inverted() testfit();
else if (part == "assembly")  assembly();
else if (part == "raised")    assembly(lift = slide_travel);
else if (part == "all")       { clip(); cupholder(); }
