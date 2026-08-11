// =====================================================================
//  Chicco Corso — handle clip with top hook, and hook-mounted
//  attachments.
//
//  Coordinate convention, as fitted to the stroller:
//      +X   outboard — the direction the hook points, away from the
//           person pushing
//      +Y   up
//      +Z   along the handle axis
//
//  ---------------------------------------------------------------
//  !! THE HANDLE DIMENSIONS BELOW ARE UNMEASURED PLACEHOLDERS.     !!
//  !! Print gauge.scad, measure the handle in both directions, and !!
//  !! set handle_w / handle_h / handle_r before printing the clip. !!
//  ---------------------------------------------------------------
//
//  Why the clip is a flat profile extruded along Z: printed standing
//  on end (handle axis vertical) every feature comes out without
//  support, the band's hoop stress runs along the extrusion lines
//  rather than across layers, and the in-use load on the hook lies in
//  the layer plane instead of trying to peel layers apart.
// =====================================================================

part = "clip";   // [clip, cupholder, testfit, all]

$fn = 96;

// --- handle ----------------------------------------------------------
// Modelled as a rounded rectangle. A round handle is just the case
// where handle_w == handle_h == 2 * handle_r.

handle_w    = 30;    // front-to-back, mm   *** PLACEHOLDER ***
handle_h    = 26;    // top-to-bottom, mm   *** PLACEHOLDER ***
handle_r    = 11;    // corner radius, mm   *** PLACEHOLDER ***

// The grip is textured rubber, so the bore is deliberately undersized:
// the clip squeezes into the rubber instead of sliding on it. Raise for
// more grip, lower if it will not go on. 0 = neutral fit.
rubber_bite = 0.5;

// --- clamp band ------------------------------------------------------

band_wall   = 4;     // wall thickness around the handle
clamp_len   = 30;    // length along the handle (= extrusion depth)
// Opening width as a fraction of handle_h. Below 1.0 the band wraps
// past halfway and self-retains. This is the main thing to tune, and
// it is a real trade-off: the band has to spring open by
// (handle_h - mouth_w) to get over the handle at all. At 0.86 that is
// ~3.6 mm total, which PETG will take and PLA may crack at. Go lower
// only if you are running without the pinch bolt.
mouth_frac  = 0.86;
fillet_r    = 2;     // fillet where the tower meets the band

// --- tower and hook --------------------------------------------------

tower_t     = 5;     // thickness of the pillar outboard of the band
tower_bottom = -14;  // how far the pillar runs below the handle centre
hook_y      = 8;     // underside of the hook arm
hook_t      = 5;     // hook stock thickness
hook_reach  = 16;    // how far the arm reaches outboard
hook_rise   = 16;    // height of the upturned tip = throat depth

// --- pinch bolt ------------------------------------------------------
// Optional. Rubber takes a compression set over time, so a snap fit
// that feels tight today can be loose in a month; the bolt lets you
// take that up. Set false for a hardware-free snap-on clip.

use_bolt      = true;
ear_len       = 8;    // how far the ears project past the band
ear_t         = 6;    // ear thickness
ear_overlap   = 6;    // how far the ears reach back into the band
bolt_clear_r  = 1.75; // M3 clearance, near ear
bolt_pilot_r  = 1.35; // M3 self-tapping pilot, far ear

// --- attachment interface -------------------------------------------
// Attachments drop straight down over the hook: a tongue into the
// throat, a crown over the top, a spine down the outboard face, and
// two cheeks straddling the clip so nothing slides along the handle.

fit_gap     = 0.35;  // clearance per side between tongue and throat
crown_t     = 6;     // thickness of the plate that bridges the top
spine_t     = 6;     // thickness of the outboard spine
spine_gap   = 0.6;   // clearance between spine and the hook tip
cheek_t     = 3;     // cheek thickness
cheek_gap   = 0.4;   // clearance per side between cheeks and the clip
head_up     = 8;     // how far the tongue rises above the hook tip

// --- sample cup holder ----------------------------------------------

cup_id      = 80;    // inside diameter of the ring
ring_wall   = 3;
ring_h      = 45;
ring_top    = 8;     // height of the ring's rim, in stroller coords
base_t       = 3;
drain_r      = 9;

// =====================================================================
//  Derived geometry
// =====================================================================

bore_w = handle_w - 2 * rubber_bite;
bore_h = handle_h - 2 * rubber_bite;
bore_r = max(0.5, handle_r - rubber_bite);

band_w = bore_w + 2 * band_wall;
band_h = bore_h + 2 * band_wall;
band_r = bore_r + band_wall;

mouth_w = handle_h * mouth_frac;

face_x     = band_w / 2 + tower_t;                 // outboard face of the pillar
tip_inner  = face_x + hook_reach - hook_t;         // inboard face of the upturned tip
tip_outer  = face_x + hook_reach;
throat_w   = hook_reach - hook_t;                  // gap you hang things in
throat_y   = hook_y + hook_t;                      // floor of the throat
hook_top   = throat_y + hook_rise;

tower_x0   = band_w / 2 - 4;                       // pillar overlaps the band

// Attachment side
tongue_x0  = face_x + fit_gap;
tongue_t   = throat_w - 2 * fit_gap;
tongue_y0  = throat_y + fit_gap;
tongue_top = hook_top + head_up;
spine_x    = tip_outer + spine_gap;
mount_w_outer = clamp_len + 2 * (cheek_gap + cheek_t);
cup_or     = cup_id / 2 + ring_wall;
ring_cx    = spine_x + spine_t + cup_or - 3;       // 3 mm merge into the spine

// =====================================================================
//  Helpers
// =====================================================================

// Rounded rectangle centred on the origin.
module rrect(w, h, r) {
    rr = min(r, w / 2, h / 2);
    hull()
        for (sx = [-1, 1], sy = [-1, 1])
            translate([sx * (w / 2 - rr), sy * (h / 2 - rr)])
                circle(r = rr);
}

// Rectangle given opposite corners, so the profile code below can be
// read straight off the coordinate convention.
// Corners may be given in either order -- the mirrored ear pair below
// relies on that.
module box(x0, y0, x1, y1) {
    translate([min(x0, x1), min(y0, y1)])
        square([abs(x1 - x0), abs(y1 - y0)]);
}

// =====================================================================
//  The clip
// =====================================================================

module band_2d()  { rrect(band_w, band_h, band_r); }
module bore_2d()  { rrect(bore_w, bore_h, bore_r); }

module tower_2d() { box(tower_x0, tower_bottom, face_x, hook_top); }

module hook_2d() {
    box(face_x - 1, hook_y,    tip_outer, throat_y);   // arm
    box(tip_inner,  throat_y,  tip_outer, hook_top);   // upturned tip
}

// Opening faces inboard (-X), towards the person pushing, so the clip
// is pushed on from behind and the hanging load never pulls in line
// with the gap.
module mouth_2d() { box(-band_w, -mouth_w / 2, 0, mouth_w / 2); }

module ears_2d() {
    for (s = [-1, 1])
        box(-(band_w / 2 + ear_len), s * mouth_w / 2,
            -(band_w / 2 - ear_overlap), s * (mouth_w / 2 + ear_t));
}

module clip_profile() {
    difference() {
        // Fillet the concave corners where the added shapes meet, then
        // cut the holes, so the bore keeps its exact size.
        offset(r = -fillet_r) offset(r = fillet_r) {
            band_2d();
            tower_2d();
            hook_2d();
            if (use_bolt) ears_2d();
        }
        bore_2d();
        mouth_2d();
    }
}

module bolt_holes() {
    bx = -(band_w / 2 + ear_len / 2);
    bz = 0;
    span = mouth_w + 2 * ear_t + 2;

    // Pilot straight through both ears...
    translate([bx, -(mouth_w / 2 + ear_t + 1), bz])
        rotate([-90, 0, 0]) cylinder(h = span, r = bolt_pilot_r);
    // ...opened out to clearance in the near ear only, so the bolt
    // pulls the two ears together instead of just spinning.
    translate([bx, mouth_w / 2 - 0.1, bz])
        rotate([-90, 0, 0]) cylinder(h = ear_t + 0.2, r = bolt_clear_r);
}

// Centred on z = 0, the same datum the attachments use, so the two
// halves line up along the handle.
module clip() {
    difference() {
        linear_extrude(height = clamp_len, center = true) clip_profile();
        if (use_bolt) bolt_holes();
    }
}

// =====================================================================
//  Attachment interface
//
//  Include this in any attachment you design. Everything outboard of
//  spine_x + spine_t is yours.
// =====================================================================

// Tongue, with its lower corners clipped so it clears the fillets in
// the bottom of the throat.
module tongue_2d() {
    difference() {
        box(tongue_x0, tongue_y0, tongue_x0 + tongue_t, tongue_top);
        translate([tongue_x0, tongue_y0])
            rotate([0, 0, 45]) square(fillet_r * 1.6, center = true);
        translate([tongue_x0 + tongue_t, tongue_y0])
            rotate([0, 0, 45]) square(fillet_r * 1.6, center = true);
    }
}

// Crown over the top of the hook, and the spine down its outboard
// face. Extruded to the full outer width so the cheeks land on it.
module yoke_2d(spine_bottom) {
    box(tongue_x0, tongue_top - crown_t, spine_x + spine_t, tongue_top);
    box(spine_x, spine_bottom, spine_x + spine_t, tongue_top);
}

// Reaches back inboard alongside the clip, locating the attachment
// along the handle.
module cheek_2d() { box(face_x - 10, hook_y, spine_x, tongue_top); }

// spine_bottom: how far down the outboard face the spine runs, in
// stroller coordinates. Your attachment hangs off it below the hook.
module hook_mount(spine_bottom = ring_top - ring_h) {
    linear_extrude(height = clamp_len - 1, center = true) tongue_2d();
    linear_extrude(height = mount_w_outer, center = true) yoke_2d(spine_bottom);
    for (s = [-1, 1])
        translate([0, 0, s * (clamp_len / 2 + cheek_gap + cheek_t / 2)])
            linear_extrude(height = cheek_t, center = true) cheek_2d();
}

// =====================================================================
//  Parts
// =====================================================================

// Minimal print to prove the interface fits before committing to a
// full-size attachment. ~15 min.
module testfit() {
    hook_mount(spine_bottom = ring_top - 14);
    difference() {
        translate([spine_x + spine_t - 2, ring_top - 4, 0])
            rotate([0, 90, 0]) cylinder(h = 12, r = 9);
        translate([spine_x + spine_t - 3, ring_top - 4, 0])
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

// Attachments are not extrusions, so each one gets the orientation
// that suits it.
//
// upright:   in-use +Y -> +Z. The cup ring prints axis-vertical, so it
//            comes out round and its base is on the plate. Costs a
//            little support under the tongue, which hangs in mid-air.
// inverted:  in-use +Y -> -Z. Everything grows upward off the crown,
//            so it needs no support at all -- but only works for
//            attachments with no downward-facing cavity.
module upright()  { rotate([ 90, 0, 0]) children(); }
module inverted() { rotate([-90, 0, 0]) children(); }

// Fit check only -- both parts in their as-fitted positions, so you
// can see the tongue sitting in the throat. Not for printing.
module assembly() {
    upright() {
        color("SteelBlue")  clip();
        color("Goldenrod")  cupholder();
        // stand-in for the handle
        %linear_extrude(height = clamp_len * 3.4, center = true)
            rrect(handle_w, handle_h, handle_r);
    }
}

if      (part == "clip")      clip();
else if (part == "cupholder") upright()  cupholder();
else if (part == "testfit")   inverted() testfit();
else if (part == "assembly")  assembly();
else if (part == "all")       { clip(); cupholder(); }
