// =====================================================================
//  Chicco Corso — corner caddy. One piece. Snaps onto the side arm,
//  slides down, and locks under the top arc.
//
//  World frame, standing behind the stroller:
//      +X   toward the middle of the handle
//      +Y   up
//      +Z   inboard, toward the seat. The cup hangs this way.
//  The handle's U lies in the XY plane. The side arm hangs from the
//  corner, leaning out by arm_tilt.
//
//  The top bar does NOT run along X. It turns at the corner and runs
//  perpendicular to the side arm, hook_rot off horizontal -- so the two
//  bars meet at a right angle, not the shallow one the first draft
//  assumed. The saddle follows that turn; the throat does not, because
//  the cup has to stay upright in world Y.
//
//  The hook sits at the ORIGIN on the arc. The claw sits at
//  (-nom_run, -nom_drop) on the arm. The model is drawn SEATED.
//
//  HOW IT WORKS
//    On:  hold it high so the hook clears the arc, snap the claw onto
//         the side arm, slide down until the hook seats on the arc.
//    Off: slide up until the hook lifts clear, then unsnap.
//
//    The hook is an inverted U resting on top of the arc. It carries
//    the weight and sets the height — sliding down stops when it
//    bottoms out, which is what keeps the caddy from slipping down the
//    arm. The claw is a light snap; because the arm is tilted, the claw
//    slides freely ALONG it, and that is where the travel comes from.
//
//  WHY IT LOCKS
//    Each grip blocks the direction the other releases in.
//      - The hook straddles the arc front-to-back, so front-to-back is
//        what it constrains: 0.4 mm of play.
//      - The claw's mouth points REARWARD, into that constraint. It
//        cannot open while the hook is seated.
//      - The claw wraps the arm side-to-side, stopping the caddy
//        sliding along the arc.
//      - Down is stopped by the hook resting on the bar.
//    Up is the only motion left, and that is the intended release.
//
//    Point the mouth any other way and this falls apart — outboard, and
//    the whole caddy pulls straight off sideways with the hook still
//    seated. The mouth direction IS the lock.
//
//    Neither snap carries load, so neither has to be tight.
//
//  ANGLES
//    One piece means the corner geometry is baked in, so nom_run,
//    nom_drop and arm_tilt have to be right. Two things make that
//    easier than it sounds:
//      - The claw slides along the arm, so error ALONG the arm just
//        parks the caddy slightly higher or lower. Self-correcting.
//        Only the PERPENDICULAR offset matters.
//      - A short claw tolerates angle error. Misalignment across the
//        bore is about claw_len * sin(error)/2, so at claw_len = 20 a
//        5-degree error is 0.9 mm, which the clearance absorbs.
//
//  PRINTING
//    Exported standing on the arm axis. The claw is then a true
//    vertical prism, the strut and throat are vertical walls, and the
//    throat opens upward. The only overhang is the hook's top plate,
//    which bridges the bar channel — a routine 24 mm bridge.
// =====================================================================

part = "caddy";  // [caddy, fused, cupholder, testfit, assembly, all]

$fn = 96;

// --- the two bars ----------------------------------------------------
// MEASURED on a Corso, 2026-08. One continuous 29 x 19 mm oval tube:
// bare on the side arm, wrapped in ~2-3 mm of rubber on the top arc
// (35 - 29 = 3 mm per side, 23 - 19 = 2 mm per side).
//
// The bend happens IN the handle plane, so the 19 mm axis stays
// front-to-back the whole way round, while the 29 mm axis rotates from
// vertical on the arc to side-to-side on the arm. That is why arc_h and
// arm_w are the same number wearing different hats.

arc_w    = 23;   // top arc, over the rubber: front to back
arc_h    = 35;   // top arc, over the rubber: top to bottom
arc_r    = 11.5; // stadium ends — half the short axis

arm_w    = 29;   // side arm, bare metal: across the handle plane
arm_h    = 19;   // side arm, bare metal: front to back
arm_r    = 9.5;  // stadium ends — half the short axis

fit      = 0.4;  // clearance to the bars
arm_slop = 0.35; // extra room in the claw, to absorb angle error
wall     = 4;

// --- corner geometry --- measured, see solve_corner.py ---------------
// From the spot on the arc where the hook goes, to the spot on the arm
// where the claw goes. Only the component perpendicular to the arm
// really matters; see ANGLES above.
//
// Read off the corner gauge: window 1 (35 mm down) edges at 18 and 66,
// window 2 (105 mm down) edges at 94 and 138. The two edges fitted
// separately give 47.35 and 45.81 degrees, so the marks are good to
// about a millimetre. Run the axis back to the arc's centreline and it
// lands 5 mm from the gauge's origin -- the gauge butts against the
// bend, so that near-zero is what says it was held in the handle's
// plane. It was 46.6, not the 12 this was first drawn around.
//
// The claw sits 75 mm down rather than further: the arm is steep, so
// sliding along it barely lifts the hook, and below about 61 mm of drop
// the caddy can no longer be taken off (checks.scad, liftoff). 75 keeps
// ~13 mm of margin on that while staying the shortest -- hence the
// stiffest -- strut that clears.

nom_run   = 84.3;  // horizontal, hook to claw
nom_drop  = 75;    // vertical,   hook to claw
arm_tilt  = 46.6;  // degrees the arm leans out from vertical

// The handle is ONE CONTINUOUS ARC, not two bars meeting at a corner.
// Radius from a tape laid across the bend: chord 200 mm, sagitta 37 mm,
// R = c^2/8s + s/2 = 155 mm.
//
// Everything about the hook now follows from that. The claw's position
// is fixed -- it fits, and it is not moving -- so the arc is pinned by
// the claw and its tangent, and the hook can only sit somewhere ON that
// arc. Its position AND the angle its saddle lies at are both derived.
// Neither is a dial to guess at, which is the whole point.
//
// hook_sweep is the one remaining choice: how far round the arc from
// the claw the hook sits. 35 deg puts the saddle about 8 deg above
// horizontal -- "angled slightly up", which is what the wide axis of
// the oval does there.
// The handle is STRAIGHT - 90 deg CORNER - STRAIGHT, not one gentle
// sweep. The part has to be the same shape: a leg down the arm, a full
// quarter turn, a leg along the top bar.
//
// This contradicts my own tape reading of R=155, and the tape loses. A
// 90 degree turn at R=155 spans 219 mm of chord, so a part of that shape
// could not be the 138 mm the last one was -- the two are not
// reconcilable, and the reading came from guessing where a tape touched
// in a photo. bend_r is the one dial here; everything else follows it.
bend_r     = 70;    // corner radius of the handle's turn
turn       = 90;    // degrees the handle turns through
leg_dn     = 38;    // straight run from the claw up to the corner
leg_up     = 46;    // straight run from the corner along the top bar
throat_len = 34;

// --- hook and claw ---------------------------------------------------

hook_len    = 34;   // along the top bar
hook_engage = 10;   // how far the legs reach down past the arc's top
claw_len    = 20;   // along the arm — short, for angle tolerance
claw_frac   = 0.90; // mouth width as a fraction of arm_w. The click.
strut_t     = 6;

// --- attachment interface -------------------------------------------

mnt_y      = -14;
mnt_t      = 5;
mnt_reach  = 15;
mnt_rise   = 15;
tongue_gap = 0.35;
tongue_len = 26;
crown_t    = 6;
spine_t    = 6;
spine_gap  = 0.6;
cheek_t    = 3;
cheek_gap  = 0.4;
head_up    = 8;

// --- sample cup holder ----------------------------------------------

cup_id     = 80;
ring_wall  = 3;
ring_h     = 45;
ring_top   = -34;
base_t     = 3;
drain_r    = 9;

// =====================================================================
//  Derived
// =====================================================================

// --- the path: leg, quarter turn, leg --------------------------------
// Built FROM THE CLAW, which fits and does not move: up the arm
// for leg_dn, a quarter turn of radius bend_r, then leg_up along the
// top bar. Turning right (clockwise) going up, so the centre sits to
// the right of the up-direction.
_claw    = [-nom_run, -nom_drop];
_u       = [sin(arm_tilt), cos(arm_tilt)];         // up the arm
_right   = [_u[1], -_u[0]];
_arc0    = [_claw[0] + leg_dn * _u[0], _claw[1] + leg_dn * _u[1]];
bend_c   = [_arc0[0] + bend_r * _right[0], _arc0[1] + bend_r * _right[1]];

ang_claw = atan2(_arc0[1] - bend_c[1], _arc0[0] - bend_c[0]);  // arc start
ang_hook = ang_claw - turn;                                    // arc end
hook_rot = ang_hook - 90;                          // tangent at the arc end
_t_up    = [cos(hook_rot), sin(hook_rot)];         // along the top bar
hook_pos = [bend_c[0] + bend_r * cos(ang_hook) + (leg_up - hook_len / 2) * _t_up[0],
            bend_c[1] + bend_r * sin(ang_hook) + (leg_up - hook_len / 2) * _t_up[1]];
saddle_deg = hook_len / bend_r * 180 / PI;
ang_mid    = (ang_hook + ang_claw) / 2;            // where the cup hangs

arc_top   = arc_h / 2 + fit;
hook_bot  = arc_top - hook_engage;
hook_out  = arc_w / 2 + fit + wall;

claw_bore = arm_w + 2 * (fit + arm_slop);   // across the handle plane
claw_deep = arm_h + 2 * (fit + arm_slop);   // front to back
claw_out  = claw_deep / 2 + wall;
claw_mouth = arm_w * claw_frac;

// The strut runs inboard of BOTH bars, so it crosses the corner without
// fouling either.
strut_z0  = arc_w / 2 + fit;
strut_z1  = strut_z0 + strut_t;

mnt_z0     = strut_z1;
throat_y   = mnt_y + mnt_t;
throat_top = throat_y + mnt_rise;
tip_inner  = mnt_z0 + mnt_reach - mnt_t;
tip_outer  = mnt_z0 + mnt_reach;
throat_w   = mnt_reach - mnt_t;

tongue_z0  = mnt_z0 + tongue_gap;
tongue_t   = throat_w - 2 * tongue_gap;
tongue_y0  = throat_y + tongue_gap;
tongue_top = throat_top + head_up;
spine_z    = tip_outer + spine_gap;
cheek_x    = hook_len / 2 + cheek_gap + cheek_t / 2;

cup_or     = cup_id / 2 + ring_wall;
cup_cz     = spine_z + spine_t + cup_or - 3;

// =====================================================================
//  Sanity. An undefined name silently becomes undef in OpenSCAD and
//  linear_extrude(undef) quietly builds something enormous instead of
//  failing, so these guard the values the geometry depends on.
// =====================================================================

assert(is_num(strut_z0) && is_num(claw_out) && is_num(hook_out),
       "core constants must all be numbers");
assert(strut_z0 >= arc_w / 2 + fit,  "strut would foul the top arc");
assert(strut_z0 > arm_h / 2,         "strut would foul the side arm");
assert(strut_z1 > claw_out - wall,   "strut misses the claw body");
assert(hook_bot < arc_top,           "hook legs grip nothing");
assert(claw_mouth < arm_w,           "claw mouth is wider than the arm");
assert(tongue_t > 0 && throat_w > 0, "throat is degenerate");

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

module box(x0, y0, x1, y1) {
    translate([min(x0, x1), min(y0, y1)])
        square([abs(x1 - x0), abs(y1 - y0)]);
}

// Everything belonging to the claw is drawn about the origin and moved
// onto the arm by this.
module at_arm() {
    translate([-nom_run, -nom_drop, 0]) rotate([0, 0, -arm_tilt])
        children();
}

// --- the handle, for fit checks. Not printed. ------------------------

// The handle is one tube, so it gets modelled as one tube. Sweeping a
// section of it between two angles on the bend.
//
// Splitting it into an "arc" and an "arm" that each ran past the other
// made both solids cover the same physical metal near the claw, and the
// clearance checks then reported the claw's legitimate grip as an
// interference. Two reference solids may not overlap.
module tube_arc(a0, a1, w, h, r) {
    translate([bend_c[0], bend_c[1], 0])
        rotate([0, 0, a0])
            rotate_extrude(angle = a1 - a0, $fn = 240)
                translate([bend_r, 0])
                    rotate(-90)
                        rrect(w, h, r);
}

// Rubber above, bare metal below, meeting midway between the two grips.
ang_trans = ang_claw - turn / 2;

module arc_bar(len = 300) {
    tube_arc(ang_hook, ang_trans, arc_w, arc_h, arc_r);
    // straight leg along the top bar. arc_h is the IN-PLANE size and
    // arc_w the front-to-back one, so they go in in that order.
    translate([bend_c[0] + bend_r * cos(ang_hook),
               bend_c[1] + bend_r * sin(ang_hook), 0])
        rotate([0, 0, ang_hook]) rotate([90, 0, 0])
            linear_extrude(leg_up + 30) rrect(arc_h, arc_w, arc_r);
}

module arm_bar(len = 240, up = 40) {
    // arm_h then arm_w, NOT the other way round. tube_arc's first size
    // is the front-to-back one, and that axis stays 19 mm the whole way
    // round the bend -- it is the 29 mm axis that rotates from vertical
    // on the arc to side-to-side on the arm. Passing them swapped makes
    // the arm 29 mm deep, which reaches z = 14.5 and eats the strut.
    tube_arc(ang_trans, ang_claw, arm_h, arm_w, arm_r);
    // straight leg down the arm, past the claw
    translate([bend_c[0] + bend_r * cos(ang_claw),
               bend_c[1] + bend_r * sin(ang_claw), 0])
        rotate([0, 0, ang_claw + 180]) rotate([90, 0, 0])
            linear_extrude(leg_dn + 90) rrect(arm_w, arm_h, arm_r);
}

// =====================================================================
//  The caddy — one piece
// =====================================================================

// Drawn in the arc's cross-section: local x -> world Z, local y -> world
// Y, extruded along the bar.
//
// The saddle and the throat are separate because they no longer point
// the same way. The top bar turns at the corner and runs perpendicular
// to the side arm, so the saddle has to lie along THAT, rotated by
// hook_rot -- while the throat carries the cup and has to stay upright
// in world Y or the drink tips. Extruding them together, as this did,
// forces one of the two to be wrong.
module saddle_profile() {
    box(-hook_out, hook_bot, hook_out, arc_top + wall);      // inverted U
}

module throat_profile() {
    // Web tying the throat back to the hook. It starts at strut_z0, not
    // inside it: any material within the arc's envelope is hidden by
    // the bar cut while seated, then fouls the bar on the way off.
    box(strut_z0, mnt_y, mnt_z0, arc_top + wall);
    box(mnt_z0 - 1, mnt_y, tip_outer, throat_y);             // throat arm
    box(tip_inner, throat_y - 1, tip_outer, throat_top);     // upturned tip
}

// Sweep a cross-section along the bend. rotate_extrude maps profile x
// to radius and profile y to Z, so the profile is transposed on the way
// in: local y (radially out) -> radius, local x (front-back) -> Z.
//
// A straight 34 mm saddle on a 155 mm radius stands 34^2/(8R) = 0.93 mm
// off the bar at its ends, against 0.4 mm of fit clearance. That is the
// rocking -- it is a curvature problem, not an angle problem, and no
// amount of rotating a STRAIGHT saddle fixes it.
module swept_on_arc(deg) {
    translate([bend_c[0], bend_c[1], 0])
        rotate([0, 0, ang_hook - deg / 2])
            rotate_extrude(angle = deg, $fn = 240)
                translate([bend_r, 0])
                    // (x,y) -> (y,-x). A transpose would be the obvious
                    // map, but its determinant is -1, which reverses the
                    // polygon's winding and makes rotate_extrude build
                    // the solid inside out. This rotation is det +1. The
                    // sign it puts on the axial axis is harmless because
                    // both profiles swept here are symmetric front-to-back.
                    rotate(-90)
                        children();
}

module hook_and_throat() {
    swept_on_arc(saddle_deg) saddle_profile();
    translate([bend_c[0] + bend_r * cos(ang_mid),
               bend_c[1] + bend_r * sin(ang_mid), 0])
        rotate([0, -90, 0]) linear_extrude(throat_len, center = true)
            throat_profile();
}

// A flat plate lying inboard of both bars, crossing the corner.
// A wedge of the bend, for trimming the band to its angular span.
module wedge_2d(r, a0, a1, steps = 48) {
    polygon(concat([[0, 0]],
        [for (i = [0 : steps]) let(a = a0 + (a1 - a0) * i / steps)
            [r * cos(a), r * sin(a)]]));
}

// The body is a BAND FOLLOWING THE BEND, not a hull.
//
// hull() was the bug, and it is a bug by definition: a convex hull is
// straight-sided, so no arrangement of blobs inside one can ever produce
// a curve. The old body cut a flat chord straight across the inside of
// the bend while the handle curved away from it. Only the 34 mm saddle
// touched.
//
// Swept between the two grips instead, and carried inboard of the tube
// so the cup hangs in the crook of the curve.
band_out = 20;   // radially outboard of the tube's centreline
band_in  = 34;   // radially inboard, toward the cup
band_pad = 7;    // angular overrun past each grip

// Straight leg, quarter turn, straight leg -- the handle's actual shape.
// A bare sector (what this was) is all corner and no legs, which reads
// as a gentle sweep rather than a turn.
// A straight run of band leaving the arc at angle a. rotate(a+90) puts
// local +X on the counter-clockwise tangent and local +Y pointing in
// toward the bend's centre, so the band spans -band_out .. +band_in.
// dirn = +1 runs counter-clockwise, -1 clockwise.
module leg_2d(a, len, dirn) {
    translate([bend_c[0] + bend_r * cos(a), bend_c[1] + bend_r * sin(a)])
        rotate(a + 90)
            translate([dirn > 0 ? 0 : -len, -band_out])
                square([len, band_out + band_in]);
}

module strut_2d() {
    // the corner
    translate(bend_c)
        intersection() {
            difference() {
                circle(r = bend_r + band_out);
                circle(r = bend_r - band_in);
            }
            wedge_2d(bend_r + band_out + 5, ang_hook, ang_claw);
        }
    // Going claw -> hook the angle DECREASES, so running back toward the
    // claw is counter-clockwise and running on past the hook is clockwise.
    leg_2d(ang_claw, leg_dn + band_pad,  1);
    leg_2d(ang_hook, leg_up + band_pad, -1);
}

module strut() {
    translate([0, 0, strut_z0]) linear_extrude(strut_t) strut_2d();
}

// Claw, drawn in the arm's cross-section: local x -> world X (across the
// handle plane), local y -> world -Z (front to back).
module claw_profile() {
    rrect(claw_bore + 2 * wall, claw_deep + 2 * wall, arm_r + fit + wall);
}

module claw_cut_profile() {
    rrect(claw_bore, claw_deep, arm_r + fit);
    // Mouth opens REARWARD. See WHY IT LOCKS: that is the one direction
    // the hook constrains, so it is the only direction it may face.
    box(-claw_mouth / 2, 0, claw_mouth / 2, claw_deep);
}

module claw() {
    at_arm() rotate([-90, 0, 0])
        linear_extrude(claw_len, center = true) claw_profile();
}

module claw_cut(over = 2) {
    at_arm() rotate([-90, 0, 0])
        linear_extrude(claw_len + over, center = true) claw_cut_profile();
}

module caddy() {
    difference() {
        union() { hook_and_throat(); strut(); claw(); }
        swept_on_arc(saddle_deg + 4)
            rrect(arc_w + 2 * fit, arc_h + 2 * fit, arc_r + fit);
        claw_cut();
    }
}

// =====================================================================
//  Attachment interface
//
//  Call hook_mount() and put your geometry outboard of
//  spine_z + spine_t.
// =====================================================================

module tongue_profile() {
    difference() {
        box(tongue_z0, tongue_y0, tongue_z0 + tongue_t, tongue_top);
        for (s = [0, 1])
            translate([tongue_z0 + s * tongue_t, tongue_y0])
                rotate([0, 0, 45]) square(3.2, center = true);
    }
}

module yoke_profile(spine_bottom) {
    box(tongue_z0, tongue_top - crown_t, spine_z + spine_t, tongue_top);
    box(spine_z, spine_bottom, spine_z + spine_t, tongue_top);
}

module cheek_profile() { box(mnt_z0 + 0.4, mnt_y, spine_z, tongue_top); }

module hook_mount(spine_bottom = ring_top - ring_h) {
    rotate([0, -90, 0]) {
        linear_extrude(tongue_len, center = true) tongue_profile();
        linear_extrude(2 * cheek_x + cheek_t, center = true)
            yoke_profile(spine_bottom);
        for (s = [-1, 1])
            translate([0, 0, s * cheek_x])
                linear_extrude(cheek_t, center = true) cheek_profile();
    }
}

// =====================================================================
//  Parts
// =====================================================================

module testfit() {
    hook_mount(spine_bottom = mnt_y - 16);
    difference() {
        translate([0, mnt_y - 6, spine_z + spine_t - 2]) cylinder(h = 12, r = 9);
        translate([0, mnt_y - 6, spine_z + spine_t - 3]) cylinder(h = 14, r = 4.5);
    }
}

module cup_ring() {
    translate([0, ring_top - ring_h, cup_cz]) rotate([-90, 0, 0])
        difference() {
            cylinder(h = ring_h, r = cup_or);
            translate([0, 0, base_t]) cylinder(h = ring_h, r = cup_id / 2);
            translate([0, 0, -1]) cylinder(h = base_t + 2, r = drain_r);
        }
}

module cupholder() { hook_mount(); cup_ring(); }

// =====================================================================
//  Fused variant — cup welded straight onto the caddy, no drop-in
//  interface.
//
//  Worth it if you never intend to swap attachments: without the throat
//  and tongue in the way the cup moves ~30 mm closer to the handle,
//  which is 30 mm off the lever arm the hook has to carry.
//
//  To use YOUR OWN cup holder instead of the ring below, set
//  external_cup to its path:
//
//      openscad -o fused.stl -D 'part="fused"' \
//               -D 'external_cup="my_cup.stl"' \
//               -D 'cup_rot=[90,0,0]' -D 'cup_pos=[0,-30,40]' \
//               stroller_clip.scad
//
//  The STL must be a closed mesh in millimetres. cup_rot is applied
//  first, then cup_pos, both in the world frame above: +X along the
//  arc, +Y up, +Z inboard. Aim to bury it a few mm into the bracket so
//  the union is a real weld rather than two solids touching.
// =====================================================================

// Your own cup holder, instead of the ring below. Rotation is applied
// as: spin about the cup's own axis first, then cup_rot, then cup_pos.
// Two stages because getting a cup upright AND facing the handle needs
// rotations about two different axes, and a single rotate() vector
// applies X, then Y, then Z — the wrong order for this.
//
// The STL must be a closed mesh in millimetres. Bury it a few mm into
// the bracket so the union is a real weld, not two solids touching.

external_cup = "";          // e.g. "vendor/PriamCupHolderV3.stl"
cup_spin     = 180;         // about the cup's own axis, applied first
cup_rot      = [-90, 0, 0]; // stands a Z-up cup upright in this frame
cup_pos      = [0, -36, 12.9];

// --- built-in ring, used when external_cup is empty ------------------

fused_top    = -20;
fused_cz     = mnt_z0 + cup_or - 4;
fused_inner  = fused_cz - cup_id / 2;

module fused_ring() {
    translate([0, fused_top - ring_h, fused_cz]) rotate([-90, 0, 0])
        difference() {
            cylinder(h = ring_h, r = cup_or);
            translate([0, 0, base_t]) cylinder(h = ring_h, r = cup_id / 2);
            translate([0, 0, -1]) cylinder(h = base_t + 2, r = drain_r);
        }
}

// --- the weld --------------------------------------------------------
// A slab tying the caddy's inboard face into the cup's mounting boss,
// so the two meet across a real area rather than at a tangent. Sized to
// suit either cup.

brk_x  = 15;    // half width, along the arc
brk_y0 = -64;   // bottom
brk_y1 = -8;    // top
brk_z1 = 22;    // how far inboard it reaches

module fused_bracket() {
    translate([-brk_x, brk_y0, strut_z0])
        cube([2 * brk_x, brk_y1 - brk_y0, brk_z1 - strut_z0]);
}

module fused_cup() {
    if (external_cup == "") fused_ring();
    else translate(cup_pos) rotate(cup_rot) rotate([0, 0, cup_spin])
             import(external_cup);
}

module fused() {
    union() {
        caddy();
        fused_bracket();
        fused_cup();
    }
}

// Fit check only. `lift` slides the caddy up the arm, the way it comes
// off. Not for printing.
module assembly(lift = 0) {
    translate([lift * sin(arm_tilt), lift * cos(arm_tilt), 0]) {
        color("SteelBlue") caddy();
        color("Goldenrod") cupholder();
    }
    %arc_bar();
    %arm_bar();
}

// Stand it on the arm axis: the claw becomes a true vertical prism and
// the throat opens upward.
module lay() { rotate([90, 0, 0]) rotate([0, 0, arm_tilt]) children(); }

echo(str("bend centre ", bend_c, "  hook at ", hook_pos));
echo(str("saddle lies ", -hook_rot, " deg off horizontal, sweeping ",
         saddle_deg, " deg of arc"));
echo(str("claw springs ", arm_w - claw_mouth, " mm to click on"));
echo(str("hook allows ", 2 * fit, " mm of front-to-back play when seated"));
echo(str("angle slack ", claw_len * sin(5) / 2,
         " mm across the bore at 5 degrees of error"));

if      (part == "caddy")     lay() caddy();
else if (part == "fused")     lay() fused();
else if (part == "cupholder") cupholder();
else if (part == "testfit")   testfit();
else if (part == "assembly")  assembly();
else if (part == "raised")    assembly(lift = 26);
else if (part == "all")       { caddy(); cupholder(); }
