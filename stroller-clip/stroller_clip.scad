// =====================================================================
//  Chicco Corso — corner caddy. Hooks over the top arc, clips to the
//  side arm, carries a cup inboard.
//
//  World frame, standing behind the stroller:
//      +X   along the top arc, toward the middle of the handle
//      +Y   up
//      +Z   inboard, toward the seat. The cup hangs this way.
//  The handle's U lies in the XY plane. The arc runs along X; the side
//  arm hangs down from the corner, leaning out by arm_tilt.
//
//  The hook sits at the ORIGIN, on the arc. The claw sits at
//  (-nom_run, -nom_drop), on the arm.
//
//  ---------------------------------------------------------------
//  !! EVERY HANDLE DIMENSION BELOW IS AN UNMEASURED PLACEHOLDER.   !!
//  ---------------------------------------------------------------
//
//  HOW IT WORKS
//    The hook is an inverted U resting on top of the arc. It carries
//    the weight and sets the height: sliding the caddy down the arm
//    stops when the hook bottoms out on the arc. That is what keeps
//    the side clip from slipping down.
//
//    The claw is a light snap around the side arm. Because the arm is
//    tilted, the claw slides freely ALONG it, and that is where the
//    up-and-down travel comes from. No slot needed anywhere.
//
//    On:  hold it high so the hook clears the arc, click the claw onto
//         the arm, slide down until the hook seats.
//    Off: slide up until the hook lifts clear, then unclick.
//
//    It locks because the two grips wrap bars that are not parallel.
//    With the hook down over the arc, the caddy cannot translate away
//    from the arm to release the claw without the hook binding on the
//    arc. Neither snap carries load, so neither has to be tight.
//
//  WHY TWO PIECES
//    The hook is a prism about a bar running along X; the claw is a
//    prism about one running roughly along Y. Prisms 90 degrees apart
//    cannot both print support-free in one piece. Splitting them also
//    makes the corner adjustable, which matters because the bend is
//    the one thing that is genuinely hard to measure: the slotted
//    joint lets you set it on the stroller instead.
//
//  PRINTING
//    Each piece is exported standing on its own bar axis. No supports.
// =====================================================================

part = "hook";   // [hook, claw, cupholder, testfit, assembly, all]

$fn = 96;

// --- the two bars ----------------------------------------------------
// Each measured perpendicular to its own axis.

arc_w    = 30;   // top arc: width, measured along Z (front to back)  ***
arc_h    = 26;   // top arc: height, top to bottom                    ***
arc_r    = 11;   // corner radius                                     ***

arm_w    = 28;   // side arm: width in the handle plane               ***
arm_h    = 28;   // side arm: depth, front to back                    ***
arm_r    = 14;   // corner radius (= w/2 if round)                    ***

fit      = 0.4;  // clearance to the bars
wall     = 4;

// --- corner geometry -------------------------------------------------
// Nominal only — the slotted joint absorbs the error, so these just
// need to be close. Measure roughly, with a tape, from the spot on the
// arc where you want the hook to the spot on the arm where you want
// the claw.

nom_run   = 70;  // horizontal, hook to claw
nom_drop  = 90;  // vertical,   hook to claw
arm_tilt  = 12;  // degrees the arm leans out from vertical

// --- hook (over the arc) ---------------------------------------------

hook_len    = 34;  // along the arc
hook_engage = 10;  // how far the legs reach down past the shoulder

// --- claw (around the arm) -------------------------------------------

claw_len   = 30;   // along the arm
claw_frac  = 0.90; // mouth width as a fraction of arm_w. The click.
                   // It holds nothing but itself, so it stays light.

// --- adjustable joint ------------------------------------------------
// Two flat pads meeting face to face in a plane parallel to the handle
// plane, clear of both bars. Slots run at right angles to each other,
// giving two axes of adjustment plus a few degrees of swivel.

joint_t     = 5;    // thickness of each pad
slot_len    = 18;   // adjustment range per axis
joint_pitch = 26;   // bolt spacing
bolt_r      = 1.75; // M3 clearance

// --- attachment interface -------------------------------------------

mnt_y      = -14;   // underside of the throat arm
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

arc_top   = arc_h / 2 + fit;
hook_bot  = arc_top - hook_engage;
hook_out  = arc_w / 2 + fit + wall;      // outer face of the hook legs

claw_bore = arm_w + 2 * fit;
claw_deep = arm_h + 2 * fit;
claw_out  = claw_deep / 2 + wall;        // how far the claw stands off in Z
claw_mouth = arm_w * claw_frac;

// The joint plane has to clear the widest thing near it, which is the
// claw body, not the bars.
claw_pad_z0 = claw_out - 3;              // buried in the claw body
joint_gap   = 0.2;                       // clearance in the lap joint
joint_z     = claw_out + 1.6;            // the mating face
strut_z1    = joint_z + joint_t;         // top of the hook's strut

mnt_z0     = strut_z1;                   // throat root, outboard of it all
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

// Where the joint sits, and which way the strut runs.
joint_ang  = atan2(-nom_drop, -nom_run);

// --- sanity ----------------------------------------------------------
// An undefined name silently becomes undef in OpenSCAD, and
// linear_extrude(undef) quietly builds something enormous rather than
// failing. These catch that, and the geometry mistakes that look fine
// in preview.

assert(is_num(joint_gap) && is_num(joint_z) && is_num(claw_pad_z0),
       "joint constants must all be numbers");
assert(claw_pad_z0 < joint_z - joint_gap,
       "claw pad has no thickness -- check claw_out and joint_z");
assert(joint_z > claw_out,
       "the joint plane is inside the claw body");
assert(joint_pad_r >= slot_len / 2 + bolt_r + 2,
       "joint pad is too small to contain its slot at full travel");
assert(hook_bot < arc_top,
       "hook_engage is too small for the legs to grip anything");
assert(claw_mouth < arm_w,
       "claw mouth is wider than the arm -- it would not click on");
assert(tongue_t > 0 && throat_w > 0,
       "throat is degenerate -- check mnt_reach against mnt_t");

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

// Round-ended slot, so a bolt can slide along it.
module slot(len, r) {
    hull() for (s = [-1, 1]) translate([s * len / 2, 0]) circle(r = r);
}

// The pair of bolt slots at the joint, lying in the XY plane, centred
// on the joint and elongated along `ang`.
module joint_slots(ang) {
    for (s = [-1, 1])
        translate([-nom_run, -nom_drop] +
                  s * joint_pitch / 2 * [cos(joint_ang + 90),
                                         sin(joint_ang + 90)])
            rotate([0, 0, ang]) slot(slot_len, bolt_r);
}

// --- the handle, for fit checks. Not a printed part. -----------------

module arc_bar(len = 300) {
    rotate([0, 90, 0]) linear_extrude(len, center = true)
        rrect(arc_w, arc_h, arc_r);
}

// Hangs from the corner downward only, so the preview does not show it
// running up through the arc.
module arm_bar(len = 240, up = 40) {
    translate([-nom_run, -nom_drop, 0]) rotate([0, 0, -arm_tilt])
        translate([0, up, 0]) rotate([90, 0, 0])
            linear_extrude(len) rrect(arm_w, arm_h, arm_r);
}

// =====================================================================
//  Piece 1 — hook, strut and attachment throat
//
//  The hook and throat are one prism about the arc. The strut is a flat
//  plate lying parallel to the handle plane, outboard of both bars, so
//  it reaches diagonally down to the joint without fouling anything.
// =====================================================================

// Cross-section of the arc bar: local x is world Z, local y is world Y.
module arc_section(grow = 0) {
    rrect(arc_w + 2 * fit + 2 * grow, arc_h + 2 * fit + 2 * grow,
          arc_r + fit + grow);
}

module hook_profile() {
    // inverted U over the bar
    box(-hook_out, hook_bot, hook_out, arc_top + wall);
    // throat for attachments, hung off the outboard side
    box(mnt_z0 - 1,  mnt_y,    tip_outer, throat_y);
    box(tip_inner,   throat_y, tip_outer, throat_top);
    // web tying the throat back to the hook
    box(hook_out - 2, mnt_y, mnt_z0, arc_top + wall);
}

// Big enough to contain both slots at full travel, with meat left over.
joint_pad_r = slot_len / 2 + bolt_r + 4;

module joint_pad_2d() {
    hull() for (s = [-1, 1])
        translate([-nom_run, -nom_drop] +
                  s * joint_pitch / 2 * [cos(joint_ang + 90),
                                         sin(joint_ang + 90)])
            circle(r = joint_pad_r);
}

module strut_2d() {
    hull() { translate([0, -6]) circle(r = 15); joint_pad_2d(); }
}

module hook_part() {
    difference() {
        union() {
            rotate([0, -90, 0]) linear_extrude(hook_len, center = true)
                hook_profile();
            translate([0, 0, joint_z]) linear_extrude(joint_t) strut_2d();
        }
        // the bar itself
        rotate([0, -90, 0]) linear_extrude(hook_len + 2, center = true)
            arc_section();
        // slots, elongated along the strut
        translate([0, 0, joint_z - 1])
            linear_extrude(joint_t + 2) joint_slots(joint_ang);
    }
}

// =====================================================================
//  Piece 2 — claw
//
//  A prism about the arm. Drawn upright at the origin of its own frame
//  (local x -> world X, local y -> world -Z, extruded along world Y),
//  then tilted and moved onto the arm.
// =====================================================================

module claw_profile() {
    difference() {
        rrect(claw_bore + 2 * wall, claw_deep + 2 * wall, arm_r + fit + wall);
        rrect(claw_bore, claw_deep, arm_r + fit);
        // mouth opens outboard, away from the stroller, so the caddy is
        // pushed on from outside and the load never pulls in line with
        // the gap
        box(-claw_bore, -claw_mouth / 2, 0, claw_mouth / 2);
    }
}

// The pad is buried in the claw body and stands proud of it to form the
// mating face. Drawn in the XY plane, at the joint's Z.
module claw_pad_2d() { translate([nom_run, nom_drop]) joint_pad_2d(); }

module claw_part() {
    difference() {
        union() {
            rotate([-90, 0, 0])
                linear_extrude(claw_len, center = true) claw_profile();
            // pad is buried in the claw body and stands proud of it to
            // form the mating face
            translate([0, 0, claw_pad_z0])
                linear_extrude(joint_z - joint_gap - claw_pad_z0)
                    claw_pad_2d();
        }
        translate([0, 0, claw_pad_z0 - 1])
            linear_extrude(joint_z - claw_pad_z0 + 2)
                translate([nom_run, nom_drop]) joint_slots(joint_ang + 90);
    }
}

// Placed onto the arm, in world coordinates.
module claw_placed() {
    translate([-nom_run, -nom_drop, 0]) rotate([0, 0, -arm_tilt])
        claw_part();
}

// =====================================================================
//  Attachment interface
//
//  Call hook_mount() and put your geometry outboard of
//  spine_z + spine_t. The cheeks straddle the hook along the arc.
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

// Fit check only. `lift` slides the caddy up the arm, the way it comes
// off. Not for printing.
module assembly(lift = 0) {
    translate([lift * sin(arm_tilt), lift * cos(arm_tilt), 0]) {
        color("SteelBlue") hook_part();
        color("SlateGray") claw_placed();
        color("Goldenrod") cupholder();
    }
    %arc_bar();
    %arm_bar();
}

// Each piece stands on its own bar axis for printing.
module lay_hook() { rotate([0, -90, 0]) children(); }
module lay_claw() { rotate([-90, 0, 0]) children(); }

echo(str("claw springs ", arm_w - claw_mouth, " mm to click on"));
echo(str("joint adjusts +/-", slot_len / 2, " mm on each axis"));
echo(str("joint plane at z = ", joint_z, ", clear of both bars"));

if      (part == "hook")      lay_hook() hook_part();
else if (part == "claw")      lay_claw() claw_part();
else if (part == "cupholder") cupholder();
else if (part == "testfit")   testfit();
else if (part == "assembly")  assembly();
else if (part == "raised")    assembly(lift = 26);
else if (part == "all")       { hook_part(); claw_placed(); cupholder(); }
