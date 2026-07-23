// ===========================================================================
//  Apple Watch "Pocket Carry" Case  —  parametric, 3D-printable   (v2, sleek)
//  Adapted for the 1st-generation Apple Watch (Series 0: 38 mm / 42 mm)
//
//  Inspired by the screwed-sandwich pocket fob in the reference video
//  (an Apple Watch Ultra brick).  This version is re-dimensioned for the
//  smaller, curved-back Series 0 body and is fully parametric so you can
//  tune it to YOUR watch and YOUR printer.
//
//  v2 ergonomic pass:
//    * all outer edges filleted (pebble feel, no sharp corners in the palm)
//    * slimmer top/bottom blocks  ->  ~4 mm shorter overall
//    * finger scallop around the Digital Crown opening
//    * lanyard hole moved inboard (v1 left only ~0.5 mm of rim below it)
//  v3:
//    * hex-head bolts now FULLY EMBEDDED: the hex pockets swallow the whole
//      head (~0.4 mm sub-flush), so the bolt face lays dead flat on a table.
//      The keyed hex seat still locks the heads against self-loosening.
//      Backplate thickened 3 -> 4 mm to keep a strong web under the heads.
//
//  TWO PRINTED PARTS + hardware:
//    * bezel      : front tray. Watch drops in face-first; a lip around the
//                   screen window retains it. Right wall has openings for the
//                   Digital Crown and the side button. Lanyard hole at bottom.
//    * backplate  : bolts onto the back and clamps the watch. A central round
//                   window exposes the sensor so the magnetic charger still
//                   reaches the back.
//    Hardware     : 4x M3x10 HEX-HEAD bolts (DIN 933, 5.5 mm across flats),
//                   self-tapping into printed pilot holes + paracord lanyard.
//
//  UNITS: millimetres.
// ===========================================================================

/* [Part to render / export] */
part = "assembly";        // ["assembly","bezel","backplate","watch_mock","side_gauge"]
show_watch = true;        // show the mock watch inside the assembly preview
show_bolts = true;        // show mock hex bolts in the assembly preview

/* [Watch size] */
// 1st-gen Apple Watch came in two sizes. If unsure, measure the body height:
//   ~38.6 mm tall  -> choose 38      ~42.5 mm tall -> choose 42
WATCH_SIZE = 42;          // [38,42]

/* [Fit & tolerance — tune for YOUR printer] */
tol        = 0.40;        // horizontal clearance around the watch body (per side)
depth_tol  = 0.40;        // clearance along the thickness axis
win_lip    = 2.2;         // how far the front window overlaps the watch edge (retention)

/* [Wall & shell] */
wall       = 2.6;         // side-wall thickness beside the watch
face_t     = 1.8;         // front face (bezel) thickness in front of the screen
back_d     = 4.0;         // backplate thickness (holds the fully-sunk hex heads)
end_top    = 8.0;         // solid material above the watch (top bolts)
end_bot    = 14.0;        // solid material below the watch (bottom bolts + lanyard)
outer_r    = 5.0;         // outer corner radius of the brick
fillet_bez = 1.2;         // edge fillet radius on the bezel (0.4–1.6 sensible)
fillet_bak = 1.0;         // edge fillet radius on the backplate

/* [Openings] */
back_win_d   = 0;         // central back window Ø (0 = auto: watch_w - 5)
crown_cut_d  = 11.5;      // Digital Crown access opening Ø (crown wall)
crown_scallop= 2.5;       // extra Ø of the finger scallop around the crown (0 = off)
crown_off_y  = 0;         // crown centre offset from watch mid-height (0 = auto)
button_cut_w = 5.0;       // side-button slot width
button_cut_h = 14.5;      // side-button slot length (along height)
button_off_y = 0;         // button centre offset from watch mid-height (0 = auto)

/* [Lanyard] */
lanyard_d  = 5.0;         // transverse lanyard hole Ø (type-III paracord is ~4 mm)

/* [Bolts — M3 hex head (DIN 933), self-tapping into printed pilots] */
screw_pilot_d = 2.5;      // pilot hole in the bezel (self-tap). Inserts? use 4.0
screw_clear_d = 3.4;      // clearance hole through the backplate
hex_af        = 5.5;      // bolt head size across flats (M3 DIN 933 = 5.5)
hex_clr       = 0.40;     // pocket clearance on the across-flats size
hex_seat      = 2.4;      // hex pocket depth; 2 mm head sinks ~0.4 mm SUB-FLUSH

/* [Render quality] */
$fn = 64;

// ===========================================================================
//  DERIVED DIMENSIONS
//  Body sizes from Apple's official 1st-gen (A1553/A1554) spec sheets:
//    38 mm: 38.6 x 33.3 x 10.5      42 mm: 42.0 x 35.9 x 10.5
//  NOTE the 42 mm 1st-gen body is 42.0 x 35.9 — the often-quoted
//  42.5 x 36.4 is the *Series 1-3* case, which is 0.5 mm larger.
// ===========================================================================
watch_w = (WATCH_SIZE==38) ? 33.3 : 35.9;   // body width
watch_h = (WATCH_SIZE==38) ? 38.6 : 42.0;   // body height
watch_d = 10.5;                              // body thickness (incl. curved back)
watch_r = (WATCH_SIZE==38) ? 8.8 : 9.3;     // body corner radius (estimate — tune)

// Digital Crown centre sits ~31% of body height down from the TOP edge, the
// side button centre ~57.5% (derived from Series 0 profile photos; Apple
// publishes no drawing). Offsets are from body mid-height, + = up.
_crown_off  = (crown_off_y  != 0) ? crown_off_y  : (watch_h/2 - 0.310*watch_h);
_button_off = (button_off_y != 0) ? button_off_y : (watch_h/2 - 0.575*watch_h);
_back_win   = (back_win_d   != 0) ? back_win_d   : (watch_w - 5.0);

pocket_w = watch_w + 2*tol;
pocket_h = watch_h + 2*tol;
pocket_r = watch_r + tol;
pocket_d = watch_d + depth_tol;

outer_w  = watch_w + 2*wall + 2*tol;
outer_h  = watch_h + 2*tol + end_top + end_bot;
outer_yc = (end_top - end_bot)/2;            // Y shift so the watch stays centred at 0
bezel_d  = face_t + pocket_d;                // total bezel thickness

win_w = watch_w - 2*win_lip;
win_h = watch_h - 2*win_lip;
win_r = max(1.0, watch_r - win_lip);

zc = face_t + watch_d/2;                      // right-wall opening centre (thickness)

// bolt / lanyard positions (watch-centred coordinates)
sx        = outer_w/2 - 5.2;
sy        = watch_h/2 + tol + 3.6;            // bolts sit 3.6 mm into each end block
lanyard_y = -(watch_h/2 + tol + end_bot*0.68);
screw_pos = [[ sx, sy],[-sx, sy],[ sx,-sy],[-sx,-sy]];

hex_r = (hex_af + hex_clr)/sqrt(3);           // hex pocket circumradius

// ===========================================================================
//  2D / 3D helpers
// ===========================================================================
module rrect(w,h,r){
    hull() for(ix=[-1,1], iy=[-1,1])
        translate([ix*(w/2-r), iy*(h/2-r)]) circle(r=r);
}
module slab(w,h,d,r){ linear_extrude(height=d) rrect(w,h,r); }

// slab with ALL edges filleted by radius e (rounded-rect prism (+) sphere)
module fslab(w,h,d,r,e){
    if(e<=0) slab(w,h,d,r);
    else minkowski(){
        translate([0,0,e]) slab(w-2*e, h-2*e, d-2*e, max(0.6, r-e));
        sphere(r=e, $fn=32);
    }
}

// ===========================================================================
//  PART: BEZEL (front tray)
// ===========================================================================
module bezel(){
    difference(){
        // solid outer body, filleted edges
        translate([0,outer_yc,0]) fslab(outer_w, outer_h, bezel_d, outer_r, fillet_bez);
        // watch pocket, open toward the back (+Z)
        translate([0,0,face_t]) slab(pocket_w, pocket_h, pocket_d+2, pocket_r);
        // screen window through the front face
        translate([0,0,-0.5]) slab(win_w, win_h, face_t+1, win_r);
        // Digital Crown opening — crown sits on the RIGHT when viewing the
        // screen; the front faces -z, so that is the -x wall of the model
        translate([-outer_w/2, _crown_off, zc])
            rotate([0,90,0]) cylinder(h=wall*4, d=crown_cut_d, center=true, $fn=48);
        // finger scallop around the crown opening (cone sunk into the wall)
        if(crown_scallop>0)
            translate([-(outer_w/2+0.01), _crown_off, zc]) rotate([0,90,0])
                cylinder(h=1.6, d1=crown_cut_d+crown_scallop, d2=crown_cut_d, $fn=48);
        // side-button slot (same wall as the crown, just below it)
        translate([-outer_w/2, _button_off, zc]) rotate([0,90,0])
            hull() for(iy=[-1,1])
                translate([0, iy*(button_cut_h-button_cut_w)/2, 0])
                    cylinder(h=wall*4, d=button_cut_w, center=true, $fn=32);
        // bolt pilot holes (drilled from the back, blind at the front)
        for(p=screw_pos)
            translate([p[0],p[1], face_t+1.0])
                cylinder(h=bezel_d, d=screw_pilot_d, $fn=24);
        // transverse lanyard hole through the bottom block
        translate([0, lanyard_y, bezel_d/2])
            rotate([0,90,0]) cylinder(h=outer_w+4, d=lanyard_d, center=true, $fn=32);
    }
}

// ===========================================================================
//  PART: BACKPLATE
// ===========================================================================
module backplate(){
    difference(){
        translate([0,outer_yc,0]) fslab(outer_w, outer_h, back_d, outer_r, fillet_bak);
        // central sensor / charging window
        translate([0,0,-0.5]) cylinder(h=back_d+1, d=_back_win, $fn=96);
        // bolt clearance holes + HEX pockets on the outer face
        //   (M3 hex heads drop in fully — ~0.4 mm below the surface, so the
        //    back lays flat; the hex seat keys them against rotation so they
        //    cannot self-loosen)
        for(p=screw_pos){
            translate([p[0],p[1],-0.5]) cylinder(h=back_d+1, d=screw_clear_d, $fn=32);
            translate([p[0],p[1],back_d-hex_seat])
                cylinder(h=hex_seat+0.5, r=hex_r, $fn=6);
        }
    }
}

// ===========================================================================
//  Mock watch & bolts (for the preview only — NOT printed)
// ===========================================================================
module watch_mock(){
    // body
    color("gray") slab(watch_w, watch_h, watch_d, watch_r);
    // screen
    color("black") translate([0,0,watch_d-0.4])
        slab(watch_w-4, watch_h-6, 0.6, max(1,watch_r-3));
    // digital crown (right side as viewed from the front = model -x)
    color("silver") translate([-watch_w/2, _crown_off, watch_d/2])
        rotate([0,-90,0]) cylinder(h=2.4, d=7, $fn=40);
    // side button
    color("silver") translate([-watch_w/2, _button_off, watch_d/2])
        rotate([0,-90,0]) cylinder(h=1.6, d=4, $fn=32);
}

// ===========================================================================
//  PART: SIDE GAUGE — 5-minute test print to verify crown/button positions
//  A thin plate exactly as tall as the watch body. Hold it against the
//  crown side of the watch, ends flush with the body top/bottom edges:
//  the crown must centre in the round hole, the button in the slot.
//  If not, measure the miss and adjust crown_off_y / button_off_y.
// ===========================================================================
module side_gauge(){
    gw = 16; gt = 2.0;
    difference(){
        linear_extrude(gt) rrect(gw, watch_h, 2);
        // crown + button apertures at the exact case-cut positions
        translate([0, _crown_off, -0.5]) cylinder(h=gt+1, d=crown_cut_d, $fn=48);
        translate([0, _button_off, -0.5]) linear_extrude(gt+1)
            hull() for(iy=[-1,1])
                translate([0, iy*(button_cut_h-button_cut_w)/2])
                    circle(d=button_cut_w, $fn=32);
        // chamfered corner marks the TOP edge
        translate([-gw/2, watch_h/2, -0.5])
            linear_extrude(gt+1) rotate(45) square(6, center=true);
        // engraved size label
        translate([0, -watch_h/2+6, gt-0.6]) linear_extrude(1)
            text(str(WATCH_SIZE), size=5, halign="center", font="DejaVu Sans:style=Bold");
    }
}

module bolt_mocks(){
    for(p=screw_pos)
        color("gainsboro")
            translate([p[0],p[1], bezel_d+0.2+back_d-hex_seat])
                cylinder(h=2.0, r=hex_af/sqrt(3), $fn=6);
}

// ===========================================================================
//  Render selector
// ===========================================================================
if(part=="bezel")          bezel();
else if(part=="backplate") backplate();
else if(part=="watch_mock") watch_mock();
else if(part=="side_gauge") side_gauge();
else {                                  // assembly preview
    color("darkolivegreen") bezel();
    color("olivedrab")      translate([0,0,bezel_d+0.2]) backplate();
    if(show_watch) color("gray") translate([0,0,face_t]) watch_mock();
    if(show_bolts) bolt_mocks();
}
