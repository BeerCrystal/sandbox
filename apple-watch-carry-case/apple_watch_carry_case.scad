// ===========================================================================
//  Apple Watch "Pocket Carry" Case  —  parametric, 3D-printable
//  Adapted for the 1st-generation Apple Watch (Series 0: 38 mm / 42 mm)
//
//  Inspired by the screwed-sandwich pocket fob in the reference video
//  (an Apple Watch Ultra brick).  This version is re-dimensioned for the
//  smaller, curved-back Series 0 body and is fully parametric so you can
//  tune it to YOUR watch and YOUR printer.
//
//  TWO PRINTED PARTS + hardware:
//    * bezel      : front tray. Watch drops in face-first; a lip around the
//                   screen window retains it. Right wall has openings for the
//                   Digital Crown and the side button. Lanyard hole at bottom.
//    * backplate  : screws onto the back and clamps the watch in place. A
//                   central round window exposes the sensor so the magnetic
//                   charger still reaches the back.
//    Hardware     : 4x M3 self-tapping screws (~16 mm) + a paracord lanyard.
//
//  UNITS: millimetres.
// ===========================================================================

/* [Part to render / export] */
part = "assembly";        // ["assembly","bezel","backplate","watch_mock"]
show_watch = true;        // show the mock watch inside the assembly preview

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
back_d     = 3.0;         // backplate thickness
end_top    = 10.0;        // solid material above the watch (top screws)
end_bot    = 16.0;        // solid material below the watch (bottom screws + lanyard)
outer_r    = 4.0;         // outer corner radius of the brick
edge_cham  = 0.8;         // small chamfer on outer edges (hand feel); 0 to disable

/* [Openings] */
back_win_d   = 0;         // central back window Ø (0 = auto: watch_w - 5)
crown_cut_d  = 11.0;      // Digital Crown access opening Ø (right wall)
crown_off_y  = 0;         // crown centre offset from watch mid-height (0 = auto)
button_cut_w = 5.0;       // side-button slot width
button_cut_h = 14.0;      // side-button slot length (along height)
button_off_y = 0;         // button centre offset from watch mid-height (0 = auto)

/* [Lanyard] */
lanyard_d  = 5.5;         // transverse lanyard hole Ø at the bottom

/* [Screws — M3 self-tapping into printed bosses by default] */
screw_pilot_d = 2.5;      // pilot hole in the bezel (self-tap). Inserts? use 4.0
screw_clear_d = 3.4;      // clearance hole through the backplate
screw_head_d  = 6.2;      // counterbore Ø for the screw head
screw_cbore   = 2.2;      // counterbore depth

/* [Render quality] */
$fn = 64;

// ===========================================================================
//  DERIVED DIMENSIONS  (official Series 0 body sizes; corner radius estimated)
// ===========================================================================
watch_w = (WATCH_SIZE==38) ? 33.3 : 36.4;   // body width
watch_h = (WATCH_SIZE==38) ? 38.6 : 42.5;   // body height
watch_d = 10.5;                              // body thickness (incl. curved back)
watch_r = (WATCH_SIZE==38) ? 8.8 : 9.6;     // body corner radius (estimate — tune)

// crown high on the right side, button just below it (estimates — tune)
_crown_off  = (crown_off_y  != 0) ? crown_off_y  : ((WATCH_SIZE==38) ?  3.0 :  3.5);
_button_off = (button_off_y != 0) ? button_off_y : ((WATCH_SIZE==38) ? -7.0 : -8.0);
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

// screw / lanyard positions (watch-centred coordinates)
sx        = outer_w/2 - 5.0;
sy_top    =  (watch_h/2 + tol + end_top*0.5);
sy_bot    = -(watch_h/2 + tol + end_bot*0.42);
lanyard_y = -(watch_h/2 + tol + end_bot*0.80);
screw_pos = [[ sx, sy_top],[-sx, sy_top],[ sx, sy_bot],[-sx, sy_bot]];

// ===========================================================================
//  2D / 3D helpers
// ===========================================================================
module rrect(w,h,r){
    hull() for(ix=[-1,1], iy=[-1,1])
        translate([ix*(w/2-r), iy*(h/2-r)]) circle(r=r);
}
module slab(w,h,d,r){ linear_extrude(height=d) rrect(w,h,r); }

// ===========================================================================
//  PART: BEZEL (front tray)
// ===========================================================================
module bezel(){
    difference(){
        // solid outer body
        translate([0,outer_yc,0]) slab(outer_w, outer_h, bezel_d, outer_r);
        // watch pocket, open toward the back (+Z)
        translate([0,0,face_t]) slab(pocket_w, pocket_h, pocket_d+1, pocket_r);
        // screen window through the front face
        translate([0,0,-0.5]) slab(win_w, win_h, face_t+1, win_r);
        // Digital Crown opening (right wall)
        translate([outer_w/2, _crown_off, zc])
            rotate([0,90,0]) cylinder(h=wall*4, d=crown_cut_d, center=true, $fn=48);
        // side-button slot (right wall)
        translate([outer_w/2, _button_off, zc]) rotate([0,90,0])
            hull() for(iy=[-1,1])
                translate([0, iy*(button_cut_h-button_cut_w)/2, 0])
                    cylinder(h=wall*4, d=button_cut_w, center=true, $fn=32);
        // screw pilot holes (drilled from the back, blind at the front)
        for(p=screw_pos)
            translate([p[0],p[1], face_t+1.0])
                cylinder(h=bezel_d, d=screw_pilot_d, $fn=24);
        // transverse lanyard hole through the bottom block
        translate([0, lanyard_y, bezel_d/2])
            rotate([0,90,0]) cylinder(h=outer_w+2, d=lanyard_d, center=true, $fn=32);
    }
}

// ===========================================================================
//  PART: BACKPLATE
// ===========================================================================
module backplate(){
    difference(){
        translate([0,outer_yc,0]) slab(outer_w, outer_h, back_d, outer_r);
        // central sensor / charging window
        translate([0,0,-0.5]) cylinder(h=back_d+1, d=_back_win, $fn=96);
        // screw clearance + counterbore (heads recessed on the outer face)
        for(p=screw_pos){
            translate([p[0],p[1],-0.5]) cylinder(h=back_d+1, d=screw_clear_d, $fn=32);
            translate([p[0],p[1],back_d-screw_cbore])
                cylinder(h=screw_cbore+0.5, d=screw_head_d, $fn=32);
        }
    }
}

// ===========================================================================
//  Mock watch (for the preview only — NOT printed)
// ===========================================================================
module watch_mock(){
    // body
    color("gray") slab(watch_w, watch_h, watch_d, watch_r);
    // screen
    color("black") translate([0,0,watch_d-0.4])
        slab(watch_w-4, watch_h-6, 0.6, max(1,watch_r-3));
    // digital crown
    color("silver") translate([watch_w/2, _crown_off, watch_d/2])
        rotate([0,90,0]) cylinder(h=2.4, d=7, $fn=40);
    // side button
    color("silver") translate([watch_w/2, _button_off, watch_d/2])
        rotate([0,90,0]) cylinder(h=1.6, d=4, $fn=32);
}

// ===========================================================================
//  Render selector
// ===========================================================================
if(part=="bezel")          bezel();
else if(part=="backplate") backplate();
else if(part=="watch_mock") watch_mock();
else {                                  // assembly preview
    color("darkolivegreen") bezel();
    color("olivedrab")      translate([0,0,bezel_d+0.2]) backplate();
    if(show_watch) color("gray") translate([0,0,face_t]) watch_mock();
}
