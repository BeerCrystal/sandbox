// =====================================================================
//  Cross-sections through the two stations, with the bar drawn as an
//  outline so you can see what grips what.
//
//    make profiles
//
//  station = "jaw"  lift = 0   -> seated: bar at the TOP of the slot,
//                                 sitting on solid wall, mouth empty
//                                 below it. This is the locked state.
//    station = "jaw"  lift = 10 -> raised: bar has dropped to the
//                                 bottom of the slot and lines up with
//                                 the mouth. Only now can it click off.
//    station = "cap"  lift = 0   -> the inverted U over the top.
// =====================================================================

include <stroller_clip.scad>

station = "jaw";
lift    = 0;

linear_extrude(1) {
    if (station == "jaw") jaw_2d(); else cap_2d();

    // the bar, as a ring outline
    translate([0, -lift]) difference() {
        offset(r = 0.9) rrect(handle_w, handle_h, handle_r);
        rrect(handle_w, handle_h, handle_r);
    }
}
