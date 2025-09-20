union() {
    difference() {
        cube([6,6,12]);
        translate([1.35,1.35,-1]) {
            cube([3.3,3.3,3.5]);
        }
    }
    translate([3, 3, 11]) {
        cylinder(h = 2, r = 6, center = true);
    }
}