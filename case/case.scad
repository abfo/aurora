// max dimensions 140^3

// Pi Mount Holes
translate([-29,-24.5,5]){
    difference() {
        cylinder(h=5, d=6);
        cylinder(h=6, d=2.4);
    }
}

translate([29,-24.5,5]){
    difference() {
        cylinder(h=5, d=6);
        cylinder(h=6, d=2.4);
    }
}

translate([-29,24.5,5]){
    difference() {
        cylinder(h=5, d=6);
        cylinder(h=6, d=2.4);
    }
}

translate([29,24.5,5]){
    difference() {
        cylinder(h=5, d=6);
        cylinder(h=6, d=2.4);
    }
}


// Size of Board, extra 3.5mm for distance from edge to center of hole
translate([-32.5,-28,0]){
    color([.2,.8,.2])
    cube(size=[85, 56, 5], center = false);
}

// bottom of case, this needs to get to 75mm on y for the speakers and another 32mm on each side for the width of each speaker...
translate([-52.5,-37.5,-3]){
    color([.8,.2,.2])
    cube(size=[125, 75, 6], center = false);
}

// wall 1 and 2
translate([-55.5,-40.5,-3]){
    difference() {
        // wall
        cube(size=[131, 3, 55], center = false);
        // port hole
        translate([65.5,11.5,19]){
            cube(size=[85, 30, 16], center = true);
        }
        for(x = [5:5:125]) {
            for(z = [5:5:46]) {
                translate([x, 5, z]) {
                    rotate([90,0,0]) {
                        cylinder(h = 25, d = 2);
                    }
                }
            }
        }
    }
    translate([0, 0, 55]) {
        cube(size=[131, 6, 3], center = false);
    }
    difference() {
        cube(size=[3, 81, 51.75], center = false);
        for(y = [5:5:75]) {
            for(z = [5:5:46]) {
                translate([0, y, z]) {
                    rotate([0,90,0]) {
                        cylinder(h = 25, d = 2);
                    }
                }
            }
        }
    }
}

// wall 3 
translate([-55.5,37.5,-3]){
    difference() {
        cube(size=[131, 3, 55], center = false);
        for(x = [5:5:125]) {
            for(z = [5:5:46]) {
                translate([x, 5, z]) {
                    rotate([90,0,0]) {
                        cylinder(h = 25, d = 2);
                    }
                }
            }
        }
    }
    translate([0, -3, 55]) {
        cube(size=[131, 6, 3], center = false);
    }
}

// wall 3  - wall hang
translate([-55.5,37.5,-3]){
    difference() {
        cube(size=[131, 23, 6], center = false);
        translate([20, 13, -1]) {
            cylinder(h = 8, d = 3.45);
        }
        translate([111, 13, -1]) {
            cylinder(h = 8, d = 3.45);
        }
    }
}

// wall 4
translate([72.5,-40.5,-3]){
    difference() {
        cube(size=[3, 81, 51.75], center = false);
        for(y = [5:5:75]) {
            for(z = [5:5:46]) {
                translate([0, y, z]) {
                    rotate([0,90,0]) {
                        cylinder(h = 25, d = 2);
                    }
                }
            }
        }
    }
}

