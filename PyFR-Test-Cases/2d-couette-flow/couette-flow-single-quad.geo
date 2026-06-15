// Couette flow – single quadrilateral element
// Domain: x in [-1, 1], y in [0, 1]

Point(1) = {-1, 0, 0, 1};
Point(2) = { 1, 0, 0, 1};
Point(3) = { 1, 1, 0, 1};
Point(4) = {-1, 1, 0, 1};

Line(1) = {1, 4};   // left   (periodic_0_r, x = -1)
Line(2) = {2, 3};   // right  (periodic_0_l, x =  1)
Line(3) = {1, 2};   // bottom (bcwalllower,  y =  0)
Line(4) = {4, 3};   // top    (bcwallupper,  y =  1)

Curve Loop(1) = {3, 2, -4, -1};
Plane Surface(1) = {1};

Transfinite Curve {1, 2} = 2;   // 1 element in y
Transfinite Curve {3, 4} = 2;   // 1 element in x
Transfinite Surface {1} = {1, 2, 3, 4};
Recombine Surface {1};

Physical Curve("periodic_0_r") = {1};
Physical Curve("periodic_0_l") = {2};
Physical Curve("bcwalllower")  = {3};
Physical Curve("bcwallupper")  = {4};
Physical Surface("Fluid")      = {1};
