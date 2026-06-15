// 2D periodic square box for Gaussian pulse tests
// Domain: x,y in [-1, 1]  (2 x 2 square)
// 16 x 16 quads (0.125 x 0.125 cells; half the previous 0.25 size)

Point(1) = {-1, -1, 0, 1};
Point(2) = { 1, -1, 0, 1};
Point(3) = { 1,  1, 0, 1};
Point(4) = {-1,  1, 0, 1};

Line(1) = {1, 4};   // left   (x = -1)
Line(2) = {2, 3};   // right  (x =  1)
Line(3) = {1, 2};   // bottom (y = -1)
Line(4) = {4, 3};   // top    (y =  1)

Curve Loop(1) = {3, 2, -4, -1};
Plane Surface(1) = {1};

Transfinite Curve {1, 2} = 17;
Transfinite Curve {3, 4} = 17;
Transfinite Surface {1} = {1, 2, 3, 4};
Recombine Surface {1};

Physical Curve("periodic_0_r") = {1};
Physical Curve("periodic_0_l") = {2};
Physical Curve("periodic_1_l") = {3};
Physical Curve("periodic_1_r") = {4};
Physical Surface("Fluid")      = {1};
