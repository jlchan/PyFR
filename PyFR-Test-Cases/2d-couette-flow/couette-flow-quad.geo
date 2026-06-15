// Couette flow – uniform structured quadrilateral mesh
// Domain: x in [-1, 1], y in [0, 1]
// 8 divisions in x, 4 divisions in y (matches original boundary resolution)

// Corner points
Point(1) = {-1, 0, 0, 1};   // bottom-left
Point(2) = { 1, 0, 0, 1};   // bottom-right
Point(3) = { 1, 1, 0, 1};   // top-right
Point(4) = {-1, 1, 0, 1};   // top-left

// Boundary curves
Line(1) = {1, 4};   // left side  (periodic_0_r, x = -1)
Line(2) = {2, 3};   // right side (periodic_0_l, x =  1)
Line(3) = {1, 2};   // bottom     (bcwalllower,  y =  0)
Line(4) = {4, 3};   // top        (bcwallupper,  y =  1)

// Surface (counter-clockwise)
Curve Loop(1) = {3, 2, -4, -1};
Plane Surface(1) = {1};

// Transfinite structured quad mesh
Transfinite Curve {1, 2} = 5;   // 4 elements in y (5 nodes)
Transfinite Curve {3, 4} = 9;   // 8 elements in x (9 nodes)
Transfinite Surface {1} = {1, 2, 3, 4};
Recombine Surface {1};

// Physical groups – names must match couette-flow.ini boundary conditions
Physical Curve("periodic_0_r") = {1};   // left  (x = -1)
Physical Curve("periodic_0_l") = {2};   // right (x =  1)
Physical Curve("bcwalllower")  = {3};   // bottom wall
Physical Curve("bcwallupper")  = {4};   // top wall
Physical Surface("Fluid")      = {1};
