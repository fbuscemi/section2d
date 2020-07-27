% rect

include "globals.mzn";

var int: h;
var int: b;

constraint h <= 100;
constraint b <= 100;
constraint h < 100*b;
constraint b < 100*h;
constraint h > 0;
constraint b > 0;

solve satisfy;