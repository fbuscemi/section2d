% tube

include "globals.mzn";

var int: ra;
var int: ri;

constraint ra = 1000;
constraint ri/ra < 0.99;
constraint ri/ra > 0.01;
constraint ra > ri;
constraint ra > 0;
constraint ri > 0;

solve satisfy;