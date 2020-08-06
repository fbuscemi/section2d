% Z8 - machined/extruded, with round (half-circular) flange
include "globals.mzn";

var int: h;
var int: tw;
var int: rf;
var int: ba;
var int: ta;
var int: ra;

constraint tw = 10;

constraint h <= 600; % nicht zu groß machen, sonst zu viele Lösungen
constraint h mod 50 = 0; 
constraint h >= 10*tw;

constraint ba mod 10 = 0;
constraint ba >= tw + ra;
constraint 3*ba <= 2*h;

constraint ta mod 5 = 0;
constraint 2*ta >= tw;
constraint ta <= 3*tw;

constraint ra mod 5 = 0;
constraint ra >= tw;
constraint ra < 5*tw;

constraint rf mod 10 = 0;
constraint 5*rf >= h;
constraint 3*rf <= h;

solve satisfy;

% run with -n 0 (all solutions)
% output ["h=\(h),tw=\(tw),bf=\(bf),tf=\(tf),ba=\(ba),ta=\(ta)"];
