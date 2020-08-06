% L2 - sheet metal L section
include "globals.mzn";

var int: h;
var int: tw;
var int: ba;
var int: ra;

constraint h <= 80; % nicht zu groß machen, sonst zu viele Lösungen
constraint h mod 4 = 0; 
constraint h >= 5*tw;
constraint tw = 1;

constraint ba >= tw + ra;
constraint ra >= tw;
constraint ra <= 8*tw;
constraint ba <= h;

solve satisfy;

% run with -n 0 (all solutions)
% output ["h=\(h),tw=\(tw),bf=\(bf),tf=\(tf),ba=\(ba),ta=\(ta)"];
