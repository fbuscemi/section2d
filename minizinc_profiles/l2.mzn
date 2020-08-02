% C2 - sheet metal c section
include "globals.mzn";

var int: h;
var int: tw;
var int: ba;
var int: ra;

constraint h = 100; % nicht zu groß machen, sonst zu viele Lösungen
constraint tw > 0;
constraint ba > tw + ra;
constraint ra > tw;
constraint ra < 5*tw;

constraint 5*tw <= h;
constraint 100*tw >= h; 

constraint ba < h;

constraint ra + 2*tw < 0.8*h;

solve satisfy;

% run with -n 0 (all solutions)
% output ["h=\(h),tw=\(tw),bf=\(bf),tf=\(tf),ba=\(ba),ta=\(ta)"];
