% C2 - sheet metal c section
include "globals.mzn";

var int: h;
var int: tw;
var int: bf;
var int: rf;
var int: ba;
var int: ra;

constraint h <= 80; % nicht zu groß machen, sonst zu viele Lösungen
constraint h mod 10 = 0; 
constraint h >= 4*tw;
constraint tw = 1;

constraint ba mod 4 = 0;
constraint ba >= tw + ra;
constraint ra >= tw;
constraint ra <= 8*tw;
constraint 3*ba <= 2*h;
constraint ba <= bf;  % eindeutig bleiben...

constraint bf mod 4 = 0;
constraint bf >= tw + rf;
constraint rf >= tw;
constraint rf <= 8*tw;
constraint 3*bf <= 2*h;

solve satisfy;

% run with -n 0 (all solutions)
% output ["h=\(h),tw=\(tw),bf=\(bf),tf=\(tf),ba=\(ba),ta=\(ta)"];
