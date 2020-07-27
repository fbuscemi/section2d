% C2 - sheet metal c section
include "globals.mzn";

var int: h;
var int: tw;
var int: bf;
var int: rf;
var int: ba;
var int: ra;

constraint h = 23; % nicht zu groß machen, sonst zu viele Lösungen
constraint tw > 0;
constraint bf > tw + rf;
constraint ba > tw + ra;
constraint rf > tw;
constraint ra > tw;
constraint rf < 5*tw;
constraint ra < 5*tw;

constraint 5*tw <= h;
constraint 100*tw >= h; 

constraint ba < h;
constraint bf < h;

constraint ra + rf + 2*tw < 0.8*h;
constraint ba >= bf; % to avoid ambuguities in classification and finding the parameters

solve satisfy;

% run with -n 0 (all solutions)
% output ["h=\(h),tw=\(tw),bf=\(bf),tf=\(tf),ba=\(ba),ta=\(ta)"];
