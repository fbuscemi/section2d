% Z5 - machined/extruded, with fillets
include "globals.mzn";

var int: h;
var int: tw;

var int: ba;
var int: ta;
var int: ra;

var int: bf;
var int: tf;
var int: rf;

constraint tw = 10;

constraint h <= 300; % nicht zu groß machen, sonst zu viele Lösungen
constraint h mod 50 = 0; 
constraint h >= 10*tw;

constraint ba mod 30 = 0;
constraint ba >= tw + ra;
constraint 3*ba <= 2*h;

constraint ta mod 5 = 0;
constraint 2*ta >= tw;
constraint ta <= 2*tw;

constraint ra mod 4 = 0;
constraint ra >= tw;
constraint ra < 3*tw;

constraint bf mod 30 = 0;
constraint bf >= tw + rf;
constraint bf <= h;

constraint tf mod 5 = 0;
constraint 2*tf >= tw;
constraint tf <= 2*tw;

constraint rf mod 4 = 0;
constraint rf >= tw;
constraint rf < 3*tw;

constraint ba <= bf;  % eindeutig bleiben...

solve satisfy;

% run with -n 0 (all solutions)
% output ["h=\(h),tw=\(tw),bf=\(bf),tf=\(tf),ba=\(ba),ta=\(ta)"];
