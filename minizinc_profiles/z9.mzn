% Z6 - machined/extruded, with fillets, nd one lipped flange
include "globals.mzn";

var int: h;
var int: tw;

var int: ba;
var int: ta;
var int: ra;

var int: bf;
var int: tf;
var int: rf;

var int: blf;
var int: tlf;
var int: rlf;

constraint tw = 10;

constraint h <= 300; % nicht zu groß machen, sonst zu viele Lösungen
constraint h mod 50 = 0; 
constraint h >= 10*tw;

constraint ba mod 40 = 0;
constraint ba >= tw + ra;
constraint 3*ba <= 2*h;

constraint ta mod 6 = 0;
constraint 2*ta >= tw;
constraint ta <= 2*tw;

constraint ra mod 6 = 0;
constraint ra >= tw;
constraint ra < 3*tw;

constraint bf mod 40 = 0;
constraint bf >= tw + rf + rlf + tlf;
constraint 2*bf <= 3*h;

constraint tf mod 6 = 0;
constraint 2*tf >= tw;
constraint tf <= 2*tw;

constraint rf mod 6 = 0;
constraint rf >= tw;
constraint rf < 3*tw;

constraint blf mod 20 = 0;
constraint blf >= rlf;
constraint 2*blf <= 3*bf;
constraint 3*blf <= h;

constraint tlf mod 6 = 0;
constraint 2*tlf >= tf;
constraint tlf <= 2*tf;

constraint rlf mod 6 = 0;
constraint rlf >= tw;
constraint rlf < 3*tw;


solve satisfy;

% run with -n 0 (all solutions)
% output ["h=\(h),tw=\(tw),bf=\(bf),tf=\(tf),ba=\(ba),ta=\(ta)"];
