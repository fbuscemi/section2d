% Z3 - sheet metal, with lipped flange
include "globals.mzn";

var int: h;
var int: tw;
var int: bf;
var int: rf;
var int: ba;
var int: ra;
var int: blf;
var int: rlf;

constraint h <= 70; % nicht zu groß machen, sonst zu viele Lösungen
constraint h mod 10 = 0; 
constraint ba mod 3 = 0;
constraint bf mod 3 = 0;
constraint blf mod 2 = 0;
constraint h >= 10*tw;
constraint tw = 1;

constraint ba >= tw + ra;
constraint h >= 3*ba;
constraint ra >= tw;
constraint ra < 8*tw;

constraint bf >= 3*tw + rf + rlf;
constraint h >= 3*bf;
constraint rf >= tw;
constraint rf < 8*tw;

constraint blf >= rlf + tw;
constraint blf <= bf;
constraint rlf >= tw;
constraint rlf < 8*tw;


solve satisfy;

% run with -n 0 (all solutions)
% output ["h=\(h),tw=\(tw),bf=\(bf),tf=\(tf),ba=\(ba),ta=\(ta)"];
