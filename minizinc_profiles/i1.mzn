% i1 - simple i section
include "globals.mzn";

var int: h;
var int: tw;
var int: bf;
var int: tf;
var int: ba;
var int: ta;

constraint h = 13; % nicht zu groß machen, sonst zu viele Lösungen
constraint tw > 0;
constraint bf > tw;
constraint tf > 0;
constraint ba > tw;
constraint ta > 0;

constraint tf < h/3;
constraint ta < h/3; 

constraint ba < h;
constraint bf < h;

constraint bf < 50*tf;
constraint ba < 50*ta;
constraint tf < 3*bf;
constraint ta < 3*ba;


constraint tw < 1.5 * min(tf, ta);
constraint tw > h/50;
constraint tw < h/2;

constraint ba >= bf;

solve satisfy;

% run with -n 0 (all solutions)
% output ["h=\(h),tw=\(tw),bf=\(bf),tf=\(tf),ba=\(ba),ta=\(ta)"];
