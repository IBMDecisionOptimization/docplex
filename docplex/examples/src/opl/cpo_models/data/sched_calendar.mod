// --------------------------------------------------------------------------
// Licensed Materials - Property of IBM
//
// 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55
// Copyright IBM Corporation 1998, 2013. All Rights Reserved.
//
// Note to U.S. Government Users Restricted Rights:
// Use, duplication or disclosure restricted by GSA ADP Schedule
// Contract with IBM Corp.
// --------------------------------------------------------------------------

/* ------------------------------------------------------------

Problem Description
-------------------

This is a problem of building five houses. The masonry, roofing,
painting, etc. must be scheduled.  Some tasks must necessarily take
place before others and these requirements are expressed through
precedence constraints.

There are two workers and each task requires a specific worker.  The
worker has a calendar of days off that must be taken into account. The
objective is to minimize the overall completion date.

------------------------------------------------------------ */
 
using CP;

int NbHouses = ...; 
range Houses = 1..NbHouses;

{string} WorkerNames = ...;  
{string} TaskNames   = ...;

int    Duration [t in TaskNames] = ...;
string Worker   [t in TaskNames] = ...;

tuple Precedence {
  string pre;
  string post;
};

{Precedence} Precedences = ...;

tuple Break {
  int s;
  int e;
};

{Break} Breaks[WorkerNames] = ...; 

// Set of break steps
tuple Step {
  int v;
  key int x;
};

sorted {Step} Steps[w in WorkerNames] = 
   { <100, b.s> | b in Breaks[w] } union 
   { <0, b.e>   | b in Breaks[w] };
   
stepFunction Calendar[w in WorkerNames] = 
  stepwise (s in Steps[w]) { s.v -> s.x; 100 };

dvar interval itvs[h in Houses, t in TaskNames] 
  size      Duration[t]
  intensity Calendar[Worker[t]];

minimize max(h in Houses) endOf(itvs[h]["moving"]);

subject to {
  forall(h in Houses) {
    forall(p in Precedences)
      endBeforeStart(itvs[h][p.pre], itvs[h][p.post]);
    forall(t in TaskNames) {
      forbidStart(itvs[h][t], Calendar[Worker[t]]);
      forbidEnd  (itvs[h][t], Calendar[Worker[t]]);
    }
  }
  forall(w in WorkerNames)
    noOverlap( all(h in Houses, t in TaskNames: Worker[t]==w) itvs[h][t]);
}
