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

using CP;

int nbPerm = ...;
range r = 1..nbPerm;
int dist[r][r] = ...;
int flow[r][r] =...;

range R1 = 1..nbPerm;
dvar int perm[R1] in r;

dexpr int cost[i in r][j in r] = dist[i][j]*flow[perm[i]][perm[j]];

minimize sum(i in r, j in r) cost[i][j];
subject to {
  allDifferent(perm);
};
