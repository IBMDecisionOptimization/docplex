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

{string} Products =...;
{int} TimePeriods =...;

{int} ZTime = {0} union TimePeriods;

float Rate[Products] =...;
float Inv0[Products] =...;
float Avail[TimePeriods] =...;
float Market[Products][TimePeriods] =...;
float Prodcost[Products] =...;
float Invcost[Products] =...;
float Revenue[Products][TimePeriods] =...;

dvar float+ Make[Products][TimePeriods];
dvar float+ Inv[Products][ZTime];
dvar float+ Sell[Products][TimePeriods];

maximize
  sum( p in Products , t in TimePeriods )
    ( Revenue[p][t] * Sell[p][t] 
    - Prodcost[p] * Make[p][t] - Invcost[p] * Inv[p][t] );

subject to{
  forall( t in TimePeriods )
    ctAvailable: 
      sum( p in Products ) 
        ( 1 / Rate[p] ) * Make[p][t] <= Avail[t];
  forall( p in Products )
    ctInit: 
      Inv[p][0] == Inv0[p];
  forall( p in Products , t in TimePeriods )
    ctProd:
      Make[p][t] + Inv[p][t-1] == Sell[p][t] + Inv[p][t];
  forall( p in Products , t in TimePeriods )
    ctMarket: 
      Sell[p][t] <= Market[p][t]; 
}
