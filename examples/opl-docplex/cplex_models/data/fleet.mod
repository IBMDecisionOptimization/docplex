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

/*  ----------------------------------------------------
 *   OPL Model for Fleet Assignment Example
 *
 *   This model is described in the documentation. 
 *   See IDE and OPL > Language and Interfaces Examples.
 *
 * This model is greater than the size allowed in trial mode. 
 * You therefore need a commercial edition of CPLEX Studio 
 * to run this example. 
 * If you are a student or teacher, you can also get a 
 * full version through the IBM Academic Initiative.
 *
 */

int      Horizon = ...;
int      MaxRefuel = ...;
float    MaxSpill = ...;
{string} Airports = ...;
{string} Fleets = ...;

// Characteristics of each fleet
tuple fleetType 
{
  int      aircrafts;  // number of aircrafts for each fleet
  string   dist;       // long-medium-short distance flight
  int      seats;      // number of seats
  int      refuelT;    // Time on ground between flights
  int      a;          // fixed cost
  int      b;          // variable cost : total cost = a + BH * b * seats where BH is the flight time
}
fleetType FleetInfo[Fleets] = ...; 
assert forall(pl in Fleets) FleetInfo[pl].refuelT <= MaxRefuel;

// Flight legs
tuple flightLeg
{
  int       id;    //flight id
  string    depA;  //departure airport
  int       depT;  //departure time 
  string    arrA;  //arrival airport
  int       arrT;  //arrival time 
  string    dist;  //long-medium-short distance flight
  int       pax;   //passenger demand
  int       price; //ticket price
}

{flightLeg} FlightLegs = ...; 
{flightLeg} Flights = ...;

assert forall(fl in FlightLegs)  0 <= fl.depT; // <= Horizon;
assert forall(fl in FlightLegs)  0 <= fl.arrT; // <= Horizon;
assert forall(fl in FlightLegs)  Horizon >= fl.depT;
assert forall(fl in FlightLegs)  Horizon >= fl.arrT;
assert forall(fl in FlightLegs)  fl.depT < fl.arrT;

// One-stop flights
// These are flights which are broken into 2 sub-flights, and are flown by the
// same aircraft.
tuple oneStop
{
  key int firstId;
  int secondId;
}
{oneStop} OneStopFlights = ...;

// Cash Direct Operating Costs
// for every couple (fleet,flight) we define cdoc = a + b*(demand-spill)*(arrTime-depTime)
int Cost[Flights][Fleets] = ...;

// Profit
// for every couple (fleet,flight) we define profit = (demand-spill)*ticket_price
int Profit[Flights][Fleets] = ...;


/*  ----------------------------------------------------
 *   Variables:
 *   assignment -- assignment[fl][pl] means flights[fl] is 
 *         covered by a plane in fleet[pl].
 *   --------------------------------------------------- */
dvar boolean Assignment[Flights][Fleets];

dexpr float objective = 
  sum(fl in Flights, pl in Fleets)
     (-Profit[fl][pl] + Cost[fl][pl]) * Assignment[fl][pl];
      
minimize objective;

subject to {
  // Every plane of each fleet must come from the source only once.
  forall(pl in Fleets)
    ctSource: sum(fl in Flights: fl.depA == "Source") Assignment[fl][pl] <= FleetInfo[pl].aircrafts;

  // Every plane of each fleet must go to the sink only once.
  forall(pl in Fleets)
    ctSink: sum(fl in Flights: fl.arrA == "Sink") Assignment[fl][pl] <= FleetInfo[pl].aircrafts;

  forall(fl in Flights: fl.depA != "Source" && fl.arrA != "Sink") 
    // Every "real" flight must have a plane assigned to it
    sum(pl in Fleets) Assignment[fl][pl] == 1;

  forall(pl in Fleets, fl in Flights: fl.depA != "Source" && fl.arrA != "Sink") 
    // The plane must be at the airport in order to use it!
    sum(prevf in Flights: prevf.arrA == fl.depA && prevf.arrT + FleetInfo[pl].refuelT <= fl.depT) 
      Assignment[prevf][pl] -
    sum(prevf in Flights: prevf != fl && prevf.depA == fl.depA && prevf.depT <= fl.depT) 
      Assignment[prevf][pl]
    >= Assignment[fl][pl];
      
  forall(pl in Fleets, fl in Flights: fl.arrA == "Sink")
    // The plane must be at the airport in order to use it!
    sum(prevf in Flights: prevf.arrA == fl.depA && prevf.arrT + FleetInfo[pl].refuelT <= fl.depT) 
      Assignment[prevf][pl] -
    sum(prevf in Flights: prevf != fl && prevf.depA == fl.depA && prevf.depT <= fl.depT) 
      Assignment[prevf][pl]
    >= Assignment[fl][pl];
     
  // Type-compatibility check between fleets and flights (long/medium/short distance)
  // Long-haul aircrafts can fly long, medium and short flights, medium-haul aircrafts
  // can only fly medium and short flights, and so on..
  forall(fl in Flights: fl.depA != "Source" && fl.arrA != "Sink")
    forall(pl in Fleets: FleetInfo[pl].dist == "Short" && (fl.dist == "Medium" || fl.dist == "Long"))
      Assignment[fl][pl] == 0;
  forall(fl in Flights: fl.depA != "Source" && fl.arrA != "Sink")
    forall(pl in Fleets: FleetInfo[pl].dist == "Medium" && fl.dist == "Long")
      Assignment[fl][pl] == 0;

  forall(fl in Flights: fl.depA != "Source" && fl.arrA != "Sink")
    // MaxSpill constraint
    // if MaxSpill=.1 and the demand for the flight f is 100, then the maximum no of passengers 
    // that can be spilled is 10 (i.e. you must use an aircraft which has seat capacity >= 90 )
    ctSpill: sum(pl in Fleets) Assignment[fl][pl]*FleetInfo[pl].seats >= fl.pax*(1-MaxSpill);

  forall(pl in Fleets) {
    // Planes must end the day where they started the day
    forall(ap in Airports) {
      sum(fl in Flights: fl.depA == "Source" && fl.arrA == ap) Assignment[fl][pl] ==
      sum(fl in Flights: fl.depA == ap && fl.arrA == "Sink") Assignment[fl][pl];
    }

    // One-Stop services
    // force each ordered pair of forced turn flights to use the same equipment type
    forall(pair in OneStopFlights) {
      sum(f1 in Flights: f1.id == pair.firstId) Assignment[f1][pl] ==
      sum(f2 in Flights: f2.id == pair.secondId) Assignment[f2][pl];
    }
  }
}
