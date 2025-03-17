pomdp

const int N = 5;// grid size

// only the target is observable which is in the south east corner
// (also if the initialisation step has been done)
formula target = x=4 & y=4;
observable "target" = target;
observable "started" = started;

module grid

	x : [0..N-1]; // x coordinate
	y : [0..N-1]; // y coordinate
	started : bool; // initialised?

	// initially randomly placed within the grid (not at the target)
	[] !started -> 1/24: (started'=true) & (x'=0) & (y'=0)
			+ 1/24 : (started'=true) & (x'=0) & (y'=1)
			+ 1/24 : (started'=true) & (x'=0) & (y'=2)
			+ 1/24 : (started'=true) & (x'=0) & (y'=3)
			+ 1/24 : (started'=true) & (x'=0) & (y'=4)
			+ 1/24: (started'=true) & (x'=1) & (y'=0)
			+ 1/24 : (started'=true) & (x'=1) & (y'=1)
			+ 1/24 : (started'=true) & (x'=1) & (y'=2)
			+ 1/24 : (started'=true) & (x'=1) & (y'=3)
			+ 1/24 : (started'=true) & (x'=1) & (y'=4)
			+ 1/24: (started'=true) & (x'=2) & (y'=0)
			+ 1/24 : (started'=true) & (x'=2) & (y'=1)
			+ 1/24 : (started'=true) & (x'=2) & (y'=2)
			+ 1/24 : (started'=true) & (x'=2) & (y'=3)
			+ 1/24 : (started'=true) & (x'=2) & (y'=4)
			+ 1/24: (started'=true) & (x'=3) & (y'=0)
			+ 1/24 : (started'=true) & (x'=3) & (y'=1)
			+ 1/24 : (started'=true) & (x'=3) & (y'=2)
			+ 1/24 : (started'=true) & (x'=3) & (y'=3)
			+ 1/24 : (started'=true) & (x'=3) & (y'=4)
			+ 1/24: (started'=true) & (x'=4) & (y'=0)
			+ 1/24 : (started'=true) & (x'=4) & (y'=1)
			+ 1/24 : (started'=true) & (x'=4) & (y'=2)
			+ 1/24 : (started'=true) & (x'=4) & (y'=3);
			//+ 1/24 : (started'=true) & (x'=4) & (y'=4)

	// move around the grid
	[east] started & !target -> (x'=min(x+1,N-1));
	[west] started & !target -> (x'=max(x-1,0));
	[north] started & !target -> (x'=min(y+1,N-1));
	[south] started & !target -> (y'=max(y-1,0));

	// reached target
	[done] target -> true;

endmodule

// reward structure for number of steps to reach the target
rewards
        [east] true : 1;
        [west] true : 1;
        [north] true : 1;
        [south] true : 1;
endrewards
