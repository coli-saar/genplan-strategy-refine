(define
	(problem test)
	(:domain beluga)
  (:objects
		t1 - beluga-truck
		t2 - fab-truck
		rack1 - rack
		rack2 - rack
		part1 - part
		part2 - part
		plA - production-line
		n05 - num
		n10 - num
		n15 - num
	)
  (:init
		(empty t1)
		(empty t2)
		(fit-n-sum  n00  n05  n05)
		(fit-n-sum  n05  n05  n10)
		(fit-n-sum  n10  n05  n15)
		
		(on-rack noner part1)
		(on-rack noner part2)
		(ready-to-unload part1)
		(next-to-unload part1 part2)
		(next-to-unload part2 nonep)
		(no-rack-succ part1)
		(no-rack-pre part1)
		(no-rack-succ part2)
		(no-rack-pre part2)
		(size part1 n05)
		(size part2 n05)
		(level rack1 n00)
		(level rack2 n00)
		(unused rack1)
		(unused rack2)
		(needed-next nonep)
		(p-line-next plA part1)
		(p-line-succ part1 part2)
		(p-line-succ part2 nonep)
		(p-line-succ nonep nonep)
	)
  (:goal (and 
		(processed part1)
		(processed part2)
	))
  
)