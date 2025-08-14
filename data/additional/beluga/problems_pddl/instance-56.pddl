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
		part3 - part
		plA - production-line
		n03 - num
		n06 - num
		n09 - num
		n12 - num
		n15 - num
	)
  (:init
		(empty t1)
		(empty t2)
		(fit-n-sum  n00  n03  n03)
		(fit-n-sum  n03  n03  n06)
		(fit-n-sum  n06  n03  n09)
		(fit-n-sum  n09  n03  n12)
		(fit-n-sum  n12  n03  n15)
		
		(on-rack noner part2)
		(on-rack noner part3)
		(on-rack noner part1)
		(ready-to-unload part2)
		(next-to-unload part2 part3)
		(next-to-unload part3 part1)
		(next-to-unload part1 nonep)
		(no-rack-succ part1)
		(no-rack-pre part1)
		(no-rack-succ part2)
		(no-rack-pre part2)
		(no-rack-succ part3)
		(no-rack-pre part3)
		(size part1 n03)
		(size part2 n03)
		(size part3 n03)
		(level rack1 n00)
		(level rack2 n00)
		(unused rack1)
		(unused rack2)
		(needed-next nonep)
		(p-line-next plA part2)
		(p-line-succ part2 part1)
		(p-line-succ part1 part3)
		(p-line-succ part3 nonep)
		(p-line-succ nonep nonep)
	)
  (:goal (and 
		(processed part1)
		(processed part2)
		(processed part3)
	))
  
)