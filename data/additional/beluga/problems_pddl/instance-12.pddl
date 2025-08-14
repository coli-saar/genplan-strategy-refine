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
		n06 - num
		n07 - num
		n12 - num
		n13 - num
		n14 - num
	)
  (:init
		(empty t1)
		(empty t2)
		(fit-n-sum  n00  n06  n06)
		(fit-n-sum  n00  n07  n07)
		(fit-n-sum  n06  n06  n12)
		(fit-n-sum  n06  n07  n13)
		(fit-n-sum  n07  n06  n13)
		(fit-n-sum  n07  n07  n14)
		
		(on-rack noner part3)
		(on-rack noner part2)
		(on-rack noner part1)
		(ready-to-unload part3)
		(next-to-unload part3 part2)
		(next-to-unload part2 part1)
		(next-to-unload part1 nonep)
		(no-rack-succ part1)
		(no-rack-pre part1)
		(no-rack-succ part2)
		(no-rack-pre part2)
		(no-rack-succ part3)
		(no-rack-pre part3)
		(size part1 n07)
		(size part2 n06)
		(size part3 n07)
		(level rack1 n00)
		(level rack2 n00)
		(unused rack1)
		(unused rack2)
		(needed-next nonep)
		(p-line-next plA part1)
		(p-line-succ part1 part3)
		(p-line-succ part3 part2)
		(p-line-succ part2 nonep)
		(p-line-succ nonep nonep)
	)
  (:goal (and 
		(processed part1)
		(processed part2)
		(processed part3)
	))
  
)