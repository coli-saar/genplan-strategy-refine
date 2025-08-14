(define
	(problem test)
	(:domain beluga)
  (:objects
		t1 - beluga-truck
		t2 - fab-truck
		rack1 - rack
		rack2 - rack
		rack3 - rack
		part1 - part
		part2 - part
		plA - production-line
		n03 - num
		n04 - num
		n06 - num
		n07 - num
		n08 - num
		n09 - num
		n10 - num
		n11 - num
		n12 - num
		n13 - num
		n14 - num
		n15 - num
	)
  (:init
		(empty t1)
		(empty t2)
		(fit-n-sum  n00  n03  n03)
		(fit-n-sum  n00  n04  n04)
		(fit-n-sum  n03  n03  n06)
		(fit-n-sum  n03  n04  n07)
		(fit-n-sum  n04  n03  n07)
		(fit-n-sum  n04  n04  n08)
		(fit-n-sum  n06  n03  n09)
		(fit-n-sum  n06  n04  n10)
		(fit-n-sum  n07  n03  n10)
		(fit-n-sum  n07  n04  n11)
		(fit-n-sum  n08  n03  n11)
		(fit-n-sum  n08  n04  n12)
		(fit-n-sum  n09  n03  n12)
		(fit-n-sum  n09  n04  n13)
		(fit-n-sum  n10  n03  n13)
		(fit-n-sum  n10  n04  n14)
		(fit-n-sum  n11  n03  n14)
		(fit-n-sum  n11  n04  n15)
		(fit-n-sum  n12  n03  n15)
		
		(on-rack noner part1)
		(on-rack noner part2)
		(ready-to-unload part1)
		(next-to-unload part1 part2)
		(next-to-unload part2 nonep)
		(no-rack-succ part1)
		(no-rack-pre part1)
		(no-rack-succ part2)
		(no-rack-pre part2)
		(size part1 n03)
		(size part2 n04)
		(level rack1 n00)
		(level rack2 n00)
		(level rack3 n00)
		(unused rack1)
		(unused rack2)
		(unused rack3)
		(needed-next nonep)
		(p-line-next plA part2)
		(p-line-succ part2 part1)
		(p-line-succ part1 nonep)
		(p-line-succ nonep nonep)
	)
  (:goal (and 
		(processed part1)
		(processed part2)
	))
  
)