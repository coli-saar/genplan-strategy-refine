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
		part3 - part
		part4 - part
		part5 - part
		plA - production-line
		n04 - num
		n05 - num
		n06 - num
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
		(fit-n-sum  n00  n04  n04)
		(fit-n-sum  n00  n05  n05)
		(fit-n-sum  n00  n06  n06)
		(fit-n-sum  n04  n04  n08)
		(fit-n-sum  n04  n05  n09)
		(fit-n-sum  n04  n06  n10)
		(fit-n-sum  n05  n04  n09)
		(fit-n-sum  n05  n05  n10)
		(fit-n-sum  n05  n06  n11)
		(fit-n-sum  n06  n04  n10)
		(fit-n-sum  n06  n05  n11)
		(fit-n-sum  n06  n06  n12)
		(fit-n-sum  n08  n04  n12)
		(fit-n-sum  n08  n05  n13)
		(fit-n-sum  n08  n06  n14)
		(fit-n-sum  n09  n04  n13)
		(fit-n-sum  n09  n05  n14)
		(fit-n-sum  n09  n06  n15)
		(fit-n-sum  n10  n04  n14)
		(fit-n-sum  n10  n05  n15)
		(fit-n-sum  n11  n04  n15)
		
		(on-rack noner part2)
		(on-rack noner part5)
		(on-rack noner part3)
		(on-rack noner part4)
		(on-rack noner part1)
		(ready-to-unload part2)
		(next-to-unload part2 part5)
		(next-to-unload part5 part3)
		(next-to-unload part3 part4)
		(next-to-unload part4 part1)
		(next-to-unload part1 nonep)
		(no-rack-succ part1)
		(no-rack-pre part1)
		(no-rack-succ part2)
		(no-rack-pre part2)
		(no-rack-succ part3)
		(no-rack-pre part3)
		(no-rack-succ part4)
		(no-rack-pre part4)
		(no-rack-succ part5)
		(no-rack-pre part5)
		(size part1 n04)
		(size part2 n05)
		(size part3 n05)
		(size part4 n05)
		(size part5 n06)
		(level rack1 n00)
		(level rack2 n00)
		(level rack3 n00)
		(unused rack1)
		(unused rack2)
		(unused rack3)
		(needed-next nonep)
		(p-line-next plA part3)
		(p-line-succ part3 part1)
		(p-line-succ part1 part4)
		(p-line-succ part4 part2)
		(p-line-succ part2 part5)
		(p-line-succ part5 nonep)
		(p-line-succ nonep nonep)
	)
  (:goal (and 
		(processed part1)
		(processed part2)
		(processed part3)
		(processed part4)
		(processed part5)
	))
  
)