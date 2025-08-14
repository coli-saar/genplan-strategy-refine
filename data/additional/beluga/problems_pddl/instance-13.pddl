(define
	(problem test)
	(:domain beluga)
  (:objects
		t1 - beluga-truck
		t2 - fab-truck
		rack1 - rack
		rack2 - rack
		rack3 - rack
		rack4 - rack
		part1 - part
		part2 - part
		part3 - part
		part4 - part
		plA - production-line
		n03 - num
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
		(fit-n-sum  n00  n03  n03)
		(fit-n-sum  n00  n05  n05)
		(fit-n-sum  n00  n06  n06)
		(fit-n-sum  n03  n03  n06)
		(fit-n-sum  n03  n05  n08)
		(fit-n-sum  n03  n06  n09)
		(fit-n-sum  n05  n03  n08)
		(fit-n-sum  n05  n05  n10)
		(fit-n-sum  n05  n06  n11)
		(fit-n-sum  n06  n03  n09)
		(fit-n-sum  n06  n05  n11)
		(fit-n-sum  n06  n06  n12)
		(fit-n-sum  n08  n03  n11)
		(fit-n-sum  n08  n05  n13)
		(fit-n-sum  n08  n06  n14)
		(fit-n-sum  n09  n03  n12)
		(fit-n-sum  n09  n05  n14)
		(fit-n-sum  n09  n06  n15)
		(fit-n-sum  n10  n03  n13)
		(fit-n-sum  n10  n05  n15)
		(fit-n-sum  n11  n03  n14)
		(fit-n-sum  n12  n03  n15)
		
		(on-rack noner part4)
		(on-rack noner part1)
		(on-rack noner part2)
		(on-rack noner part3)
		(ready-to-unload part4)
		(next-to-unload part4 part1)
		(next-to-unload part1 part2)
		(next-to-unload part2 part3)
		(next-to-unload part3 nonep)
		(no-rack-succ part1)
		(no-rack-pre part1)
		(no-rack-succ part2)
		(no-rack-pre part2)
		(no-rack-succ part3)
		(no-rack-pre part3)
		(no-rack-succ part4)
		(no-rack-pre part4)
		(size part1 n05)
		(size part2 n06)
		(size part3 n05)
		(size part4 n03)
		(level rack1 n00)
		(level rack2 n00)
		(level rack3 n00)
		(level rack4 n00)
		(unused rack1)
		(unused rack2)
		(unused rack3)
		(unused rack4)
		(needed-next nonep)
		(p-line-next plA part3)
		(p-line-succ part3 part4)
		(p-line-succ part4 part1)
		(p-line-succ part1 part2)
		(p-line-succ part2 nonep)
		(p-line-succ nonep nonep)
	)
  (:goal (and 
		(processed part1)
		(processed part2)
		(processed part3)
		(processed part4)
	))
  
)