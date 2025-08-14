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
		rack5 - rack
		part01 - part
		part02 - part
		part03 - part
		part04 - part
		part05 - part
		part06 - part
		part07 - part
		part08 - part
		part09 - part
		part10 - part
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
		
		(on-rack noner part05)
		(on-rack noner part09)
		(on-rack noner part10)
		(on-rack noner part07)
		(on-rack noner part04)
		(on-rack noner part06)
		(on-rack noner part03)
		(on-rack noner part02)
		(on-rack noner part08)
		(on-rack noner part01)
		(ready-to-unload part05)
		(next-to-unload part05 part09)
		(next-to-unload part09 part10)
		(next-to-unload part10 part07)
		(next-to-unload part07 part04)
		(next-to-unload part04 part06)
		(next-to-unload part06 part03)
		(next-to-unload part03 part02)
		(next-to-unload part02 part08)
		(next-to-unload part08 part01)
		(next-to-unload part01 nonep)
		(no-rack-succ part01)
		(no-rack-pre part01)
		(no-rack-succ part02)
		(no-rack-pre part02)
		(no-rack-succ part03)
		(no-rack-pre part03)
		(no-rack-succ part04)
		(no-rack-pre part04)
		(no-rack-succ part05)
		(no-rack-pre part05)
		(no-rack-succ part06)
		(no-rack-pre part06)
		(no-rack-succ part07)
		(no-rack-pre part07)
		(no-rack-succ part08)
		(no-rack-pre part08)
		(no-rack-succ part09)
		(no-rack-pre part09)
		(no-rack-succ part10)
		(no-rack-pre part10)
		(size part01 n06)
		(size part02 n03)
		(size part03 n03)
		(size part04 n05)
		(size part05 n05)
		(size part06 n05)
		(size part07 n05)
		(size part08 n03)
		(size part09 n06)
		(size part10 n06)
		(level rack1 n00)
		(level rack2 n00)
		(level rack3 n00)
		(level rack4 n00)
		(level rack5 n00)
		(unused rack1)
		(unused rack2)
		(unused rack3)
		(unused rack4)
		(unused rack5)
		(needed-next nonep)
		(p-line-next plA part05)
		(p-line-succ part05 part10)
		(p-line-succ part10 part06)
		(p-line-succ part06 part01)
		(p-line-succ part01 part07)
		(p-line-succ part07 part02)
		(p-line-succ part02 part04)
		(p-line-succ part04 part09)
		(p-line-succ part09 part08)
		(p-line-succ part08 part03)
		(p-line-succ part03 nonep)
		(p-line-succ nonep nonep)
	)
  (:goal (and 
		(processed part01)
		(processed part02)
		(processed part03)
		(processed part04)
		(processed part05)
		(processed part06)
		(processed part07)
		(processed part08)
		(processed part09)
		(processed part10)
	))
  
)