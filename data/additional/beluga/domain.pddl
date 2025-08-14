(define (domain beluga)
  (:requirements :strips :typing :equality :negative-preconditions :action-costs)
  (:types
		part - object
		rack - object
		part-type - object
		truck - object
		beluga-truck - truck
		fab-truck - truck
		production-line - object
		num - object
		num-swaps - object
)
  (:constants
		nonep - part
		noner - rack
		n00 - num
	)


  (:predicates
		(ready-to-unload ?p - part)
		(next-to-unload ?pc - part ?pn - part)
		(on-truck ?p - part ?t - truck)
		(empty ?t - truck)
		(on-rack ?r - rack ?p - part)
		(rack-succ ?p1 - part ?p2 - part)
		(no-rack-succ ?p - part)
		(no-rack-pre ?p - part)
		(p-line-next ?pl - production-line ?p - part)
		(p-line-succ ?p1 - part ?p2 - part)
		(needed-next ?p - part)
		(processed ?p - part)
		(unused ?r - rack)
		(size ?p - part ?n - num)
		(level ?r - rack ?n - num)
		(fit-n-sum ?n1 - num ?n2 - num ?n3 - num)
	)


	(:action unload-from-beluga
		:parameters (?p - part ?np - part ?t - beluga-truck)
		:precondition (and
			(ready-to-unload ?p)
			(next-to-unload ?p ?np)
			(empty ?t)
		)
		:effect (and
			(not (ready-to-unload ?p))
			(ready-to-unload ?np)
			(not (empty ?t))
			(on-truck ?p ?t)
		)
	)

	(:action put-on-rack-first-beluga
		:parameters (?t - beluga-truck ?p - part ?r - rack ?pSize - num)
		:precondition (and
			(on-truck ?p ?t)
			(on-rack noner ?p)
			(no-rack-pre ?p)
			(no-rack-succ ?p)
			(size ?p ?pSize)
			(level ?r n00)
			(fit-n-sum n00 ?pSize ?pSize)
		)
		:effect (and
			(not (on-truck ?p ?t))
			(empty ?t)
			(not (on-rack noner ?p))
			(on-rack ?r ?p)
			(not (unused ?r))
			(not (level ?r n00))
			(level ?r ?pSize)
		)
	)

	(:action put-on-truck-first-beluga
		:parameters (?t - beluga-truck ?p - part ?r - rack ?pSize - num)
		:precondition (and
			(empty ?t)
			(on-rack ?r ?p)
			(no-rack-pre ?p)
			(no-rack-succ ?p)
			(size ?p ?pSize)
			(level ?r ?pSize)
			(fit-n-sum n00 ?pSize ?pSize)
		)
		:effect (and
			(on-truck ?p ?t)
			(not (empty ?t))
			(on-rack noner ?p)
			(not (on-rack ?r ?p))
			(not (level ?r ?pSize))
			(level ?r n00)
		)
	)



	(:action put-on-rack-beluga
		:parameters (?t - beluga-truck ?p - part ?r - rack ?pFirst - part ?pSize - num ?rLevel - num ?newLevel - num)
		:precondition (and
			(not (= ?p ?pFirst))
			(on-truck ?p ?t)
			(on-rack ?r ?pFirst)
			(on-rack noner ?p)
			(no-rack-pre ?pFirst)
			(no-rack-pre ?p)
			(no-rack-succ ?p)
			(size ?p ?pSize)
			(level ?r ?rLevel)
			(fit-n-sum ?rLevel ?pSize ?newLevel)
		)
		:effect (and
			(not (on-truck ?p ?t))
			(empty ?t)
			(not (on-rack noner ?p))
			(on-rack ?r ?p)
			(not (no-rack-pre ?pFirst))
			(not (no-rack-succ ?p))
			(rack-succ ?p ?pFirst)
			(not (ready-to-unload ?p))
			(not (unused ?r))
			(not (level ?r ?rLevel))
			(level ?r ?newLevel)
		)
	)

	(:action put-on-truck-beluga
		:parameters (?t - beluga-truck ?p - part ?r - rack ?pFirst - part ?pSize - num ?rLevel - num ?newLevel - num)
		:precondition (and
			(not (= ?p ?pFirst))
			(empty ?t)
			(on-rack ?r ?pFirst)
			(on-rack ?r ?p)
			(no-rack-pre ?p)
			(rack-succ ?p ?pFirst)
			(size ?p ?pSize)
			(level ?r ?newLevel)
			(fit-n-sum ?rLevel ?pSize ?newLevel)
		)
		:effect (and
			(not (on-rack ?r ?p))
			(on-rack noner ?p)
			(on-truck ?p ?t)
			(not (empty ?t))
			(no-rack-pre ?pFirst)
			(no-rack-succ ?p)
			(not (rack-succ ?p ?pFirst))
			(not (level ?r ?newLevel))
			(level ?r ?rLevel)
		)
	)



	(:action set-next
		:parameters (?pl - production-line ?pA - part ?pn - part)
		:precondition (and
			(needed-next nonep)
			(p-line-next ?pl ?pA)
			(p-line-succ ?pA ?pn)
		)
		:effect (and
			(needed-next ?pA)
			(not (needed-next nonep))
			(not (p-line-next ?pl ?pA))
			(p-line-next ?pl ?pn)
		)
	)



	(:action put-on-truck-fab-last
		:parameters (?t - fab-truck ?r - rack ?p - part ?pSize - num ?rLevel - num ?newLevel - num)
		:precondition (and
			(empty ?t)
			(on-rack ?r ?p)
			(no-rack-pre ?p)
			(no-rack-succ ?p)
			(size ?p ?pSize)
			(level ?r ?pSize)
		)
		:effect (and
			(not (on-rack ?r ?p))
			(on-rack noner ?p)
			(not (empty ?t))
			(on-truck ?p ?t)
			(not (level ?r ?pSize))
			(level ?r n00)
		)
	)


	(:action put-on-rack-fab-last
		:parameters (?t - fab-truck ?r - rack ?p - part ?pSize - num ?rLevel - num ?newLevel - num)
		:precondition (and
			(on-truck ?p ?t)
			(on-rack noner ?p)
			(no-rack-pre ?p)
			(no-rack-succ ?p)
			(size ?p ?pSize)
			(level ?r n00)
		)
		:effect (and
			(not (on-rack noner ?p))
			(on-rack ?r ?p)
			(empty ?t)
			(not (on-truck ?p ?t))
			(not (level ?r n00))
			(level ?r ?pSize)
		)
	)



	(:action put-on-truck-fab
		:parameters (?t - fab-truck ?r - rack ?pRight - part ?pBefore - part ?pSize - num ?rLevel - num ?newLevel - num)
		:precondition (and
			(not (= ?pRight ?pBefore))
			(empty ?t)
			(on-rack ?r ?pRight)
			(on-rack ?r ?pBefore)
			(rack-succ ?pBefore ?pRight)
			(no-rack-succ ?pRight)
			(size ?pRight ?pSize)
			(level ?r ?rLevel)
			(fit-n-sum ?newLevel ?pSize ?rLevel)
		)
		:effect (and
			(not (empty ?t))
			(on-truck ?pRight ?t)
			(not (on-rack ?r ?pRight))
			(on-rack noner ?pRight)
			(not (rack-succ ?pBefore ?pRight))
			(no-rack-pre ?pRight)
			(no-rack-succ ?pBefore)
			(not (level ?r ?rLevel))
			(level ?r ?newLevel)
		)
	)


	(:action put-on-rack-fab
		:parameters (?t - fab-truck ?r - rack ?pRight - part ?pBefore - part ?pSize - num ?rLevel - num ?newLevel - num)
		:precondition (and
			(not (= ?pRight ?pBefore))
			(on-truck ?pRight ?t)
			(on-rack noner ?pRight)
			(no-rack-pre ?pRight)
			(no-rack-succ ?pRight)
			(on-rack ?r ?pBefore)
			(no-rack-succ ?pBefore)
			(size ?pRight ?pSize)
			(level ?r ?newLevel)
			(fit-n-sum ?newLevel ?pSize ?rLevel)
		)
		:effect (and
			(empty ?t)
			(not (on-truck ?pRight ?t))
			(not (on-rack noner ?pRight))
			(on-rack ?r ?pRight)
			(not (no-rack-pre ?pRight))
			(rack-succ ?pBefore ?pRight)
			(not (level ?r ?newLevel))
			(level ?r ?rLevel)
		)
	)

	(:action to-fab
		:parameters (?t - fab-truck ?p - part)
		:precondition (and
			(needed-next ?p)
			(on-truck ?p ?t)
		)
		:effect (and
			(not (on-truck ?p ?t))
			(empty ?t)
			(not (needed-next ?p))
			(needed-next nonep)
			(processed ?p)
		)
	)

)
