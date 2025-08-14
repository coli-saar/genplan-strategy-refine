(define (problem grid-6)
(:domain grid-visit-all)
(:objects 
	loc-x1-y0
	loc-x1-y1
	loc-x2-y0
	loc-x3-y0
	loc-x3-y1
	loc-x4-y0
	loc-x5-y0
	loc-x5-y1
- place 
        
)
(:init
	(at-robot loc-x3-y0)
	(visited loc-x3-y0)
	(connected loc-x1-y0 loc-x2-y0)
 	(connected loc-x1-y0 loc-x1-y1)
 	(connected loc-x1-y1 loc-x1-y0)
 	(connected loc-x2-y0 loc-x1-y0)
 	(connected loc-x2-y0 loc-x3-y0)
 	(connected loc-x3-y0 loc-x2-y0)
 	(connected loc-x3-y0 loc-x4-y0)
 	(connected loc-x3-y0 loc-x3-y1)
 	(connected loc-x3-y1 loc-x3-y0)
 	(connected loc-x4-y0 loc-x3-y0)
 	(connected loc-x4-y0 loc-x5-y0)
 	(connected loc-x5-y0 loc-x4-y0)
 	(connected loc-x5-y0 loc-x5-y1)
 	(connected loc-x5-y1 loc-x5-y0)
 
)
(:goal
(and 
	(visited loc-x1-y0)
	(visited loc-x1-y1)
	(visited loc-x2-y0)
	(visited loc-x3-y0)
	(visited loc-x3-y1)
	(visited loc-x4-y0)
	(visited loc-x5-y0)
)
)
)
