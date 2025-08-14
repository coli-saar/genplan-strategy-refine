(define (problem grid-8)
(:domain grid-visit-all)
(:objects 
	loc-x0-y1
	loc-x1-y0
	loc-x2-y0
	loc-x3-y0
	loc-x4-y1
	loc-x5-y1
	loc-x6-y1
	loc-x6-y2
	loc-x7-y1
- place 
        
)
(:init
	(at-robot loc-x7-y1)
	(visited loc-x7-y1)
	(connected loc-x1-y0 loc-x2-y0)
 	(connected loc-x2-y0 loc-x1-y0)
 	(connected loc-x2-y0 loc-x3-y0)
 	(connected loc-x3-y0 loc-x2-y0)
 	(connected loc-x4-y1 loc-x5-y1)
 	(connected loc-x5-y1 loc-x4-y1)
 	(connected loc-x5-y1 loc-x6-y1)
 	(connected loc-x6-y1 loc-x5-y1)
 	(connected loc-x6-y1 loc-x7-y1)
 	(connected loc-x6-y1 loc-x6-y2)
 	(connected loc-x6-y2 loc-x6-y1)
 	(connected loc-x7-y1 loc-x6-y1)
 
)
(:goal
(and 
	(visited loc-x4-y1)
	(visited loc-x7-y1)
)
)
)
