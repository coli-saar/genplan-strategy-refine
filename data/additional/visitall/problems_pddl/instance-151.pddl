(define (problem grid-2)
(:domain grid-visit-all)
(:objects 
	loc-x0-y0
	loc-x0-y4
	loc-x0-y5
	loc-x1-y0
	loc-x1-y1
	loc-x1-y2
	loc-x1-y3
- place 
        
)
(:init
	(at-robot loc-x1-y2)
	(visited loc-x1-y2)
	(connected loc-x0-y0 loc-x1-y0)
 	(connected loc-x0-y4 loc-x0-y5)
 	(connected loc-x0-y5 loc-x0-y4)
 	(connected loc-x1-y0 loc-x0-y0)
 	(connected loc-x1-y0 loc-x1-y1)
 	(connected loc-x1-y1 loc-x1-y0)
 	(connected loc-x1-y1 loc-x1-y2)
 	(connected loc-x1-y2 loc-x1-y1)
 	(connected loc-x1-y2 loc-x1-y3)
 	(connected loc-x1-y3 loc-x1-y2)
 
)
(:goal
(and 
	(visited loc-x1-y2)
)
)
)
