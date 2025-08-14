(define (problem puzzle_task)
	(:domain puzzle)
	(:objects o404 o570 o997)

(:init
    (predicate_5)
    (predicate_3 o404)
    (predicate_3 o570)
    (predicate_3 o997)
    (predicate_1 o997 o404)
    (predicate_1 o997 o570)
    (predicate_1 o404 o570)
)

(:goal (and (predicate_2 o404) (predicate_2 o570) (predicate_2 o997)))
)
