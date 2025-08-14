(define (domain puzzle)
   (:predicates
		(predicate_1 ?object1 ?object2)
        (predicate_2 ?object)
        (predicate_3 ?object)
        (predicate_4 ?object)
        (predicate_5)
	)

   (:action action_1
       :parameters (?object)
       :precondition (and (predicate_5))
       :effect (and (not (predicate_5)) (predicate_2 ?object) (predicate_4 ?object) (not (predicate_3 ?object))))

   (:action action_2
       :parameters (?object1 ?object2)
       :precondition (and (predicate_2 ?object1) (predicate_4 ?object1) (predicate_1 ?object1 ?object2) (predicate_3 ?object2))
       :effect (and (predicate_2 ?object2) (predicate_4 ?object2) (not (predicate_4 ?object1)) (not (predicate_3 ?object2))))
)
