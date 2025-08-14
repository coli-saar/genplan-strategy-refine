(define (domain shopping)
   (:predicates
		(cheaper ?item1 ?item2)
        (added ?item)
        (not-added ?item)
        (nothing-above ?item)
        (shopping-cart-empty)
	)

   (:action add-first
       :parameters (?item)
       :precondition (and (shopping-cart-empty))
       :effect (and (not (shopping-cart-empty)) (added ?item) (nothing-above ?item) (not (not-added ?item))))

   (:action add-to-prev
       :parameters (?bottom ?top)
       :precondition (and (added ?bottom) (nothing-above ?bottom) (cheaper ?bottom ?top) (not-added ?top))
       :effect (and (added ?top) (nothing-above ?top) (not (nothing-above ?bottom)) (not (not-added ?top))))
)
