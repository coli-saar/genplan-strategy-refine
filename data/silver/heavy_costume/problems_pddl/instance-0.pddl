(define (problem heavy-pack-prob)
	(:domain shopping)
	(:objects o404 o570 o997)

(:init
    (shopping-cart-empty)
    (not-added o404)
    (not-added o570)
    (not-added o997)
    (cheaper o997 o404)
    (cheaper o997 o570)
    (cheaper o404 o570)
)

(:goal (and (added o404) (added o570) (added o997)))
)
