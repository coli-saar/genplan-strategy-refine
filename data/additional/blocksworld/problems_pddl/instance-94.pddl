

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 )
(:init
(handempty)
(on b1 b3)
(on b2 b1)
(ontable b3)
(clear b2)
)
(:goal
(and
(on b2 b3))
)
)


