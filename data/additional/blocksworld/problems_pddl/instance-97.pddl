

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 )
(:init
(handempty)
(on b1 b2)
(on b2 b3)
(ontable b3)
(clear b1)
)
(:goal
(and
(on b2 b1))
)
)


