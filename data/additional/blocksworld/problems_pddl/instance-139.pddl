

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 )
(:init
(handempty)
(on b1 b3)
(ontable b2)
(ontable b3)
(on b4 b1)
(on b5 b2)
(on b6 b4)
(clear b5)
(clear b6)
)
(:goal
(and
(on b3 b1)
(on b4 b3)
(on b5 b2))
)
)


