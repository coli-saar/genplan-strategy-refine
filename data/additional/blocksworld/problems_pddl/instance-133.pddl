

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 )
(:init
(handempty)
(on b1 b6)
(on b2 b7)
(on b3 b2)
(ontable b4)
(on b5 b1)
(ontable b6)
(on b7 b5)
(ontable b8)
(clear b3)
(clear b4)
(clear b8)
)
(:goal
(and
(on b2 b6)
(on b4 b7)
(on b5 b4)
(on b7 b2))
)
)


