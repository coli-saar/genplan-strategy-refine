

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 )
(:init
(handempty)
(on b1 b7)
(on b2 b4)
(ontable b3)
(ontable b4)
(on b5 b1)
(on b6 b5)
(ontable b7)
(on b8 b3)
(clear b2)
(clear b6)
(clear b8)
)
(:goal
(and
(on b1 b5)
(on b2 b1)
(on b4 b2)
(on b5 b6)
(on b7 b3))
)
)


