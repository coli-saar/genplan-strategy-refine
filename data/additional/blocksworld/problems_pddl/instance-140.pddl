

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(ontable b1)
(on b2 b7)
(on b3 b8)
(on b4 b3)
(on b5 b6)
(on b6 b1)
(ontable b7)
(on b8 b5)
(on b9 b2)
(clear b4)
(clear b9)
)
(:goal
(and
(on b2 b6)
(on b3 b5)
(on b4 b2)
(on b8 b3)
(on b9 b8))
)
)


