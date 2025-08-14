

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(ontable b1)
(on b2 b5)
(on b3 b8)
(on b4 b6)
(on b5 b3)
(on b6 b2)
(on b7 b1)
(ontable b8)
(on b9 b7)
(clear b4)
(clear b9)
)
(:goal
(and
(on b3 b2)
(on b4 b5)
(on b5 b9)
(on b6 b1)
(on b7 b6)
(on b9 b8))
)
)


