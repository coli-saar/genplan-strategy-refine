

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(on b1 b3)
(on b2 b8)
(on b3 b9)
(on b4 b7)
(ontable b5)
(on b6 b2)
(on b7 b6)
(on b8 b5)
(on b9 b4)
(clear b1)
)
(:goal
(and
(on b1 b8)
(on b2 b1)
(on b4 b6)
(on b5 b4)
(on b6 b3)
(on b8 b9)
(on b9 b7))
)
)


