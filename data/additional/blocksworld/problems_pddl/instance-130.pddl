

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(on b1 b3)
(on b2 b4)
(on b3 b5)
(ontable b4)
(on b5 b7)
(ontable b6)
(on b7 b9)
(on b8 b2)
(on b9 b6)
(clear b1)
(clear b8)
)
(:goal
(and
(on b1 b3)
(on b4 b1)
(on b5 b8)
(on b6 b2)
(on b7 b4)
(on b8 b6))
)
)


