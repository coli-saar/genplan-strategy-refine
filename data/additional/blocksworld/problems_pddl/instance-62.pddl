

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(ontable b1)
(on b2 b5)
(on b3 b6)
(on b4 b9)
(on b5 b8)
(on b6 b2)
(on b7 b4)
(on b8 b1)
(ontable b9)
(clear b3)
(clear b7)
)
(:goal
(and
(on b2 b4)
(on b4 b8)
(on b5 b1)
(on b6 b9)
(on b7 b2)
(on b9 b5))
)
)


