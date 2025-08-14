

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(on b1 b2)
(on b2 b4)
(ontable b3)
(on b4 b6)
(on b5 b1)
(on b6 b7)
(ontable b7)
(on b8 b3)
(ontable b9)
(clear b5)
(clear b8)
(clear b9)
)
(:goal
(and
(on b2 b6)
(on b3 b9)
(on b4 b3)
(on b5 b7)
(on b6 b1)
(on b9 b8))
)
)


