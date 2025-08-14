

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(on b1 b6)
(on b2 b4)
(on b3 b1)
(on b4 b3)
(on b5 b7)
(ontable b6)
(ontable b7)
(ontable b8)
(on b9 b5)
(clear b2)
(clear b8)
(clear b9)
)
(:goal
(and
(on b1 b4)
(on b2 b3)
(on b3 b7)
(on b5 b9)
(on b7 b5)
(on b9 b1))
)
)


