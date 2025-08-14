

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(on b1 b9)
(ontable b2)
(on b3 b7)
(ontable b4)
(on b5 b1)
(ontable b6)
(on b7 b5)
(ontable b8)
(on b9 b4)
(clear b2)
(clear b3)
(clear b6)
(clear b8)
)
(:goal
(and
(on b1 b2)
(on b2 b7)
(on b4 b3)
(on b5 b1)
(on b6 b5)
(on b9 b6))
)
)


