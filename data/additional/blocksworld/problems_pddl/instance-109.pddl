

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(on b1 b8)
(on b2 b1)
(ontable b3)
(on b4 b6)
(ontable b5)
(on b6 b5)
(ontable b7)
(ontable b8)
(on b9 b3)
(clear b2)
(clear b4)
(clear b7)
(clear b9)
)
(:goal
(and
(on b1 b2)
(on b2 b7)
(on b3 b4)
(on b6 b5)
(on b7 b8)
(on b8 b3))
)
)


