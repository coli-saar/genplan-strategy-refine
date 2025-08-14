

(define (problem BW-rand-9)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 )
(:init
(handempty)
(ontable b1)
(ontable b2)
(on b3 b7)
(on b4 b3)
(on b5 b9)
(on b6 b4)
(ontable b7)
(ontable b8)
(on b9 b8)
(clear b1)
(clear b2)
(clear b5)
(clear b6)
)
(:goal
(and
(on b1 b9)
(on b2 b1)
(on b6 b4)
(on b7 b2)
(on b8 b7))
)
)


