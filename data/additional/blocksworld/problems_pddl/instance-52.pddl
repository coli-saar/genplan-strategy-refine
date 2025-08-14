

(define (problem BW-rand-13)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 )
(:init
(handempty)
(on b1 b13)
(ontable b2)
(on b3 b11)
(ontable b4)
(on b5 b6)
(on b6 b3)
(ontable b7)
(on b8 b1)
(on b9 b4)
(on b10 b7)
(on b11 b2)
(on b12 b8)
(on b13 b5)
(clear b9)
(clear b10)
(clear b12)
)
(:goal
(and
(on b1 b12)
(on b5 b1)
(on b6 b8)
(on b7 b5)
(on b10 b4)
(on b11 b7)
(on b13 b6))
)
)


