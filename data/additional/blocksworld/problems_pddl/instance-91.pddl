

(define (problem BW-rand-13)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 )
(:init
(handempty)
(on b1 b11)
(on b2 b7)
(ontable b3)
(ontable b4)
(ontable b5)
(on b6 b5)
(on b7 b8)
(on b8 b3)
(on b9 b6)
(on b10 b4)
(on b11 b13)
(on b12 b9)
(on b13 b2)
(clear b1)
(clear b10)
(clear b12)
)
(:goal
(and
(on b1 b8)
(on b3 b4)
(on b4 b10)
(on b7 b5)
(on b8 b13)
(on b9 b2)
(on b11 b3)
(on b12 b7))
)
)


