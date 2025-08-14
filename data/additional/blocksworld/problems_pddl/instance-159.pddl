

(define (problem BW-rand-13)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 )
(:init
(handempty)
(ontable b1)
(ontable b2)
(on b3 b10)
(on b4 b8)
(on b5 b1)
(on b6 b2)
(on b7 b4)
(on b8 b11)
(on b9 b7)
(on b10 b13)
(ontable b11)
(on b12 b3)
(on b13 b5)
(clear b6)
(clear b9)
(clear b12)
)
(:goal
(and
(on b1 b4)
(on b3 b2)
(on b4 b5)
(on b5 b6)
(on b8 b11)
(on b9 b7)
(on b10 b3)
(on b12 b1)
(on b13 b9))
)
)


