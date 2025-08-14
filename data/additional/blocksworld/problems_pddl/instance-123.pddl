

(define (problem BW-rand-15)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 )
(:init
(handempty)
(on b1 b12)
(on b2 b3)
(on b3 b10)
(on b4 b2)
(on b5 b8)
(on b6 b7)
(on b7 b14)
(on b8 b1)
(on b9 b15)
(on b10 b11)
(ontable b11)
(on b12 b6)
(on b13 b5)
(on b14 b9)
(on b15 b4)
(clear b13)
)
(:goal
(and
(on b1 b11)
(on b2 b14)
(on b3 b1)
(on b4 b3)
(on b5 b15)
(on b7 b5)
(on b8 b12)
(on b12 b13)
(on b14 b7))
)
)


