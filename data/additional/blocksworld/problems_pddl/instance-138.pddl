

(define (problem BW-rand-14)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 )
(:init
(handempty)
(ontable b1)
(on b2 b11)
(on b3 b9)
(on b4 b5)
(on b5 b12)
(on b6 b7)
(on b7 b10)
(on b8 b13)
(on b9 b2)
(on b10 b14)
(on b11 b1)
(ontable b12)
(on b13 b3)
(on b14 b8)
(clear b4)
(clear b6)
)
(:goal
(and
(on b1 b2)
(on b3 b1)
(on b5 b4)
(on b6 b8)
(on b8 b14)
(on b11 b9)
(on b12 b10)
(on b13 b3)
(on b14 b7))
)
)


