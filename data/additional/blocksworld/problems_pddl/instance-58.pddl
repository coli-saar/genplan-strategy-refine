

(define (problem BW-rand-12)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(handempty)
(on b1 b6)
(ontable b2)
(on b3 b1)
(on b4 b2)
(on b5 b12)
(on b6 b10)
(on b7 b11)
(on b8 b4)
(on b9 b5)
(ontable b10)
(ontable b11)
(on b12 b7)
(clear b3)
(clear b8)
(clear b9)
)
(:goal
(and
(on b1 b6)
(on b3 b10)
(on b5 b1)
(on b6 b7)
(on b7 b2)
(on b8 b5)
(on b9 b8)
(on b11 b3)
(on b12 b11))
)
)


