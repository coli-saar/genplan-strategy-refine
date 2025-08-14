

(define (problem BW-rand-12)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(handempty)
(on b1 b5)
(ontable b2)
(on b3 b11)
(on b4 b10)
(on b5 b7)
(on b6 b3)
(on b7 b9)
(on b8 b6)
(ontable b9)
(on b10 b2)
(on b11 b4)
(on b12 b1)
(clear b8)
(clear b12)
)
(:goal
(and
(on b2 b6)
(on b3 b7)
(on b4 b5)
(on b5 b12)
(on b6 b9)
(on b7 b2)
(on b9 b8)
(on b10 b1)
(on b11 b10))
)
)


