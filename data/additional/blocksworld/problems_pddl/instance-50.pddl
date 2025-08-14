

(define (problem BW-rand-12)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(handempty)
(on b1 b2)
(on b2 b4)
(on b3 b8)
(on b4 b3)
(ontable b5)
(on b6 b12)
(on b7 b9)
(on b8 b10)
(ontable b9)
(on b10 b11)
(ontable b11)
(on b12 b5)
(clear b1)
(clear b6)
(clear b7)
)
(:goal
(and
(on b3 b5)
(on b4 b10)
(on b6 b1)
(on b7 b8)
(on b8 b11)
(on b9 b6)
(on b12 b4))
)
)


