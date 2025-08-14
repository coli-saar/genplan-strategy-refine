

(define (problem BW-rand-11)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(handempty)
(ontable b1)
(on b2 b11)
(on b3 b1)
(on b4 b9)
(on b5 b2)
(ontable b6)
(on b7 b10)
(on b8 b7)
(on b9 b6)
(on b10 b4)
(on b11 b3)
(clear b5)
(clear b8)
)
(:goal
(and
(on b1 b5)
(on b2 b6)
(on b3 b1)
(on b5 b7)
(on b6 b10)
(on b8 b9)
(on b9 b11)
(on b10 b4))
)
)


