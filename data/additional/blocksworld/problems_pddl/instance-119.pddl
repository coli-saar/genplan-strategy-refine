

(define (problem BW-rand-12)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
(handempty)
(on b1 b4)
(on b2 b8)
(on b3 b1)
(on b4 b11)
(on b5 b9)
(ontable b6)
(ontable b7)
(on b8 b12)
(ontable b9)
(on b10 b7)
(on b11 b2)
(on b12 b5)
(clear b3)
(clear b6)
(clear b10)
)
(:goal
(and
(on b1 b2)
(on b2 b4)
(on b4 b9)
(on b5 b8)
(on b6 b1)
(on b8 b3)
(on b9 b7)
(on b10 b11)
(on b11 b12)
(on b12 b5))
)
)


