

(define (problem BW-rand-13)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 )
(:init
(handempty)
(ontable b1)
(on b2 b10)
(on b3 b8)
(on b4 b7)
(ontable b5)
(on b6 b5)
(on b7 b6)
(on b8 b1)
(on b9 b13)
(ontable b10)
(ontable b11)
(ontable b12)
(on b13 b4)
(clear b2)
(clear b3)
(clear b9)
(clear b11)
(clear b12)
)
(:goal
(and
(on b3 b4)
(on b4 b13)
(on b5 b12)
(on b6 b2)
(on b8 b1)
(on b11 b5)
(on b12 b10))
)
)


