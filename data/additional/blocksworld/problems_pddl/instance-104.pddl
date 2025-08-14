

(define (problem BW-rand-11)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(handempty)
(on b1 b4)
(on b2 b9)
(ontable b3)
(on b4 b8)
(on b5 b2)
(ontable b6)
(ontable b7)
(ontable b8)
(on b9 b11)
(on b10 b7)
(on b11 b1)
(clear b3)
(clear b5)
(clear b6)
(clear b10)
)
(:goal
(and
(on b1 b11)
(on b3 b9)
(on b4 b10)
(on b5 b4)
(on b6 b7)
(on b8 b2)
(on b9 b8))
)
)


