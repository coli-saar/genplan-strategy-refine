

(define (problem BW-rand-11)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(handempty)
(on b1 b7)
(on b2 b11)
(ontable b3)
(ontable b4)
(on b5 b4)
(ontable b6)
(on b7 b6)
(ontable b8)
(ontable b9)
(on b10 b9)
(on b11 b10)
(clear b1)
(clear b2)
(clear b3)
(clear b5)
(clear b8)
)
(:goal
(and
(on b1 b11)
(on b3 b10)
(on b4 b1)
(on b5 b6)
(on b6 b7)
(on b7 b4)
(on b8 b5)
(on b9 b3)
(on b11 b9))
)
)


