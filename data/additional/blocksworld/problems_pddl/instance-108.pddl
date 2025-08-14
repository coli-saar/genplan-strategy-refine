

(define (problem BW-rand-11)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 )
(:init
(handempty)
(ontable b1)
(ontable b2)
(on b3 b8)
(ontable b4)
(on b5 b7)
(on b6 b11)
(ontable b7)
(ontable b8)
(on b9 b5)
(on b10 b4)
(on b11 b1)
(clear b2)
(clear b3)
(clear b6)
(clear b9)
(clear b10)
)
(:goal
(and
(on b1 b10)
(on b3 b8)
(on b4 b3)
(on b6 b7)
(on b7 b11)
(on b8 b2)
(on b9 b4)
(on b10 b6)
(on b11 b9))
)
)


