

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 )
(:init
(handempty)
(ontable b1)
(ontable b2)
(on b3 b2)
(ontable b4)
(ontable b5)
(ontable b6)
(on b7 b5)
(clear b1)
(clear b3)
(clear b4)
(clear b6)
(clear b7)
)
(:goal
(and
(on b1 b2)
(on b2 b4)
(on b3 b6)
(on b5 b7)
(on b6 b1)
(on b7 b3))
)
)


