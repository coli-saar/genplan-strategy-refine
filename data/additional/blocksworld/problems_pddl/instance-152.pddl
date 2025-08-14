

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 b7 )
(:init
(handempty)
(on b1 b5)
(on b2 b1)
(on b3 b2)
(on b4 b6)
(on b5 b7)
(ontable b6)
(ontable b7)
(clear b3)
(clear b4)
)
(:goal
(and
(on b1 b5)
(on b3 b2)
(on b6 b7))
)
)


