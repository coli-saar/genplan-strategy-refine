

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 b6 )
(:init
(handempty)
(ontable b1)
(on b2 b5)
(on b3 b2)
(ontable b4)
(ontable b5)
(on b6 b1)
(clear b3)
(clear b4)
(clear b6)
)
(:goal
(and
(on b1 b4)
(on b2 b5)
(on b4 b6)
(on b5 b1))
)
)


