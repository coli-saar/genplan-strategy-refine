(define (problem gripper-2-45-2)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 room11 room12 room13 room14 room15 room16 room17 room18 room19 room20 room21 room22 room23 room24 room25 room26 room27 room28 room29 room30 room31 room32 room33 room34 room35 room36 room37 room38 room39 room40 room41 room42 room43 room44 room45 - room
ball1 ball2 - object)
(:init
(at-robby robot1 room24)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room21)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball1 room34)
(at ball2 room44)
)
(:goal
(and
(at ball1 room38)
(at ball2 room41)
)
)
)
