(define (problem gripper-3-44-5)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 room11 room12 room13 room14 room15 room16 room17 room18 room19 room20 room21 room22 room23 room24 room25 room26 room27 room28 room29 room30 room31 room32 room33 room34 room35 room36 room37 room38 room39 room40 room41 room42 room43 room44 - room
ball1 ball2 ball3 ball4 ball5 - object)
(:init
(at-robby robot1 room6)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room10)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room14)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at ball1 room44)
(at ball2 room10)
(at ball3 room34)
(at ball4 room34)
(at ball5 room44)
)
(:goal
(and
(at ball1 room22)
(at ball2 room32)
(at ball3 room33)
(at ball4 room8)
(at ball5 room17)
)
)
)
