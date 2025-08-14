(define (problem gripper-1-37-7)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 room11 room12 room13 room14 room15 room16 room17 room18 room19 room20 room21 room22 room23 room24 room25 room26 room27 room28 room29 room30 room31 room32 room33 room34 room35 room36 room37 - room
ball1 ball2 ball3 ball4 ball5 ball6 ball7 - object)
(:init
(at-robby robot1 room16)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball1 room26)
(at ball2 room35)
(at ball3 room15)
(at ball4 room29)
(at ball5 room1)
(at ball6 room32)
(at ball7 room5)
)
(:goal
(and
(at ball1 room34)
(at ball2 room32)
(at ball3 room22)
(at ball4 room19)
(at ball5 room7)
(at ball6 room29)
(at ball7 room20)
)
)
)
