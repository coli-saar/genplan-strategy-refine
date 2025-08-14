(define (problem gripper-2-33-7)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 room11 room12 room13 room14 room15 room16 room17 room18 room19 room20 room21 room22 room23 room24 room25 room26 room27 room28 room29 room30 room31 room32 room33 - room
ball1 ball2 ball3 ball4 ball5 ball6 ball7 - object)
(:init
(at-robby robot1 room24)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room25)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball1 room29)
(at ball2 room5)
(at ball3 room2)
(at ball4 room6)
(at ball5 room17)
(at ball6 room3)
(at ball7 room25)
)
(:goal
(and
(at ball1 room2)
(at ball2 room11)
(at ball3 room12)
(at ball4 room6)
(at ball5 room20)
(at ball6 room31)
(at ball7 room9)
)
)
)
