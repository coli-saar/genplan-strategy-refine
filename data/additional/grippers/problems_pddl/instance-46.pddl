(define (problem gripper-1-27-3)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room1 room2 room3 room4 room5 room6 room7 room8 room9 room10 room11 room12 room13 room14 room15 room16 room17 room18 room19 room20 room21 room22 room23 room24 room25 room26 room27 - room
ball1 ball2 ball3 - object)
(:init
(at-robby robot1 room25)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball1 room5)
(at ball2 room20)
(at ball3 room14)
)
(:goal
(and
(at ball1 room9)
(at ball2 room24)
(at ball3 room25)
)
)
)
