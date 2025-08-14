(define (problem gripper-1-10-9)
(:domain gripper-strips)(:objects robot_0 - robot
right_gripper_0 left_gripper_0 - gripper
room_0 room_1 room_2 room_3 room_4 room_5 room_6 room_7 room_8 room_9 - room
ball_0 ball_1 ball_2 ball_3 ball_4 ball_5 ball_6 ball_7 ball_8 - object
)
(:init
(at-robby robot_0 room_4)
(free robot_0 right_gripper_0)
(free robot_0 left_gripper_0)
(at ball_0 room_2)
(at ball_1 room_0)
(at ball_2 room_9)
(at ball_3 room_0)
(at ball_4 room_6)
(at ball_5 room_8)
(at ball_6 room_4)
(at ball_7 room_7)
(at ball_8 room_9)
)
(:goal
(and
(at ball_0 room_9)
(at ball_1 room_0)
(at ball_2 room_9)
(at ball_3 room_8)
(at ball_4 room_2)
(at ball_5 room_0)
(at ball_6 room_1)
(at ball_7 room_7)
(at ball_8 room_2)
)
)
)
