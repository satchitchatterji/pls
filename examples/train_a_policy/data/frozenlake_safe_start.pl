action(0)::action(left);
action(1)::action(down);
action(2)::action(right);
action(3)::action(up).

sensor_value(0)::left_to_hole.
sensor_value(1)::down_to_hole.
sensor_value(2)::right_to_hole.
sensor_value(3)::up_to_hole.

unsafe_next :- left_to_hole, action(left).
unsafe_next :- down_to_hole, action(down).
unsafe_next :- right_to_hole, action(right).
unsafe_next :- up_to_hole, action(up).

safe_next :- \+ unsafe_next.
