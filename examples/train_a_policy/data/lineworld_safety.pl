action(0)::action(left);
action(1)::action(stay);
action(2)::action(right).

sensor_value(0)::near_left_edge.
sensor_value(1)::near_right_edge.
sensor_value(2)::at_center.

unsafe_next :- near_left_edge, action(left).
unsafe_next :- near_right_edge, action(right).

safe_next :- \+ unsafe_next.
safe_action(A) :- action(A).
