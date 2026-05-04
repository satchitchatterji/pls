action(0)::action(left);
action(1)::action(down);
action(2)::action(right);
action(3)::action(up).

sensor_value(0)::at_start.
sensor_value(1)::at_goal.
sensor_value(2)::in_top_row.
sensor_value(3)::in_left_col.

% Conservative start-state safety prior: discourage moving left/up from start cell.
unsafe_next :- at_start, action(left).
unsafe_next :- at_start, action(up).

safe_next :- \+ unsafe_next.
