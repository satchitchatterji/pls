action(0)::action(a0);
action(1)::action(a1);
action(2)::action(a2).

sensor_value(0)::s0.
sensor_value(1)::s1.
sensor_value(2)::s2.

unsafe_next :- s0, action(a0).
unsafe_next :- s1, action(a1).

safe_next :- \+ unsafe_next.
