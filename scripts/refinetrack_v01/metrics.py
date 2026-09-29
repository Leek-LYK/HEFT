"""Pure completion semantics shared by evaluator and tests."""
def completion_result(frames, planned_transitions, terminated, reasons):
    if planned_transitions <= 0:
        raise ValueError('planned_transitions must be positive')
    success = not terminated and 'motion_timeout' in reasons and frames >= planned_transitions
    return bool(success), min(frames / planned_transitions, 1.)
