from __future__ import annotations

def eye_mapping(participant=None, recording=None, session=None, timestamp=None, timestamp_device=None, x=None, y=None, z=None, left_x=None, left_y=None, right_x=None, right_y=None, gaze_valid=None, left_valid=None, right_valid=None, confidence=None, pupil_left=None, pupil_right=None, pupil_left_valid=None, pupil_right_valid=None, fixation_id=None, blink_id=None, trial=None, item=None, stimulus=None, condition=None, response=None, score=None, response_time=None, event_name=None, event_value=None, event_type=None, biometric_channels=None, extra=None):
    """Construct a generic import mapping; `None` fields are dropped as in R."""
    vals=locals().copy(); extra=vals.pop('extra') or {}
    vals={k:v for k,v in vals.items() if v is not None}; vals.update(extra)
    return vals
