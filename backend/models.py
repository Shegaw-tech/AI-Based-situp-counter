from typing import Dict, Any
from dataclasses import dataclass
@dataclass
class SessionData:
    count: int = 0
    is_in_rep: bool = False
    last_gesture_time: float = 0
    rep_start_time: float = 0
    form_metrics: Dict[str, Any] = None

    def __post_init__(self):
        self.form_metrics = {
            'shoulder_symmetry': 0.0,
            'back_angle': 180.0,
            'torso_angle': 180.0
        }