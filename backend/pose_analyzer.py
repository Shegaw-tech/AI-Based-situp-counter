import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PoseAnalyzer")

class PoseAnalyzer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=False
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.thresholds = {
            'start': {
                'clap_hold_time': 0.5,
                'shoulder_lift_min': 0.10,
                'head_lift_min': 0.10,
                'torso_angle_max': 130
            },
            'end': {
                'shoulder_lift_max': 0.03,
                'head_lift_max': 0.03
            }
        }

        self.gesture_params = {
            'min_arm_angle': 150,
            'wrist_nose_offset': 0.08,
            'min_visibility': 0.5,
            'cooldown': 1.5
        }

        self.form_thresholds = {
            'max_shoulder_asymmetry': 0.2,
            'min_back_angle': 160
        }

        self.finger_start_time = None

    def process_frame(self, frame, session_data) -> Dict[str, Any]:
        self._init_session_data(session_data)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        hand_results = self.hands.process(frame_rgb)
        annotated_frame = frame.copy()
        current_time = datetime.now().timestamp()

        response = {
            'annotated_frame': annotated_frame,
            'feedback': [],
            'completed_rep': False,
            'form_metrics': session_data.form_metrics,
            'count': session_data.count,
            'debug': {
                'timestamp': current_time,
                'landmarks_visible': False
            }
        }

        if not results.pose_landmarks:
            print("No landmarks detected")
            return response

        self.mp_drawing.draw_landmarks(
            annotated_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
        )
        landmarks = results.pose_landmarks.landmark
        response['debug']['landmarks_visible'] = True

        if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
            fingers_count_total = 0
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                fingers_count = self._count_fingers(hand_landmarks, hand_label, frame.shape)
                fingers_count_total += fingers_count

            if fingers_count_total == 7:
                if self.finger_start_time is None:
                    self.finger_start_time = current_time
                elif current_time - self.finger_start_time > self.thresholds['start']['clap_hold_time']:
                    if current_time - session_data.last_gesture_time > self.gesture_params['cooldown']:
                        session_data.count = 0
                        session_data.is_in_rep = False
                        session_data.last_gesture_time = current_time
                        self.finger_start_time = None
                        response['debug']['gesture_detected'] = True
                        cv2.putText(annotated_frame, "COUNT RESET BY 7 FINGERS", (50, 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                self.finger_start_time = None
        else:
            self.finger_start_time = None

        # Extract landmarks
        nose = self._get_landmark(landmarks, self.mp_pose.PoseLandmark.NOSE)
        left_shoulder = self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = self._get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip = self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = self._get_landmark(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP)
        left_ear = self._get_landmark(landmarks, self.mp_pose.PoseLandmark.LEFT_EAR)

        if all([nose, left_shoulder, left_hip, right_hip]):
            shoulder_y = left_shoulder[1]
            hip_y = left_hip[1]
            head_lift = hip_y - nose[1]
            shoulder_lift = hip_y - shoulder_y
            torso_angle = self._calculate_angle(left_ear, left_shoulder, left_hip) if left_ear else 180
            back_angle = self._calculate_back_angle(left_shoulder, left_hip)
            shoulder_symmetry = abs(left_shoulder[1] - right_shoulder[1]) if right_shoulder else 0

            response['debug'].update({
                'head_lift': head_lift,
                'shoulder_lift': shoulder_lift,
                'torso_angle': torso_angle,
                'back_angle': back_angle,
                'shoulder_symmetry': shoulder_symmetry,
                'is_in_rep': session_data.is_in_rep
            })

            session_data.form_metrics.update({
                'shoulder_symmetry': shoulder_symmetry,
                'back_angle': back_angle,
                'torso_angle': torso_angle
            })

            if not session_data.is_in_rep:
                if (shoulder_lift >= self.thresholds['start']['shoulder_lift_min'] and
                        head_lift >= self.thresholds['start']['head_lift_min'] and
                        torso_angle <= self.thresholds['start']['torso_angle_max']):
                    session_data.is_in_rep = True
                    session_data.rep_start_time = current_time
                    response['debug']['rep_start'] = True

            elif session_data.is_in_rep:
                if (shoulder_lift <= self.thresholds['end']['shoulder_lift_max'] and
                        head_lift <= self.thresholds['end']['head_lift_max']):
                    session_data.is_in_rep = False
                    session_data.count += 1
                    response['completed_rep'] = True
                    response['debug']['rep_end'] = True

            feedback = []
            if back_angle < self.form_thresholds['min_back_angle']:
                feedback.append("Maintain straight back")
            response['feedback'] = feedback

        response['annotated_frame'] = annotated_frame
        response['count'] = session_data.count
        return response

    def _init_session_data(self, session_data):
        defaults = {
            'count': 0,
            'is_in_rep': False,
            'last_gesture_time': 0,
            'rep_start_time': 0,
            'form_metrics': {
                'shoulder_symmetry': 0,
                'back_angle': 180,
                'torso_angle': 180
            }
        }
        for key, value in defaults.items():
            if not hasattr(session_data, key):
                setattr(session_data, key, value)

    def _get_landmark(self, landmarks, landmark_type) -> Optional[Tuple[float, float]]:
        landmark = landmarks[landmark_type.value]
        if landmark.visibility > self.gesture_params['min_visibility']:
            return (landmark.x, landmark.y)
        return None

    def _calculate_angle(self, a: Tuple[float, float],
                         b: Tuple[float, float],
                         c: Tuple[float, float]) -> float:
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
            return 180.0
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _calculate_back_angle(self, shoulder: Tuple[float, float],
                              hip: Tuple[float, float]) -> float:
        vertical = np.array([0, -1])
        body_vector = np.array([shoulder[0] - hip[0], shoulder[1] - hip[1]])
        if np.linalg.norm(body_vector) == 0:
            return 180.0
        cosine = np.dot(vertical, body_vector) / (np.linalg.norm(vertical) * np.linalg.norm(body_vector))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _count_fingers(self, hand_landmarks, hand_label: str, img_shape) -> int:
        """
        Count raised fingers on a hand.
        hand_label: 'Left' or 'Right' hand label from mediapipe classification.
        """
        landmarks = hand_landmarks.landmark
        h, w = img_shape[:2]

        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [2, 6, 10, 14, 18]
        fingers = []

        # Thumb - different logic for left and right hand
        if hand_label == 'Right':
            if landmarks[finger_tips[0]].x < landmarks[finger_pips[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)

        # Other fingers - same logic for both hands
        for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
            if landmarks[tip].y < landmarks[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)
