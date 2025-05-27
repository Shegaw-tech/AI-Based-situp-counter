from flask import Flask, render_template, Response, jsonify, request
from flask import send_from_directory
from backend.pose_analyzer import PoseAnalyzer
from backend.health_recommender import HealthRecommender
from backend.models import SessionData
import cv2
import os
import numpy as np
import base64
import time
from threading import Lock
import json

pose_analyzer = PoseAnalyzer()
health_recommender = HealthRecommender()

# Get the base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(base_dir, 'frontend/templates'),
    static_folder=os.path.join(base_dir, 'frontend/static')
)
# Thread-safe session storage
sessions = {}
session_lock = Lock()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )
@app.route('/process_frame', methods=['POST'])
def process_frame():
    with session_lock:
        # Get session (or create new)
        session_id = request.headers.get('X-Session-ID', 'default')
        if session_id not in sessions:
            sessions[session_id] = SessionData()

        # Process frame
        frame_data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(frame_data)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Analyze pose
        results = pose_analyzer.process_frame(frame, sessions[session_id])

        # Prepare response - use .get() with defaults for optional fields
        response = {
            'count': sessions[session_id].count,
            'feedback': results.get('feedback', []),
            'completed_rep': results.get('completed_rep', False),
            'recommendations': [],
            'situp_type': results.get('current_type', 'crunch'),
            'type_changed': results.get('type_changed', False),
            'form_metrics': results.get('form_metrics', {}),
            'debug': {
                'landmarks_visible': results.get('debug', {}).get('landmarks_visible', False),
                'head_lift': results.get('debug', {}).get('head_lift'),
                'shoulder_lift': results.get('debug', {}).get('shoulder_lift'),
                'torso_angle': results.get('debug', {}).get('torso_angle'),
                'back_angle': results.get('debug', {}).get('back_angle'),
                'shoulder_symmetry': results.get('debug', {}).get('shoulder_symmetry'),
                'is_in_rep': sessions[session_id].is_in_rep,
                'rep_start': results.get('debug', {}).get('rep_start', False),
                'rep_end': results.get('debug', {}).get('rep_end', False),
                'gesture_detected': results.get('debug', {}).get('gesture_detected', False)
            }
        }

        # Add recommendations if rep completed
        if response['completed_rep']:
            response['recommendations'] = health_recommender.get_recommendations(
                sessions[session_id].count,
                sessions[session_id].form_metrics
            )

        # Encode annotated frame
        if 'annotated_frame' in results:
            _, buffer = cv2.imencode('.jpg', results['annotated_frame'])
            response['image'] = base64.b64encode(buffer).decode('utf-8')

        return jsonify(response)
@app.errorhandler(500)
def handle_server_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
