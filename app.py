import eventlet 
eventlet.monkey_patch() 
import base64
import os
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room # Import SocketIO components
from models.rag_system import OptimizedRAGSystem
from models.face_auth import FaceAuthTransformer
from config import Config
import os
from dotenv import load_dotenv
import sqlite3
import base64
import numpy as np
import cv2
from models.send_img import get_info
import logging
from utils import get_purchase_history
from models.extract_info import LLMExtract
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_very_secret_key_for_dev_only")
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

config = Config()
rag_system = OptimizedRAGSystem(config)
client_auth_transformers = {} 

logging.basicConfig(level=logging.DEBUG)

def decode_image_from_base64(base64_string):
    """Decodes a base64 image string (data URL) into an OpenCV image."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# --- Routes ---
@app.route('/')
def index():
    """Serves the main page. Shows auth page if not logged in, chat page otherwise."""
    
    if session.get('anonymous', False):

        return render_template('chat.html', user_info=None, purchase_history=[])
    elif session.get('authenticated', False) and 'user_info' in session:

        user_info = session.get('user_info')
        purchase_history = []
        if user_info and 'id' in user_info:
            purchase_history = get_purchase_history(user_info['id'])
        return render_template('chat.html', user_info=user_info, purchase_history=purchase_history)
    else:
        # Neither authenticated nor anonymous -> choice page
        # Clear potentially inconsistent state just in case
        session.pop('authenticated', None)
        session.pop('user_info', None)
        session.pop('anonymous', None)
        return render_template('choice.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles incoming chat messages from authenticated or anonymous users."""

    # Determine user_key and user_info STRICTLY based on session flags
    user_key = "anonymous" # Default to anonymous
    scoped_user_info = None # Default user_info for this request scope

    if session.get('authenticated', False):
        # Try to get authenticated user info ONLY if authenticated flag is true
        auth_user_info = session.get('user_info')
        if auth_user_info and 'id' in auth_user_info:
            scoped_user_info = auth_user_info # Set user_info for this scope
            user_key = str(scoped_user_info['id'])
        else:
            # Authenticated flag is true, but no user_info? Session corruption? Fallback.
            print("WARNING: Authenticated session but no user_info found. Treating as anonymous.")
            session.pop('authenticated', None) # Clean up corrupted state
            session['anonymous'] = True # Force anonymous
            user_key = "anonymous"
    elif session.get('anonymous', False):
        # Explicitly anonymous session
         user_key = "anonymous"
         scoped_user_info = None # Ensure user_info is None for this scope
    else:
        # This case should not be reached if routing from '/' is correct
        print("ERROR: /chat accessed without valid authenticated or anonymous session state.")
        return jsonify({"error": "Invalid session state."}), 403
    

    data = request.get_json()
    user_query = data.get('prompt')
    if not user_query:
        return jsonify({"error": "No prompt provided"}), 400

    user_query

    try:
        response = rag_system.answer_query(user_key, user_query)
        return jsonify({"role": "assistant", "content": response})
    except Exception as e:
        print(f"Error getting RAG response: {e}")
        # Add error to history for the correct user_key
        error_msg = "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn."
        try:
            rag_system.chat_history.add_chat(user_key, user_query, f"ERROR: {e}")
        except Exception as hist_e:
            print(f"Failed to add error to chat history: {hist_e}")
        return jsonify({"error": "Failed to get response from assistant"}), 500

@app.route('/logout')
def logout():
    """Logs the user out by clearing relevant session keys."""
    # user_id_to_clear = session.get('user_info', {}).get('id') # Get ID before popping if history clearing on logout is needed
    
    session.pop('authenticated', None)
    session.pop('user_info', None)
    session.pop('anonymous', None) 
  
    print("Logout completed.")
    return redirect(url_for('index'))

# --- New Routes for Pre-Auth Flow ---

@app.route('/authenticate')
def authenticate():
    """Serves the face authentication page."""
    # Clear anonymous flag if user chooses to authenticate from choice page
    session.pop('anonymous', None)
    return render_template('auth.html')

@app.route('/start_anonymous_chat')
def start_anonymous_chat():
    """Sets flag for anonymous chat and redirects to index."""
    # Ensure other potentially conflicting keys are removed FIRST
    session.pop('authenticated', None)
    session.pop('user_info', None)
    session.pop('anonymous', None) # Pop it first to ensure clean state before setting
    
    # Now, explicitly set anonymous mode
    session['anonymous'] = True
    
    # Clear history specifically for the anonymous user key
    rag_system.clear_chat_history("anonymous") 
    print("Starting anonymous chat session.")
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles displaying and processing the registration form (simulation)."""
    if request.method == 'POST':
        # Simulate processing registration data
        name = request.form.get('name')
        user_id = request.form.get('user_id')
        print(f"Simulating registration for: Name={name}, ID={user_id}")
        return redirect(url_for('index')) # Or redirect to authenticate?
    # For GET request, show the registration form
    return render_template('register.html') # New template needed

# --- WebSocket Events ---
@socketio.on('connect')
def handle_connect():
    """Handles new client connections for authentication."""
    sid = request.sid
    print(f'Client connected for auth: {sid}')
    # Create a unique FaceAuthTransformer instance for this session
    client_auth_transformers[sid] = FaceAuthTransformer()
    join_room(sid) # Use session ID as a room for targeted messages
    print(f"Auth transformer created for SID: {sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Cleans up when a client disconnects."""
    sid = request.sid
    print(f'Client disconnected: {sid}')
    # Remove the transformer instance for this client
    if sid in client_auth_transformers:
        del client_auth_transformers[sid]
        print(f"Auth transformer removed for SID: {sid}")
    leave_room(sid)

@socketio.on('video_frame')
def handle_video_frame(data):
    """Receives and processes video frames for face authentication."""
    sid = request.sid

    if sid not in client_auth_transformers:
        print(f"Error: No auth transformer found for SID {sid}. Client might need to reconnect.")
        return

    image_data_url = data.get('image')
    if not image_data_url:
        print(f"Error: No image data received from {sid}")
        return

    frame = decode_image_from_base64(image_data_url)
    if frame is None:
        print(f"Error decoding image from {sid}")
        return

    transformer = client_auth_transformers[sid]

    try:
        result = transformer.recognize_face(frame) 
        match_outcome = result.get('match')
        bbox = result.get('bbox')
        confidence = result.get('confidence')

        if bbox:
            bbox = [int(coord) for coord in bbox]

        # Prepare the payload to send back
        emit_data = {'success': False, 'message': None, 'user_info': None, 'bbox': bbox, 'confidence': confidence}

        if isinstance(match_outcome, dict) and 'id' in match_outcome: # Successful match
            print(f"Authentication successful for SID: {sid}. User: {match_outcome}")
            emit_data['success'] = True
            emit_data['user_info'] = match_outcome
            emit('auth_result', emit_data, room=sid)
        elif match_outcome is False: # Explicit failure (e.g., unknown face)
            print(f"Authentication explicitly failed for SID: {sid}")
            emit_data['success'] = False
            emit_data['message'] = 'Không nhận dạng được khuôn mặt.'
            emit('auth_result', emit_data, room=sid)
        else: # No face detected or processing issue
            emit_data['success'] = False
            emit_data['message'] = 'Đang xử lý...'
            # Send the status update anyway (includes bbox/conf if available)   
            emit('auth_result', emit_data, room=sid)

    except Exception as e:
        print(f"Error during face recognition processing for SID {sid}: {e}")
        # Notify client of the server error
        emit('auth_result', {'success': False, 'message': f'Lỗi máy chủ: {e}', 'bbox': None, 'confidence': None}, room=sid)

@app.route('/confirm_auth', methods=['POST'])
def confirm_auth():
    """Handles storing session data after successful WebSocket authentication."""
    data = request.get_json()
    user_info = data.get('user_info')

    if not user_info or 'id' not in user_info or 'name' not in user_info:
        return jsonify({'status': 'error', 'message': 'Invalid user info provided'}), 400

    # Store auth state and user info in the session
    session['authenticated'] = True
    session['user_info'] = user_info
    print(f"Session confirmed for user: {user_info.get('name')}")
    return jsonify({'status': 'ok'})

@app.route('/process_image', methods=['POST'])
def process_image():
    file_path = None
    ocr_response = None

    user_key = "anonymous" 
    user_info = None 
    purchase_history = [] 

    # Determine user info based on session
    if session.get('authenticated', False):
        auth_user_info = session.get('user_info')
        if auth_user_info and 'id' in auth_user_info:
            user_info = auth_user_info
            user_key = str(user_info['id'])
            purchase_history = get_purchase_history(user_info['id']) 

    try:
        recent_history = rag_system.chat_history.get_recent_history(user_key)
    except Exception as hist_err:
        print(f"Error fetching chat history for {user_key}: {hist_err}")
        recent_history = "" 

    try:
        # Expect only multipart/form-data now
        if not request.content_type.startswith('multipart/form-data'):
            print(f"Unsupported Content-Type for image upload: {request.content_type}")
            return jsonify({'error': 'Invalid request format for image upload'}), 415
            
        if 'image' not in request.files:
            return jsonify({'error': 'No image file part found'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            save_path = os.path.join('.', file.filename) 
            try:
                file.save(save_path)
                file_path = save_path
            except Exception as save_err:
                print(f"Error saving uploaded file: {save_err}")
                return jsonify({'error': 'Could not save uploaded file'}), 500
        else:
             # This case might be redundant but kept for robustness
            return jsonify({'error': 'Invalid file provided'}), 400

        if not file_path:
             print("Error: file_path is None.")
             return jsonify({'error': 'Internal error determining file path'}), 500

        encoded_image = LLMExtract.image_to_base64(file_path)

        extracted_info = LLMExtract.llm_extract(encoded_image=encoded_image) 
        if not extracted_info:
            return jsonify({'error': 'Failed to extract information from image'}), 500

        # --- Format Extracted Information into a Query ---
        search_query = f"Sự kết hợp từ các thành phần như {extracted_info.ingredients}, " \
               f"tạo nên một đồ uống có màu {extracted_info.drink_color}, " \
               f"thường được phục vụ trong {extracted_info.container_type}."

        # Thêm phần topping nếu có
        if extracted_info.topping != 'None':
            search_query += f" Trên bề mặt được phủ {extracted_info.topping}"

        search_query += f". Lý tưởng cho {extracted_info.suitable_for}"

        try:
            search_response = rag_system._answer_with_vector(user_key, search_query, user_info, purchase_history, is_image_upload=True) 

            if user_key != "anonymous":
                try:
                    rag_system.chat_history.add_chat(user_key, search_query, search_response)
                    print(f"Saved image search history for user: {user_key}") # Optional print
                except Exception as chat_save_err:
                    print(f"Error saving image search chat history for user {user_key}: {chat_save_err}")
   
        except Exception as search_err: 
            print(f"Exception during vector search: {search_err}")
            if os.path.exists(file_path):
                try: os.remove(file_path)
                except Exception as rm_err: print(f"Error cleaning up file {file_path} after search error: {rm_err}")
            return jsonify({'error': 'Exception during information retrieval'}), 500
        
        # --- Cleanup successful upload --- 
        if os.path.exists(file_path):
            try:    
                os.remove(file_path)
            except Exception as rm_err:
                print(f"Error cleaning up file {file_path} after success: {rm_err}")

        return jsonify({'content': search_response})

    except Exception as e:
        print(f"An unexpected error occurred in /process_image: {e}")
        if file_path and os.path.exists(file_path): # Check if file_path exists before trying to remove
             try: os.remove(file_path)
             except Exception as rm_err: print(f"Error cleaning up file {file_path} after unexpected error: {rm_err}")
        return jsonify({'error': 'An unexpected server error occurred'}), 500

if __name__ == '__main__':
    # Ensure the templates and static folders exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')

    print("Starting Flask-SocketIO server with separate chat histories...")
    # Use eventlet for async mode, disable reloader for stability with eventlet/gevent
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False) 