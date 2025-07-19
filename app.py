import os
import threading

from flask import Flask, request, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from src import global_vars
from src import chatbot
from src import qdrant_db as db

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO
script_dir = os.path.dirname(os.path.abspath(__file__))
# Session storage for user data
def init_data():
    if global_vars.database is None or global_vars.embedding_fn is None:
        global_vars.database, global_vars.embedding_fn = db.init_db(
            global_vars.AZURE_OPENAI_ENDPOINT,
            global_vars.AZURE_OPENAI_API_EMBEDDED_KEY,
            global_vars.AZURE_OPENAI_EMBEDDING_MODEL,
            global_vars.QDRANT_API_KEY,
            global_vars.QDRANT_HOST,
        )
    db.init_data(global_vars.database, global_vars.embedding_fn, script_dir)

def emit_res(resp):
    emit('message', {'response': resp})


# Route to serve the main page
@app.route('/')
def index():
    th = threading.Thread(target=init_data)
    th.start()
    chatbot.set_print_fn(emit_res)
    chatbot.set_debug(True)
    return render_template('index.html')

# API endpoint to get contextual suggestions based on user question
@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return {'error': 'Question is required'}, 400
        
        suggestions = chatbot.get_suggestion(question)
        
        return {
            'suggestions': suggestions,
            'count': len(suggestions)
        }
        
    except Exception as e:
        return {'error': str(e)}, 500

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    chatbot.set_user_id(session_id)
    chatbot.greeting()

@socketio.on('chat_message')
def handle_chat_message(data):
    question = data.get('message', '')
    if question.lower() == "exit" or question.lower() == "quit":
        chatbot.goodbye()
    else:
        response = chatbot.handle_question(question)
        
        # Send the main response
        emit('message', {'response': response})
        
        # Get and send suggestions separately for dynamic UI updates
        try:
            suggestions = chatbot.get_suggestion(question)
            # Only send suggestions if we have data from Qdrant
            if suggestions and len(suggestions) > 0:
                emit('suggestions', {
                    'suggestions': suggestions,
                    'count': len(suggestions),
                    'based_on': question
                })
            else:
                # Send empty suggestions to hide the suggestions container
                emit('suggestions', {
                    'suggestions': [],
                    'count': 0,
                    'based_on': question,
                    'hide': True
                })
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            # Don't send any suggestions on error
            emit('suggestions', {
                'suggestions': [],
                'count': 0,
                'based_on': question,
                'hide': True
            })

@socketio.on('clear_conversation')
def handle_clear_conversation():
    """Handle clearing the conversation memory"""
    try:
        chatbot.clear_memory()
        emit('conversation_cleared', {'status': 'success'})
    except Exception as e:
        emit('conversation_cleared', {'status': 'error', 'message': str(e)})

@socketio.on('get_suggestions')
def handle_get_suggestions(data):
    """Socket.IO handler for getting suggestions"""
    try:
        question = data.get('question', '')
        suggestions = chatbot.get_suggestion(question)
        
        # Only send suggestions if we have data from Qdrant
        if suggestions and len(suggestions) > 0:
            emit('suggestions', {
                'suggestions': suggestions,
                'count': len(suggestions)
            })
        else:
            # Send empty suggestions to hide the suggestions container
            emit('suggestions', {
                'suggestions': [],
                'count': 0,
                'hide': True
            })
        
    except Exception as e:
        # Don't send any suggestions on error
        emit('suggestions', {
            'suggestions': [],
            'count': 0,
            'error': str(e),
            'hide': True
        })

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
