# app.py
import os
import base64
import io
import requests # To call the FastAPI server
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# Configuration
FASTAPI_GENERATE_ENDPOINT = "http://localhost:8000/generate" # Ensure this is your FastAPI server URL
FASTAPI_FEEDBACK_ENDPOINT = "http://localhost:8000/feedback" # Your FastAPI feedback endpoint

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/chat_api', methods=['POST'])
def chat_api():
    """
    Receives image (as base64) and message from the client-side JavaScript,
    calls the FastAPI backend for generation, and returns the FastAPI's response.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        base64_image_str = data.get('image_base64')
        user_message = data.get('message')
        
        if not base64_image_str:
            return jsonify({"error": "No image data provided"}), 400
        if user_message is None:
            return jsonify({"error": "No message provided"}), 400

        fastapi_payload = {
            "image": base64_image_str,
            "message": user_message,
            "parameters": {
                "num_beams": data.get('num_beams', 1),
                "temperature": data.get('temperature', 1.0),
                "max_new_tokens": 300,
                "max_length": 2000
            }
        }

        print(f"Forwarding generation request to FastAPI: {FASTAPI_GENERATE_ENDPOINT}")
        response = requests.post(FASTAPI_GENERATE_ENDPOINT, json=fastapi_payload, timeout=300)
        response.raise_for_status()
        
        fastapi_response_json = response.json()
        bot_response = fastapi_response_json.get("response", "Error: 'response' key missing in FastAPI JSON.")
        
        # Include original_response for feedback purposes if needed directly here,
        # but frontend will primarily handle it.
        return jsonify({"bot_response": bot_response, "original_user_message": user_message})

    except requests.exceptions.Timeout:
        print("Error: FastAPI generation request timed out.")
        return jsonify({"error": "API request timed out"}), 504
    except requests.exceptions.RequestException as e:
        error_detail = str(e)
        status_code = 500
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try: error_detail = e.response.json().get("detail", e.response.text)
            except ValueError: error_detail = e.response.text
        print(f"Error calling FastAPI for generation: {error_detail}")
        return jsonify({"error": f"FastAPI Error: {error_detail}"}), status_code
    except Exception as e:
        print(f"An unexpected error occurred in /chat_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route('/feedback_api', methods=['POST'])
def feedback_api():
    """
    Receives feedback from the client-side JavaScript and forwards it
    to the FastAPI /feedback endpoint.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received for feedback"}), 400

        feedback_type = data.get('feedback_type')
        original_response = data.get('original_response')
        modified_response = data.get('modified_response') # This will be null for "accept"

        if not feedback_type or not original_response:
            return jsonify({"error": "Missing feedback_type or original_response"}), 400
        
        if feedback_type not in ["accept", "modify", "reject"]:
            return jsonify({"error": "Invalid feedback_type"}), 400

        fastapi_payload = {
            "feedback_type": feedback_type,
            "original_response": original_response,
        }
        if modified_response is not None: # For "modify" and "reject" (where user provides new text)
            fastapi_payload["modified_response"] = modified_response
        
        print(f"Forwarding feedback to FastAPI: {FASTAPI_FEEDBACK_ENDPOINT}, Payload: {fastapi_payload}")
        response = requests.post(FASTAPI_FEEDBACK_ENDPOINT, json=fastapi_payload, timeout=60)
        response.raise_for_status() # Raise an exception for HTTP errors

        # Log success or specific message from feedback API if any
        print(f"Feedback successfully sent to FastAPI. Response: {response.json()}")
        return jsonify({"status": "feedback_received", "detail": response.json()}), 200

    except requests.exceptions.Timeout:
        print("Error: FastAPI feedback request timed out.")
        return jsonify({"error": "Feedback API request timed out"}), 504
    except requests.exceptions.RequestException as e:
        error_detail = str(e)
        status_code = 500
        if hasattr(e, 'response') and e.response is not None:
            status_code = e.response.status_code
            try: error_detail = e.response.json().get("detail", e.response.text)
            except ValueError: error_detail = e.response.text
        print(f"Error calling FastAPI for feedback: {error_detail}")
        return jsonify({"error": f"FastAPI Feedback Error: {error_detail}"}), status_code
    except Exception as e:
        print(f"An unexpected error occurred in /feedback_api: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred during feedback: {str(e)}"}), 500


if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write("<h1>Flask Chatbot Placeholder</h1><p>index.html not found, created a basic one. Please replace it.</p>")
        print("templates/index.html was not found. A placeholder file has been created.")
    app.run(debug=True, port=5001, host='0.0.0.0')
