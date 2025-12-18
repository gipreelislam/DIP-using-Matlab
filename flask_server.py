from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import socket
import base64
import io
import os
from PIL import Image
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MATLAB_SERVER_HOST = 'localhost'
MATLAB_SERVER_PORT = 5050  # MATLAB server port
PYTHON_SERVER_PORT = 5051  # This Flask server port

# Working directory - set this to where MATLAB server is running
MATLAB_WORKING_DIR = os.getcwd()  # Change this if MATLAB is in a different directory

UPLOAD_FOLDER = os.path.join(MATLAB_WORKING_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(MATLAB_WORKING_DIR, 'outputs')

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def save_base64_image(base64_string, filename):
    """Save base64 image data to a file"""
    try:
        print(f"[DEBUG] Saving image as: {filename}")
        
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to image
        image_data = base64.b64decode(base64_string)
        
        # Save to file
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        print(f"[DEBUG] Image saved successfully: {filepath}")
        return filepath
    except Exception as e:
        print(f"[ERROR] Error saving image: {str(e)}")
        traceback.print_exc()
        raise Exception(f"Error saving image: {str(e)}")

def communicate_with_matlab(message):
    """Send message to MATLAB server and receive response"""
    sock = None
    try:
        print(f"[DEBUG] Connecting to MATLAB server at {MATLAB_SERVER_HOST}:{MATLAB_SERVER_PORT}")
        
        # Create socket connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)  # 30 second timeout
        
        # Connect to MATLAB server
        sock.connect((MATLAB_SERVER_HOST, MATLAB_SERVER_PORT))
        print(f"[DEBUG] Connected to MATLAB server")
        
        # Send message with newline terminator
        message_with_newline = message + '\n'
        print(f"[DEBUG] Sending to MATLAB: {message}")
        print(f"[DEBUG] Message length: {len(message)} chars")
        sock.sendall(message_with_newline.encode('utf-8'))
        print(f"[DEBUG] Message sent successfully")
        
        # Receive response
        print(f"[DEBUG] Waiting for MATLAB response...")
        response_parts = []
        while True:
            try:
                part = sock.recv(4096).decode('utf-8')
                if not part:
                    break
                response_parts.append(part)
                # If response ends with newline, we have complete message
                if '\n' in part:
                    break
            except socket.timeout:
                print(f"[DEBUG] Socket timeout while receiving")
                break
        
        response = ''.join(response_parts).strip()
        print(f"[DEBUG] MATLAB Response: {response}")
        
        return response
        
    except socket.timeout:
        print("[ERROR] Connection to MATLAB server timed out")
        raise Exception("Connection to MATLAB server timed out")
    except ConnectionRefusedError:
        print("[ERROR] Could not connect to MATLAB server")
        raise Exception("Could not connect to MATLAB server. Is it running on port 5050?")
    except Exception as e:
        print(f"[ERROR] MATLAB communication error: {str(e)}")
        traceback.print_exc()
        raise Exception(f"MATLAB communication error: {str(e)}")
    finally:
        if sock:
            sock.close()
            print("[DEBUG] Socket closed")

@app.route('/filter', methods=['POST', 'OPTIONS'])
def apply_filter():
    """Handle filter application requests"""
    
    # Handle OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        print("\n[DEBUG] ========== NEW FILTER REQUEST ==========")
        
        # Get request body
        body = request.get_data(as_text=True)
        print(f"[DEBUG] Request body length: {len(body)} chars")
        print(f"[DEBUG] First 200 chars: {body[:200]}")
        
        if not body:
            print("[ERROR] No data provided")
            return jsonify({'error': 'No data provided'}), 400
        
        # Parse the pipe-separated data: image_data|filter|option
        parts = body.split('|')
        print(f"[DEBUG] Request parts: {len(parts)} parts")
        print(f"[DEBUG] Parts breakdown:")
        for i, part in enumerate(parts):
            if i == 0:
                print(f"  Part {i}: [IMAGE DATA] (length: {len(part)} chars)")
            else:
                print(f"  Part {i}: '{part}'")
        
        if len(parts) < 2:
            print("[ERROR] Invalid request format")
            return jsonify({'error': 'Invalid request format. Expected: image_data|filter|option'}), 400
        
        base64_image = parts[0]
        filter_parts = parts[1:]  # Everything after the image is the filter command
        print(f"[DEBUG] Filter command: {filter_parts}")
        
        # Generate unique filename
        import time
        timestamp = int(time.time() * 1000)
        input_filename = f'input_{timestamp}.png'
        output_filename = f'output_{timestamp}.png'
        
        # Save the uploaded image
        input_path = save_base64_image(base64_image, input_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Convert to absolute paths for MATLAB
        input_path_abs = os.path.abspath(input_path)
        output_path_abs = os.path.abspath(output_path)
        
        print(f"[DEBUG] Input absolute path: {input_path_abs}")
        print(f"[DEBUG] Output absolute path: {output_path_abs}")
        
        # Build MATLAB command with absolute path
        filter_command = '|'.join(filter_parts)
        matlab_message = f"{input_path_abs}|{filter_command}"
        
        print(f"[DEBUG] Full MATLAB message: {matlab_message}")
        print(f"[DEBUG] Message length: {len(matlab_message)} chars")
        print(f"[DEBUG] Number of pipe separators: {matlab_message.count('|')}")
        
        # Communicate with MATLAB server
        response = communicate_with_matlab(matlab_message)
        
        # The response should be the path to the output image
        output_file_path = response.strip()
        print(f"[DEBUG] Looking for output file: {output_file_path}")
        
        # Check if output file exists
        if not os.path.exists(output_file_path):
            print(f"[ERROR] Output file not found: {output_file_path}")
            return jsonify({'error': f'Output file not found: {output_file_path}'}), 500
        
        print(f"[DEBUG] Output file found, sending to client")
        
        # Verify the file is a valid image
        try:
            with Image.open(output_file_path) as img:
                print(f"[DEBUG] Image verified: {img.format} {img.size}")
        except Exception as img_err:
            print(f"[ERROR] Invalid image file: {img_err}")
            return jsonify({'error': f'Invalid image file returned from MATLAB: {img_err}'}), 500
        
        # Send the filtered image back
        print(f"[DEBUG] Sending file to client...")
        return send_file(
            output_file_path,
            mimetype='image/png',
            as_attachment=False,
            download_name='filtered.png'
        )
        
    except Exception as e:
        print(f"[ERROR] Exception in apply_filter: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Try to connect to MATLAB server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((MATLAB_SERVER_HOST, MATLAB_SERVER_PORT))
        sock.close()
        
        matlab_status = 'connected' if result == 0 else 'disconnected'
        
        return jsonify({
            'status': 'ok',
            'matlab_server': matlab_status,
            'host': MATLAB_SERVER_HOST,
            'port': MATLAB_SERVER_PORT,
            'upload_folder': UPLOAD_FOLDER,
            'output_folder': OUTPUT_FOLDER
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'matlab_server': 'error',
            'error': str(e)
        })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Image Filter Server',
        'endpoints': {
            '/filter': 'POST - Apply image filter',
            '/health': 'GET - Check server health'
        },
        'matlab_server': f'{MATLAB_SERVER_HOST}:{MATLAB_SERVER_PORT}'
    })

if __name__ == '__main__':
    print("="*60)
    print(f"  Image Filter Server Starting")
    print("="*60)
    print(f"  MATLAB Server: {MATLAB_SERVER_HOST}:{MATLAB_SERVER_PORT}")
    print(f"  Python Server: http://localhost:{PYTHON_SERVER_PORT}")
    print(f"  MATLAB Working Dir: {MATLAB_WORKING_DIR}")
    print(f"  Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"  Output folder: {os.path.abspath(OUTPUT_FOLDER)}")
    print("="*60)
    print(f"\n⚠️  IMPORTANT: Make sure MATLAB server is running from:")
    print(f"   {MATLAB_WORKING_DIR}")
    print(f"\n✓ Server running on http://localhost:{PYTHON_SERVER_PORT}")
    print(f"✓ Web app should connect to: http://localhost:{PYTHON_SERVER_PORT}/filter")
    print(f"\nTest health: http://localhost:{PYTHON_SERVER_PORT}/health\n")
    
    app.run(host='0.0.0.0', port=PYTHON_SERVER_PORT, debug=True)