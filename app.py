from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import os
import json
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# NVIDIA NIM API configuration
NIM_API_KEY = os.environ.get('NIM_API_KEY', '')
NIM_BASE_URL = os.environ.get('NIM_BASE_URL', 'https://integrate.api.nvidia.com/v1')

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        print(f"Received request: {json.dumps(data, indent=2)}")
        
        # Extract OpenAI format parameters
        messages = data.get('messages', [])
        model = data.get('model', 'meta/llama-3.1-405b-instruct')
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 1024)
        stream = data.get('stream', False)
        
        # Prepare NVIDIA NIM request
        nim_payload = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': stream
        }
        
        headers = {
            'Authorization': f'Bearer {NIM_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        print(f"Sending to NIM: {NIM_BASE_URL}/chat/completions")
        
        if stream:
            return handle_stream(nim_payload, headers)
        else:
            return handle_non_stream(nim_payload, headers)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        error_response = {
            'error': {
                'message': str(e),
                'type': 'proxy_error',
                'code': 'internal_error'
            }
        }
        return jsonify(error_response), 500

def handle_non_stream(payload, headers):
    try:
        response = requests.post(
            f'{NIM_BASE_URL}/chat/completions',
            headers=headers,
            json=payload,
            timeout=120
        )
        
        print(f"NIM response status: {response.status_code}")
        print(f"NIM response: {response.text[:500]}")
        
        if response.status_code != 200:
            error_response = {
                'error': {
                    'message': f'NVIDIA NIM API error: {response.text}',
                    'type': 'nim_api_error',
                    'code': str(response.status_code)
                }
            }
            return jsonify(error_response), response.status_code
        
        nim_response = response.json()
        
        # Ensure response is in OpenAI format
        if 'choices' in nim_response:
            return jsonify(nim_response)
        else:
            # Convert to OpenAI format if needed
            openai_response = {
                'id': nim_response.get('id', f'chatcmpl-{int(time.time())}'),
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': payload['model'],
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': nim_response.get('content', '')
                    },
                    'finish_reason': 'stop'
                }],
                'usage': nim_response.get('usage', {})
            }
            return jsonify(openai_response)
            
    except requests.exceptions.Timeout:
        error_response = {
            'error': {
                'message': 'Request to NVIDIA NIM timed out',
                'type': 'timeout_error',
                'code': 'timeout'
            }
        }
        return jsonify(error_response), 504
    except Exception as e:
        print(f"Request error: {str(e)}")
        error_response = {
            'error': {
                'message': str(e),
                'type': 'request_error',
                'code': 'internal_error'
            }
        }
        return jsonify(error_response), 500

def handle_stream(payload, headers):
    def generate():
        try:
            response = requests.post(
                f'{NIM_BASE_URL}/chat/completions',
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            )
            
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        yield decoded + '\n\n'
        except Exception as e:
            print(f"Stream error: {str(e)}")
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models"""
    models = {
        'object': 'list',
        'data': [
            {
                'id': 'meta/llama-3.1-405b-instruct',
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'nvidia'
            },
            {
                'id': 'meta/llama-3.1-70b-instruct',
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'nvidia'
            },
            {
                'id': 'meta/llama-3.1-8b-instruct',
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'nvidia'
            }
        ]
    }
    return jsonify(models)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'nim_configured': bool(NIM_API_KEY),
        'nim_base_url': NIM_BASE_URL
    })

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'OpenAI-compatible NVIDIA NIM Proxy',
        'endpoints': {
            'chat': '/v1/chat/completions',
            'models': '/v1/models',
            'health': '/health'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
