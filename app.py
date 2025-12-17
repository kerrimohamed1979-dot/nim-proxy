from flask import Flask, request, jsonify, Response
import requests
import os
import json
import time

app = Flask(__name__)

# NVIDIA NIM API configuration
NIM_API_KEY = os.environ.get('NIM_API_KEY', '')
NIM_BASE_URL = os.environ.get('NIM_BASE_URL', 'https://integrate.api.nvidia.com/v1')

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        
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
        
        if stream:
            return handle_stream(nim_payload, headers)
        else:
            return handle_non_stream(nim_payload, headers)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def handle_non_stream(payload, headers):
    response = requests.post(
        f'{NIM_BASE_URL}/chat/completions',
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        return jsonify({'error': response.text}), response.status_code
    
    return jsonify(response.json())

def handle_stream(payload, headers):
    def generate():
        response = requests.post(
            f'{NIM_BASE_URL}/chat/completions',
            headers=headers,
            json=payload,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    yield decoded + '\n\n'
    
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
            }
        ]
    }
    return jsonify(models)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
