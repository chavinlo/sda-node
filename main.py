from flask import Flask, request, jsonify, send_file, Response
from queue import Queue
from threading import Lock, Thread, local
import json
import os
from threads.t2i import image_generator
from io import BytesIO
import base64
from PIL import Image

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

CONFIG_PATH = "/workspace/backend/nodes/v2/cfg/basic.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Create the request queue and image queue
request_queue = Queue()
image_queue = Queue()
queue_lock = Lock()

thread_local = local()

API_KEY = "password123"

# @app.before_request
# def check_api_key():
#     if not request.headers.get('APIKEY') or request.headers.get('APIKEY') != API_KEY:
#         return jsonify({"error": "Invalid API key"}), 401

@app.route("/q", methods=["GET"])
def queues():
    request_length = request_queue.qsize()
    return str(request_length)
    
@app.route("/t2i", methods=["POST"])
def txt2img():
    json = request.get_json(force=True)
    print("front req", json)
    with queue_lock:
        r = {
            "prompt": str(json['prompt']),
            "negprompt": str(json['negprompt']),
            "width": int(json['width']),
            "height": int(json['height']),
            "steps": int(json['steps']),
            "cfg": float(json['cfg']),
            "seed": int(json['seed']),
            "scheduler": str(json['scheduler']),
            "lpw": bool(json['lpw']),
            "mode": str(json['mode']) # <- "I expect a response that is X"
        }
        r['type'] = 'txt2img'
        request_queue.put(r)
    response = image_queue.get()
    if response['status'] == 'fail':
        response = jsonify(response)
        response.status_code = 400
        return response

    if json['mode'] == 'file':
        file = response['content']
        return serve_pil_image(file)
    elif json['mode'] == 'json':
        jsonify(response)

    return 

# Image Generator Engine
image_generator_thread = Thread(
    target=image_generator, 
    args=(
        config, 
        request_queue,
        image_queue,
        )
    )
image_generator_thread.start()