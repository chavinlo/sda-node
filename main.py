from flask import Flask, request, jsonify, send_file
from queue import Queue
from threading import Lock, Thread
import json
import os
from threads.t2i import image_generator
from io import BytesIO

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

CONFIG_PATH = "cfg/basic.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

app = Flask(__name__, static_folder='demo')
app.config['SECRET_KEY'] = 'secret!'

# Create the request queue and image queue
request_queue = Queue()
image_queue = Queue()
queue_lock = Lock()

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
        return jsonify(response)

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