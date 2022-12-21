#LD_PRELOAD="/workspace/internal/TensorRT/build/out/libnvinfer_plugin.so" python3 manager.py

import threading
from queue import Queue
from io import BytesIO
import time
from flask import Flask, send_file, request
from internal.demodiffusion import DemoDiffusion, TRT_LOGGER
import tensorrt as trt
import time
import asyncio
import datetime

USE_PROMPTIST = False #<--- it sucks and slows everything by 2 seconds

# <---- PROMPTIST PIPELINE --->
# Info: https://huggingface.co/spaces/microsoft/Promptist
if USE_PROMPTIST:
    from internal.promptist import load_prompter
    from internal.promptist import generate as promptist_gen
    prompter_model, prompter_tokenizer = load_prompter()
# <---- PROMPTIST PIPELINE --->

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

app = Flask(__name__)

current_inferences = 0
task = asyncio.Event()

# Queue to hold incoming requests
request_queue = Queue()

# Thread to handle inferences
class InferenceThread(threading.Thread):
    def __init__(self, request_queue, pipeline_name):
        threading.Thread.__init__(self)
        self.request_queue = request_queue
        self.pipeline_name = pipeline_name
        
        # <---- INFERENCE PIPELINE --->
        print("Starting Pipeline")
        pipeline_start_time = time.perf_counter()

        # Register TensorRT plugins
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        #Configuration
        print("Starting Base")
        start_time = time.perf_counter()
        pipeline = DemoDiffusion(
            denoising_steps=30,
            denoising_fp16=True,
            output_dir="output",
            scheduler="LMSD", #Can be LMSD or DPM
            hf_token="", #Not necesary
            verbose=True,
            nvtx_profile=True,
            max_batch_size=16
        )
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000  # elapsed time in milliseconds
        print(f"Base Loaded in {elapsed_time:.2f} ms")

        print("Loading TensorRT Engines")
        start_time = time.perf_counter()
        pipeline.loadEngines(
            engine_dir="models",
            onnx_dir="models",
            onnx_opset=16,
            opt_batch_size=16,
            opt_image_height=512,
            opt_image_width=512,
            force_export=False,
            force_optimize=False,
            force_build=False,
            minimal_optimization=False,
            static_batch=False,
            static_shape=False,
            enable_preview=False,
        )
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000  # elapsed time in milliseconds
        print(f"TensorRT Engines Loaded in {elapsed_time:.2f} ms")

        print("Loading PyTorch Modules")
        start_time = time.perf_counter()
        pipeline.loadModules()
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000  # elapsed time in milliseconds
        print(f"PyTorch Modules Loaded in {elapsed_time:.2f} ms")

        pipeline_end_time = time.perf_counter()
        elapsed_time = (pipeline_end_time - pipeline_start_time) * 1000  # elapsed time in milliseconds
        print(f"Pipeline Loaded in {elapsed_time:.2f} ms")
        # <---- INFERENCE PIPELINE --->
        self.pipeline = pipeline

    def run(self):
        while True:
            # Get the next request from the queue
            request_data = self.request_queue.get()

            # Generate the prompt
            if USE_PROMPTIST:
                prompt = promptist_gen(prompter_model, prompter_tokenizer, request_data['prompt'])
            else:
                prompt = request_data['prompt']

            negprompt = request_data['negprompt']
            height = request_data['height']
            width = request_data['width']

            start_time = datetime.datetime.now()
            # Generate the image
            image = self.pipeline.infer([prompt], [negprompt], height, width)
            end_time = datetime.datetime.now()

            print("Prompt '", prompt, "' has been processed by Pipeline", self.pipeline_name, 
            ". Which started at ", start_time, "and ended at", end_time)
            # Return the image to the client
            request_data['response_queue'].put(image)

# Start the inference thread 1
inference_thread1 = InferenceThread(request_queue, "One")
inference_thread1.start()
# Start the inference thread 2
inference_thread2 = InferenceThread(request_queue, "Two")
inference_thread2.start()

@app.route('/inference/', methods=['POST', 'GET'])
async def handle_inference():
    # Load the JSON object from the request
    data = request.get_json()

    # Create a queue to hold the response
    response_queue = Queue()

    # Add the request to the queue
    request_queue.put({
        'prompt': data['prompt'],
        "negprompt": data['negprompt'],
        "height": data['height'],
        "width": data['width'],
        'response_queue': response_queue
    })

    # Wait for the response
    image = response_queue.get()

    # Serve the image
    return serve_pil_image(image[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

