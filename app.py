# RUN WITH THIS COMMAND LOCALLY
# RUN WITH THIS COMMAND LOCALLY
# RUN WITH THIS COMMAND LOCALLY

# uvicorn app:app --host 0.0.0.0 --port 8000
# uvicorn app:app --host 0.0.0.0 --port 8000
# uvicorn app:app --host 0.0.0.0 --port 8000
# uvicorn app:app --host 0.0.0.0 --port 8000
# uvicorn app:app --host 0.0.0.0 --port 8000
# uvicorn app:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image
import traceback

# FORCE TO USE CPU
tf.config.set_visible_devices([], 'GPU')


# Initialize the FastAPI app
app = FastAPI()

# Load your custom Keras model once during app startup (global model)
model_path = "./model/melon-disease-v3-224.keras"  # Path to your saved model
model = load_model(model_path)

# List of class labels for your model (replace with your actual class labels)
CLASS_LABELS = ["Anthracnose", "Downy Mildew", "Healthy", "Mosaic Virus"]

# Clear GPU memory after each request
def clear_gpu_memory():
    tf.keras.backend.clear_session()

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName or self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.layerName).output, self.model.output])
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return output


@app.post("/generate-heatmap/")
async def generate_heatmap(file: UploadFile = File(...), enable_gradcam: bool = True, background_tasks: BackgroundTasks = None):
    try:
        # Read the uploaded image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        orig = np.array(pil_image)

        # Preprocess the image for the model
        image = pil_image.resize((224, 224)) 
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make predictions
        preds = model.predict(image)
        classIdx = np.argmax(preds[0])
        label = CLASS_LABELS[classIdx]
        prob = float(preds[0][classIdx])

        # Initialize GradCAM only if enabled
        if enable_gradcam:
            cam = GradCAM(model, classIdx)
            print(f"Using layer: {cam.layerName}")  # Debug print
            heatmap = cam.compute_heatmap(image)

            # Resize heatmap to original image size and overlay
            heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
            output = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
        else:
            output = orig  # Just return the original image if GradCAM is disabled

        # Convert the output image to bytes
        _, output_image = cv2.imencode(".jpg", output)
        image_bytes = io.BytesIO(output_image.tobytes())

        # Clear GPU memory after the request
        if background_tasks:
            background_tasks.add_task(clear_gpu_memory)

        # Return the heatmap image and prediction details
        return StreamingResponse(
            image_bytes,
            media_type="image/jpeg",
            headers={"Prediction-Label": label, "Prediction-Confidence": str(prob)}
        )

    except Exception as e:
        # Log the exception with traceback
        error_traceback = traceback.format_exc()
        print(model.summary())
        print(error_traceback)  # Logs to the console

        # Return the traceback in the response (useful for debugging)
        return JSONResponse(
            status_code=500, 
            content={"message": str(e), "traceback": error_traceback}
        )
