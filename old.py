from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image
from typing import Tuple, Optional
from functools import lru_cache

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

app = FastAPI()

# Constants
MODEL_PATH = "./model/npk-classifier-v2-functional.keras"
IMAGE_SIZE = (224, 224)
CLASS_LABELS = ["Healthy", "Nitrogen Deficient", "Phosphorus Deficient", "Potassium Deficient"]

# Cache model loading
@lru_cache(maxsize=1)
def get_model():
    return load_model(MODEL_PATH)

class GradCAM:
    def __init__(self, model: tf.keras.Model, class_idx: int):
        self.model = model
        self.class_idx = class_idx
        self.layer_name = self._find_target_layer()
        # Cache the gradient model
        self.grad_model = tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )

    def _find_target_layer(self) -> str:
        for layer in reversed(self.model.layers):
            if len(layer.output.shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    @tf.function
    def _compute_gradients(self, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(images)
            loss = predictions[:, self.class_idx]
        grads = tape.gradient(loss, conv_outputs)
        return conv_outputs, grads

    def compute_heatmap(self, image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        # Convert image to tensor
        image_tensor = tf.convert_to_tensor(image)
        
        # Compute gradients using the cached gradient model
        conv_outputs, grads = self._compute_gradients(image_tensor)
        
        # Calculate guided gradients
        guided_grads = tf.cast(conv_outputs > 0, "float32") * \
                      tf.cast(grads > 0, "float32") * grads

        # Calculate weights and CAM
        weights = tf.reduce_mean(guided_grads[0], axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

        # Resize and normalize heatmap
        heatmap = cv2.resize(cam.numpy(), (image.shape[2], image.shape[1]))
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = (numer / denom * 255).astype("uint8")
        
        return heatmap

    @staticmethod
    def overlay_heatmap(heatmap: np.ndarray, image: np.ndarray, 
                       alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        colored_heatmap = cv2.applyColorMap(heatmap, colormap)
        return cv2.addWeighted(image, alpha, colored_heatmap, 1 - alpha, 0)

async def process_image(contents: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Process image bytes into model input format and original image"""
    image = Image.open(io.BytesIO(contents))
    orig = np.array(image)
    
    # Resize and preprocess image for model
    processed = np.array(image.resize(IMAGE_SIZE))
    processed = np.expand_dims(processed, axis=0)
    
    return processed, orig

@app.post("/generate-heatmap/")
async def generate_heatmap(
    file: UploadFile = File(...),
    enable_gradcam: bool = True,
    background_tasks: BackgroundTasks = None
):
    try:
        # Load model using cached function
        model = get_model()
        
        # Process image
        contents = await file.read()
        processed_image, orig_image = await process_image(contents)

        # Make prediction
        preds = model.predict(processed_image, verbose=0)
        class_idx = np.argmax(preds[0])
        label = CLASS_LABELS[class_idx]
        probability = float(preds[0][class_idx])

        # Generate heatmap if enabled
        if enable_gradcam:
            cam = GradCAM(model, class_idx)
            heatmap = cam.compute_heatmap(processed_image)
            output = cam.overlay_heatmap(
                cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0])),
                orig_image
            )

            # Convert output to bytes
            is_success, output_buffer = cv2.imencode(".jpg", output)
            if not is_success:
                raise HTTPException(status_code=500, detail="Failed to encode output image")

            # Clean up GPU memory
            if background_tasks:
                background_tasks.add_task(tf.keras.backend.clear_session)

            return StreamingResponse(
                io.BytesIO(output_buffer.tobytes()),
                media_type="image/jpeg",
                headers={
                    "Prediction-Label": label,
                    "Prediction-Confidence": str(probability)
                }
            )
        else:
            # Return empty response body if GradCAM is disabled
            if background_tasks:
                background_tasks.add_task(tf.keras.backend.clear_session)

            return JSONResponse(
                content=None,
                headers={
                    "Prediction-Label": label,
                    "Prediction-Confidence": str(probability)
                },
                status_code=200
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)