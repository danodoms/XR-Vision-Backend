from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image
from typing import Tuple
from functools import lru_cache

# Force CPU usage and optimize TF for CPU
tf.config.set_visible_devices([], 'GPU')
# tf.config.threading.set_intra_op_parallelism_threads(4)
# tf.config.threading.set_inter_op_parallelism_threads(2)

app = FastAPI()

# Constants
MODEL_PATH = "./model/npk-classifier-v2-functional.keras"
STANDARD_SIZE = (224, 224)
PERFORMANCE_SIZE = (128, 128)
CLASS_LABELS = ["Healthy", "Nitrogen Deficient", "Phosphorus Deficient", "Potassium Deficient"]

@lru_cache(maxsize=1)
def get_model():
    model = load_model(MODEL_PATH)
    return model

class GradCAM:
    def __init__(self, model: tf.keras.Model, class_idx: int):
        self.model = model
        self.class_idx = class_idx
        self.layer_name = next(layer.name for layer in reversed(self.model.layers) 
                             if len(layer.output.shape) == 4)
        
    def compute_heatmap(self, image: np.ndarray) -> np.ndarray:
        grad_model = tf.keras.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, self.class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
        
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
        return np.uint8(255 * heatmap)

    @staticmethod
    def overlay_heatmap(heatmap: np.ndarray, image: np.ndarray) -> np.ndarray:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

def preprocess_image(image_data: bytes, performance_mode: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # Read image directly into numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    orig_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Choose size based on performance mode
    target_size = PERFORMANCE_SIZE if performance_mode else STANDARD_SIZE
    
    # For performance mode, resize the original image first to reduce memory usage
    if performance_mode:
        orig_image = cv2.resize(orig_image, PERFORMANCE_SIZE)
    
    # Resize efficiently using cv2
    input_image = cv2.resize(orig_image, target_size)
    input_image = np.expand_dims(input_image, 0)
    
    return input_image, orig_image

@app.post("/generate-heatmap/")
async def generate_heatmap(
    file: UploadFile = File(...),
    enable_gradcam: bool = True,
    performance_mode: bool = False
) -> StreamingResponse:
    try:
        # Read file content
        contents = await file.read()
        
        # Process image efficiently with performance mode option
        input_image, orig_image = preprocess_image(contents, performance_mode)
        
        # Get cached model and make prediction
        model = get_model()
        preds = model(input_image, training=False)
        class_idx = int(tf.argmax(preds[0]))
        probability = float(preds[0][class_idx])
        
        # Generate heatmap if needed
        if enable_gradcam:
            cam = GradCAM(model, class_idx)
            heatmap = cam.compute_heatmap(input_image)
            heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))
            output = cam.overlay_heatmap(heatmap, orig_image)
        else:
            output = orig_image

        # Encode output efficiently
        # Use higher compression in performance mode
        quality = 80 if performance_mode else 90
        _, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={
                "Prediction-Label": CLASS_LABELS[class_idx],
                "Prediction-Confidence": str(probability),
                "Performance-Mode": str(performance_mode)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)