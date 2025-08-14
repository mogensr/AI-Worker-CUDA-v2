import os
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Initialize SAM2 model
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

if not os.path.exists(checkpoint):
    os.makedirs("./checkpoints", exist_ok=True)
    # Download checkpoint
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="facebook/sam2-hiera-large",
        filename="sam2_hiera_large.pt",
        local_dir="./checkpoints"
    )

sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for prediction
        predictor.set_image(image_rgb)
        
        # Auto-generate masks
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True,
        )
        
        # Return best mask
        best_mask = masks[np.argmax(scores)]
        
        return JSONResponse({
            "success": True,
            "mask_shape": best_mask.shape,
            "score": float(np.max(scores))
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })

@app.get("/health")
async def health():
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
