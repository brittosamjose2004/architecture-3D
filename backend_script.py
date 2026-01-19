
# ==========================================
# PASTE THIS ENTIRE SCRIPT INTO COLAB/KAGGLE
# ==========================================

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
from diffusers import FluxPipeline
import base64
import io
import uuid
import nest_asyncio # Fix for Colab/Jupyter

# Apply Nest Asyncio to allow uvicorn to run in Colab
nest_asyncio.apply()
import threading
import time

# --- 1. SETUP MODELS ---
print("Loading FLUX.1 Model... (This may take a minute)")
try:
    # Adjust model ID if you are using a different one
    flux_pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.bfloat16
    )
    flux_pipe.enable_model_cpu_offload() # Saves VRAM
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure you have logged in with huggingface-cli login or provided a token.")

# --- 2. DATA MODELS ---
class PlanRequest(BaseModel):
    plot_width: float
    plot_length: float
    setback_front: float
    setback_sides: float
    orientation: str
    road_width: float
    floors: str
    entrance_loc: str
    pooja_type: str
    kitchen_corner: bool
    master_corner: bool
    bhk: str
    family_type: str
    dining_style: str
    has_utility: bool
    has_veranda: bool
    style: str
    steps: int
    guidance: float
    mode: str
    ref_image_base64: Optional[str] = None # Expecting raw base64 string, no data URI prefix

# --- 3. STATE MANAGEMENT ---
app = FastAPI()

# IMPORTANT: Enable CORS for React App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Task Store
# Format: { "task_id": { "status": "processing" | "completed" | "failed", "image_base64": "...", "error": "..." } }
TASKS = {} 

def generate_plan_task(task_id: str, req: PlanRequest):
    """
    Background worker function that runs the AI model.
    """
    print(f"[{task_id}] Starting generation...")
    try:
        # Construct Prompt
        prompt = (
            f"A professional architectural floor plan: {req.style} style, {req.bhk}, "
            f"Plot {req.plot_width}x{req.plot_length} ft, Facing {req.orientation}. "
            f"Vastu: Entrance {req.entrance_loc}, Pooja {req.pooja_type}. "
            f"Kitchen SE: {req.kitchen_corner}, Master SW: {req.master_corner}. "
            f"High contrast blueprint, technical drawing, white lines on blue background."
        )

        # Handle Reference Image (Img2Img) if provided
        image_input = None
        if req.ref_image_base64 and "Image" in req.mode:
            try:
                img_data = base64.b64decode(req.ref_image_base64)
                # You would load this into PIL here if acting as control/ref image
                # image_input = Image.open(io.BytesIO(img_data)).convert("RGB")
                print(f"[{task_id}] Reference image received (Logic TBD based on pipeline support)")
            except Exception as e:
                print(f"[{task_id}] Image decode warning: {e}")

        # --- GENERATION STEP ---
        # Note: Actual inputs depend on your specific pipeline (txt2img vs img2img)
        # using standard txt2img for demo stability
        image = flux_pipe(
            prompt,
            height=768, # Optimize for your VRAM
            width=1024,
            guidance_scale=req.guidance,
            num_inference_steps=req.steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]

        # Convert to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Update Status
        TASKS[task_id] = {
            "status": "completed",
            "image_base64": img_str
        }
        print(f"[{task_id}] Completed successfully.")

    except Exception as e:
        print(f"[{task_id}] FAILED: {str(e)}")
        TASKS[task_id] = {
            "status": "failed",
            "error": str(e)
        }

@app.post("/generate")
async def start_generation(req: PlanRequest, background_tasks: BackgroundTasks):
    """
    ASYNC ENDPOINT: Returns a task_id immediately.
    """
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "processing"}
    
    # Run generation in background so we don't block the request
    background_tasks.add_task(generate_plan_task, task_id, req)
    
    return {"task_id": task_id, "message": "Generation started"}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    POLLING ENDPOINT: React checks this every few seconds.
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TASKS[task_id]

@app.get("/")
def home():
    return {"status": "Online", "mode": "Async/Non-Blocking"}

# --- 4. START SERVER & CLOUDFLARE ---
if __name__ == "__main__":
    import subprocess
    import time
    import re
    import shutil
    import os

    def start_cloudflare_tunnel(port):
        """Downloads and starts Cloudflare Tunnel, returning the public URL."""
        print("-" * 60)
        print("☁️  Setting up Cloudflare Tunnel (No Token Required)...")
        
        # 1. Download cloudflared if not present
        if not os.path.exists("./cloudflared"):
            print("🔽 Downloading cloudflared binary...")
            subprocess.run(["wget", "-q", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "cloudflared"])
            subprocess.run(["chmod", "+x", "cloudflared"])

        # 2. Start the tunnel
        print("🚀 Starting tunnel...")
        # We need to capture stderr because that's where the URL is printed
        process = subprocess.Popen(
            ["./cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        # 3. Read logs to find the URL
        time.sleep(2) # Give it a moment
        cf_url = None
        end_time = time.time() + 20 # Wait max 20 seconds for URL
        
        while time.time() < end_time:
            line = process.stderr.readline()
            if not line:
                break
            
            # Look for regex: https://[random].trycloudflare.com
            match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if match:
                cf_url = match.group(0)
                break
        
        if cf_url:
            print("=" * 60)
            print(f"✅ PUBLIC API URL: {cf_url}")
            print("👉 COPY THIS URL into your VastuPlan React App")
            print("=" * 60)
        else:
            print("❌ Failed to grab Cloudflare URL. Check logs.")
            
        return process

    # Start Cloudflare
    try:
        cf_process = start_cloudflare_tunnel(8000)
    except Exception as e:
        print(f"Tunnel Error: {e}")

    # Start FastAPI
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        if 'cf_process' in locals():
            cf_process.terminate()
