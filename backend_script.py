
# ==========================================
# VASTUPLAN.AI - COLAB BACKEND (Free Tier Optimized)
# ==========================================
# Run this entire block in Google Colab

# 0. SET CACHE TO D: DRIVE (Prevents filling up C: drive)
import os
os.environ["HF_HOME"] = "D:\\huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:\\huggingface_cache\\hub"
os.environ["TRANSFORMERS_CACHE"] = "D:\\huggingface_cache\\transformers"

# 1. INSTALL DEPENDENCIES
import subprocess, sys
def install_dependencies():
    print("⏳ Installing Dependencies (this takes 2-3 mins)...")
    pkgs = [
        "fastapi", "uvicorn", "pydantic", "torch", "requests",
        "diffusers>=0.29.0", "transformers>=4.42.0",
        "accelerate>=0.30.0", "bitsandbytes>=0.43.1",
        "peft>=0.11.0", "python-multipart", "huggingface_hub",
        "sentencepiece", "nest_asyncio"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)
    
    # Fix for Colab bitsandbytes
    if os.path.exists("/usr/local/cuda/lib64"):
        os.environ["LD_LIBRARY_PATH"] += ":/usr/local/cuda/lib64"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"

try:
    import diffusers
except ImportError:
    install_dependencies()

# 2. IMPORTS & SETUP
import uvicorn
import nest_asyncio
import time, re, shutil, io, uuid, base64, threading
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from huggingface_hub import login
from transformers import BitsAndBytesConfig
from diffusers import FluxPipeline, FluxTransformer2DModel

nest_asyncio.apply()

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"🖥️  CUDA Available: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  No CUDA GPU detected!")
    print("   This script requires an NVIDIA GPU with CUDA support.")
    print("   Options:")
    print("   1. Use Google Colab with T4 GPU (free)")
    print("   2. Use Modal deployment (modal_deploy.py)")
    print("   3. Install CUDA drivers if you have an NVIDIA GPU")

# 3. LOAD MODEL (4-BIT QUANTIZED)
print("🔄 Initializing Model Pipeline...")

# Load HF_TOKEN from environment variable (set in .env file or system)
HF_TOKEN = os.environ.get("HF_TOKEN")  # Load from environment
if not HF_TOKEN:
    print("⚠️  WARNING: Please set HF_TOKEN in your .env file or environment!")
    print("   Get your token at: https://huggingface.co/settings/tokens")
else:
    login(token=HF_TOKEN)

pipe = None  # Initialize as None for non-GPU fallback

try:
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available - skipping model loading")
    
    # 4-bit Config
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model_id = "black-forest-labs/FLUX.1-Kontext-dev"

    print("📦 Loading Transformer (4-bit)...")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, 
        subfolder="transformer", 
        quantization_config=nf4_config, 
        torch_dtype=torch.bfloat16
    )

    print("🔗 Assembling Pipeline...")
    pipe = FluxPipeline.from_pretrained(
        model_id, 
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    
    pipe.enable_model_cpu_offload()
    print("✅ Model Ready!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Tip: Make sure you are using a GPU Runtime (Runtime > Change runtime type > T4 GPU)")

# 4. API SETUP
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
TASKS = {}

class PlanRequest(BaseModel):
    plot_width: float = 30.0
    plot_length: float = 40.0
    setback_front: float = 5.0
    setback_sides: float = 3.0
    orientation: str = "East"
    road_width: float = 30.0
    floors: str = "Ground Floor Only"
    entrance_loc: str = "North-East (Highly Preferred)"
    pooja_type: str = "Separate Room"
    kitchen_corner: bool = True
    master_corner: bool = True
    bhk: str = "2 BHK"
    family_type: str = "Nuclear Family"
    dining_style: str = "Central Heart (Connecting Kitchen/Living)"
    has_utility: bool = True
    has_veranda: bool = True
    style: str = "Modern Indian"
    steps: int = 25
    guidance: float = 3.5
    mode: str = "Text Only"
    ref_image_base64: Optional[str] = None 

def run_generation(task_id: str, req: PlanRequest):
    global pipe
    print(f"[{task_id}] Processing...")
    try:
        # Check if model is loaded
        if pipe is None:
            raise RuntimeError("Model not loaded. GPU required for local generation. Use Modal deployment instead.")
        
        # Construct Prompt
        vastu_constraints = []
        if req.entrance_loc: vastu_constraints.append(f"MAIN ENTRANCE: {req.entrance_loc}")
        if req.kitchen_corner: vastu_constraints.append("KITCHEN: South-East (Agni)")
        if req.master_corner: vastu_constraints.append("MASTER BEDROOM: South-West (Nairutya)")
        if req.pooja_type != "None": vastu_constraints.append(f"POOJA ROOM: {req.pooja_type}")

        features = []
        if req.has_utility: features.append("UTILITY AREA")
        if req.has_veranda: features.append("TRADITIONAL THINNAI/VERANDA")

        prompt = (
            f"Professional 2D CAD architectural floorplan for a {req.style} {req.floors}. "
            f"PLOT: {req.plot_width}x{req.plot_length}ft, {req.orientation} facing. "
            f"SETBACKS: Front {req.setback_front}ft, sides {req.setback_sides}ft. "
            f"VASTU: {', '.join(vastu_constraints)}. PROGRAM: {req.bhk}. "
            f"FEATURES: {', '.join(features)}. DINING: {req.dining_style}. "
            "Technical orthographic view, architectural drafting style, high-contrast, labeled rooms."
        )

        image = pipe(
            prompt, 
            num_inference_steps=req.steps, 
            guidance_scale=req.guidance,
            max_sequence_length=512,
            height=768,
            width=1024
        ).images[0]

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        TASKS[task_id] = {"status": "completed", "image_base64": img_str}
        print(f"[{task_id}] Done.")
    except Exception as e:
        TASKS[task_id] = {"status": "failed", "error": str(e)}

@app.post("/generate")
async def start_generation(req: PlanRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "processing"}
    background_tasks.add_task(run_generation, task_id, req)
    return {"task_id": task_id}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str): return TASKS.get(task_id, {"status": "failed"})

@app.get("/")
def root(): return {"message": "VastuPlan Colab Backend Online"}

# 5. START SERVER & CLOUDFLARE
if __name__ == "__main__":
    import platform
    
    def start_tunnel():
        is_windows = platform.system() == "Windows"
        
        if is_windows:
            cloudflared_exe = "cloudflared.exe"
            download_url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
            
            if not os.path.exists(cloudflared_exe):
                print("📥 Downloading cloudflared for Windows...")
                import urllib.request
                urllib.request.urlretrieve(download_url, cloudflared_exe)
                print("✅ Downloaded cloudflared.exe")
            
            proc = subprocess.Popen([cloudflared_exe, "tunnel", "--url", "http://localhost:8000"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        else:
            # Linux / Colab
            if not os.path.exists("./cloudflared"):
                subprocess.run(["wget", "-q", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "cloudflared"])
                subprocess.run(["chmod", "+x", "cloudflared"])
            
            proc = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        time.sleep(2)
        end = time.time() + 20
        while time.time() < end:
            line = proc.stderr.readline()
            if match := re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line):
                print(f"\n✅ PUBLIC URL: {match.group(0)}\n")
                return proc
        print("❌ Tunnel failed. Server still running at http://localhost:8000")
        return proc

    cf_proc = start_tunnel()
    try: uvicorn.run(app, host="0.0.0.0", port=8000)
    finally: 
        if cf_proc: cf_proc.terminate()
