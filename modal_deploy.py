
import modal
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
import base64
import io
import os

# 1. DEFINE IMAGE & DEPENDENCIES
def download_model_weights():
    # Helper to download weights DURING IMAGE BUILD (Saves runtime)
    from huggingface_hub import snapshot_download, login
    import os
    
    # Use the token hardcoded for build (or use Modal Secrets in production)
    token = os.environ["HF_TOKEN"]
    login(token=token)
    
    print("⬇️ Downloading FLUX.1-Kontext-dev (Build Time)...")
    snapshot_download("black-forest-labs/FLUX.1-Kontext-dev")
    print("✅ Download Complete")

image = (
    modal.Image.debian_slim()
    # Install dependencies
    .pip_install(
        "fastapi", "uvicorn", "pydantic", "torch", "requests",
        "diffusers>=0.29.0", "transformers>=4.42.0",
        "accelerate>=0.30.0", "bitsandbytes>=0.43.1",
        "peft>=0.11.0", "python-multipart", "huggingface_hub"
    )
    # Fix CUDA Paths for bitsandbytes
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib64-nvidia"})
    .env({"BITSANDBYTES_NOWELCOME": "1"})
    # Download Model Layers NOW (So it doesn't timeout later)
    .run_function(download_model_weights)
)

app = modal.App("vastu-plan-backend")

# 2. PERSISTENT STORAGE
results_dict = modal.Dict.from_name("vastu-results", create_if_missing=True)

# 3. GPU WORKER CLASS
@app.cls(gpu="A10G", image=image, timeout=3600, keep_warm=1, container_idle_timeout=300)
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig
        from huggingface_hub import login
        
        import os
        print("🔄 Initializing Model Pipeline...")
        login(token=os.environ["HF_TOKEN"])

        # 4-bit Config
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model_id = "black-forest-labs/FLUX.1-Kontext-dev"

        try:
            # Load Transformer (Should be cached from build step)
            print("📦 Loading Transformer (4-bit)...")
            transformer = FluxTransformer2DModel.from_pretrained(
                model_id, 
                subfolder="transformer", 
                quantization_config=nf4_config, 
                torch_dtype=torch.bfloat16
            )

            print("🔗 Assembling Pipeline...")
            self.pipe = FluxPipeline.from_pretrained(
                model_id, 
                transformer=transformer,
                torch_dtype=torch.bfloat16
            )
            
            print("⚙️ Offloading to CPU...")
            self.pipe.enable_model_cpu_offload()
            print("✅ Model Ready!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    @modal.method()
    def generate(self, prompt: str, steps: int = 25, guidance: float = 3.5):
        import torch
        import io
        import base64
        
        print(f"🎨 Generating: {prompt[:50]}...")
        
        image = self.pipe(
            prompt, 
            num_inference_steps=steps, 
            guidance_scale=guidance,
            max_sequence_length=512,
            height=768,
            width=1024
        ).images[0]

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str

# 4. API DEFINITION
web_app = FastAPI(title="VastuPlan AI API")

# Input Schema (Matches User's Request)
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

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.get("/")
def root():
    return {"message": "VastuPlan AI API is Online", "status": "Running"}

@web_app.get("/health")
def health():
    return {"status": "healthy"}

@web_app.post("/generate")
async def start_generation(req: PlanRequest):
    task_id = str(uuid.uuid4())
    
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

    results_dict[task_id] = {"status": "processing"}
    
    # Spawn background task
    run_generation.spawn(task_id, prompt, req.steps, req.guidance)
    
    return {"task_id": task_id}

@app.function(image=image, timeout=7200)
def run_generation(task_id: str, prompt: str, steps: int, guidance: float):
    try:
        model = Model()
        img_str = model.generate.remote(prompt, steps, guidance)
        results_dict[task_id] = {"status": "completed", "image_base64": img_str}
    except Exception as e:
        print(f"FAILED {task_id}: {e}")
        results_dict[task_id] = {"status": "failed", "error": str(e)}

@web_app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in results_dict:
        raise HTTPException(status_code=404, detail="Task not found")
    return results_dict[task_id]

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
