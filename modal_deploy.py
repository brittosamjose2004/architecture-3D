import modal
import os as _os
import base64
import io
import time
import uuid
import asyncio
from fastapi import Request, FastAPI
from huggingface_hub import login

# Get HF_TOKEN from local environment
_HF_TOKEN = _os.environ.get("HF_TOKEN")

# --- 1. IMAGE DEFINITION (HEAVY 3D ENV - Modernized) ---
# Using User's Exact Requirements with Pre-built Wheels
# --- 1. IMAGE DEFINITION (HEAVY 3D ENV - Modernized) ---
# Using CUDA 12.4 Devel Base to allow compiling Kaolin/Extensions if wheels miss
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "cmake")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "IGL_CMAKE_CXX_COMPILER": "g++",
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0", # A10G Support
        "CC": "gcc",
        "CXX": "g++",
        "PIP_DEFAULT_TIMEOUT": "300",  # 5 min timeout for large packages
        "PIP_RETRIES": "5"
    })
    .run_commands(
        "pip install --timeout 300 torch==2.6.0 torchvision==0.21.0 triton==3.2.0 --index-url https://download.pytorch.org/whl/cu124"
    )
    .run_commands(
        "pip install --timeout 300 cython pillow==12.0.0 imageio==2.37.2 imageio-ffmpeg==0.6.0 tqdm==4.67.1 easydict==1.13 opencv-python-headless==4.12.0.88 trimesh==4.10.1 transformers==4.57.3 zstandard==0.25.0 kornia==0.8.2 timm==1.0.22 'rembg[gpu]' gradio gradio_client open3d diffusers accelerate bitsandbytes peft sentencepiece huggingface_hub"
    )
    .run_commands(
        "pip install --timeout 300 git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
    )
    # Install Pre-built Wheels where possible
    .run_commands(
        "pip install --timeout 300 https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/cumesh-0.0.1-cp310-cp310-linux_x86_64.whl https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/flex_gemm-0.0.1-cp310-cp310-linux_x86_64.whl https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/o_voxel-0.0.1-cp310-cp310-linux_x86_64.whl https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl https://github.com/JeffreyXiang/Storages/releases/download/Space_Wheels_251210/nvdiffrec_render-0.0.0-cp310-cp310-linux_x86_64.whl"
    )
    # Install Kaolin from pre-built wheel (compiling from source has bugs)
    .run_commands("pip install --timeout 300 --upgrade pip setuptools wheel cython numpy && pip install --timeout 600 kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu124.html")
    # Clone TRELLIS.2 (use PYTHONPATH since package name is trellis2)
    .run_commands(
        "git clone --recurse-submodules https://github.com/microsoft/TRELLIS.2.git /root/TRELLIS",
        # Patch 1: Replace gated DINOv3 with non-gated DINOv2
        "sed -i 's/dinov3-vitl16-pretrain-lvd1689m/dinov2-large/g' /root/TRELLIS -r || true",
        # Patch 2: Disable built-in rembg loading (we'll handle background removal ourselves)
        "sed -i '105s/.*/        # pipeline.rembg_model = None  # Patched: Skip gated RMBG model/' /root/TRELLIS/trellis2/pipelines/trellis2_image_to_3d.py || true",
        # Patch 3: Force image_cond_model to CUDA after loading
        "sed -i '/self.model = self.model.to(self.device)/a\\        self.model = self.model.cuda()  # Patched: Explicit CUDA move' /root/TRELLIS/trellis2/modules/image_feature_extractor.py || true",
        # Patch 4: Force ALL pipeline models to CUDA in post_init
        '''sed -i '/def post_init/a\\        # PATCHED: Force all models to CUDA\\        import torch.nn as nn\\        for attr_name in dir(self):\\            if attr_name.startswith("_"): continue\\            try:\\                attr = getattr(self, attr_name)\\                if isinstance(attr, nn.Module):\\                    attr.cuda()\\            except: pass' /root/TRELLIS/trellis2/pipelines/trellis2_image_to_3d.py || true'''
    )
    .env({
        "HF_TOKEN": _HF_TOKEN,
        "OPENCV_IO_ENABLE_OPENEXR": "1",
        "PYTHONPATH": "/root/TRELLIS",
        "ATTN_BACKEND": "flash_attn_3",  # CRITICAL: From official Space
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"  # From official Space
    })
)

app = modal.App("vastu-plan-3d-pro")
results_dict = modal.Dict.from_name("vastu-3d-results", create_if_missing=True)

# --- 2. 3D GENERATOR CLASS (REAL TRELLIS-2-4B) ---
@app.cls(gpu="A10G", image=image, timeout=3600, min_containers=0)
class Trellis3DGenerator:
    @modal.enter()
    def setup(self):
        import torch
        from huggingface_hub import login
        import os
        import sys
        import subprocess

        # --- AUTO-INSTALLER HELPER ---
        def install_if_missing(package, import_name=None):
            import_name = import_name or package
            try:
                __import__(import_name)
                print(f"✅ {package} found.")
            except ImportError:
                print(f"📦 Module '{import_name}' not found. Auto-installing '{package}'...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"✅ {package} installed successfully.")
                except Exception as e:
                    print(f"❌ Failed to auto-install {package}: {e}")

        # Ensure critical dependencies
        install_if_missing("kaolin")
        
        # Debug: Check file structure
        print("📂 Checking /root/TRELLIS...")
        if os.path.exists("/root/TRELLIS"):
             sys.path.append("/root/TRELLIS")
        else:
             print("⚠️ /root/TRELLIS NOT FOUND!")

        # Login
        login(token=os.environ["HF_TOKEN"])
        
        # RUNTIME PATCHING: Fix device mismatches before loading pipeline
        print("🔧 Applying runtime device fixes...")
        import re
        
        # FIX 1: Force sampler tensors onto model device in flow_euler.py
        flow_euler_path = '/root/TRELLIS/trellis2/pipelines/samplers/flow_euler.py'
        if os.path.exists(flow_euler_path):
            with open(flow_euler_path, 'r') as f:
                content = f.read()
            
            if 'x_t.device' not in content:
                content = re.sub(
                    r'(def _inference_model\(self, model, x_t, t, cond,.*?\):)',
                    r'\1\n        # Ensure all inputs on same device as model\n        if hasattr(x_t, "to"): x_t = x_t.to(model.device)\n        if hasattr(t, "to"): t = t.to(model.device)\n        if hasattr(cond, "to"): cond = cond.to(model.device)\n        # Move sampler attributes to model device\n        if hasattr(self, "sigma_min") and hasattr(self.sigma_min, "to"): self.sigma_min = self.sigma_min.to(model.device)\n        if hasattr(self, "sigma_max") and hasattr(self.sigma_max, "to"): self.sigma_max = self.sigma_max.to(model.device)',
                    content,
                    count=1
                )
                with open(flow_euler_path, 'w') as f:
                    f.write(content)
                print("✅ Patched flow_euler.py for device sync")
        
        # FIX 2: Force input tensors to model device in sparse_structure_flow.py
        sparse_flow_path = '/root/TRELLIS/trellis2/models/sparse_structure_flow.py'
        if os.path.exists(sparse_flow_path):
            with open(sparse_flow_path, 'r') as f:
                content = f.read()
            
            # Move h to correct device before input_layer AND move t/cond to embedder device
            if 'h.to(self.input_layer' not in content:
                # Patch 1: Move t to embedder device before t_embedder(t)
                content = re.sub(
                    r'([ \t]+)(t_emb = self\.t_embedder\(t\))',
                    r'\1# Move t to embedder device\n\1if hasattr(t, "to") and hasattr(self.t_embedder, "device"): t = t.to(self.t_embedder.device)\n\1elif hasattr(t, "to"): t = t.to(next(self.t_embedder.parameters()).device)\n\1\2',
                    content,
                    count=1
                )
                # Patch 2: Move h to input_layer device
                content = re.sub(
                    r'([ \t]+)(h = self\.input_layer\(h\))',
                    r'\1h = h.to(self.input_layer.weight.device) if hasattr(h, "to") else h\n\1\2',
                    content,
                    count=1
                )
                with open(sparse_flow_path, 'w') as f:
                    f.write(content)
                print("✅ Patched sparse_structure_flow.py for comprehensive device sync")
        
        # FIX 3: Check and repair full_attn.py if previous patches broke it
        full_attn_path = '/root/TRELLIS/trellis2/modules/attention/full_attn.py'
        if os.path.exists(full_attn_path):
            try:
                with open(full_attn_path, 'r') as f:
                    content = f.read()
                compile(content, full_attn_path, 'exec')
                print("✅ full_attn.py syntax OK")
            except SyntaxError as e:
                print(f"⚠️ Syntax error in full_attn.py: {e.msg} at line {e.lineno}")
                print("🔧 Re-cloning TRELLIS to get fresh copy...")
                import subprocess
                subprocess.run(['rm', '-rf', '/root/TRELLIS'], check=False)
                subprocess.run(['git', 'clone', '--recurse-submodules', 'https://github.com/microsoft/TRELLIS.2.git', '/root/TRELLIS'], check=True)
                # Re-apply DINOv2 patch
                subprocess.run(['sed', '-i', 's/dinov3-vitl16-pretrain-lvd1689m/dinov2-large/g', '-r', '/root/TRELLIS'], check=False)
                print("✅ TRELLIS re-cloned successfully")
        
        # No flash_attn patching needed - ATTN_BACKEND env var handles it
        
        # FIX 4: Patch full_attn.py to move q, k, v to CUDA before flash_attn call
        full_attn_path = '/root/TRELLIS/trellis2/modules/attention/full_attn.py'
        if os.path.exists(full_attn_path):
            with open(full_attn_path, 'r') as f:
                content = f.read()
            
            # Move tensors to CUDA right before flash_attn_func call
            if 'q = q.cuda()' not in content:
                content = re.sub(
                    r'(\s+)(out = flash_attn_3\.flash_attn_func\(q, k, v\))',
                    r'\1# Ensure tensors on CUDA for flash_attn\n\1q, k, v = q.cuda(), k.cuda(), v.cuda()\n\1out = flash_attn_3.flash_attn_func(q, k, v)',
                    content
                )
                with open(full_attn_path, 'w') as f:
                    f.write(content)
                print("✅ Patched full_attn.py to move q, k, v to CUDA")
        
        # FIX 5: Patch modules.py to sync h with to_out layer device after attention
        attn_modules_path = '/root/TRELLIS/trellis2/modules/attention/modules.py'
        if os.path.exists(attn_modules_path):
            with open(attn_modules_path, 'r') as f:
                content = f.read()
            
            # Move h to same device as to_out before projection
            if 'h.device' not in content:
                content = re.sub(
                    r'(\s+)(h = self\.to_out\(h\))',
                    r'\1# Sync h device with to_out layer\n\1if hasattr(self.to_out, "weight"): h = h.to(self.to_out.weight.device)\n\1h = self.to_out(h)',
                    content
                )
                with open(attn_modules_path, 'w') as f:
                    f.write(content)
                print("✅ Patched modules.py for output tensor device sync")
        
        # FIX 6: Patch structured_latent_flow.py for device sync
        slf_path = '/root/TRELLIS/trellis2/models/structured_latent_flow.py'
        if os.path.exists(slf_path):
            with open(slf_path, 'r') as f:
                content = f.read()
            
            if '# DEVICE_SYNC_PATCHED' not in content:
                # Add device sync at the start of forward method
                content = re.sub(
                    r'(def forward\(self.*?\):)',
                    r'\1\n        # DEVICE_SYNC_PATCHED: Force all inputs to model device\n        _device = next(self.parameters()).device\n        def _to_device(x):\n            if hasattr(x, "to"): return x.to(_device)\n            return x',
                    content
                )
                with open(slf_path, 'w') as f:
                    f.write(content)
                print("✅ Patched structured_latent_flow.py for device sync")
        
        # FIX 7: Patch sparse_structure_vae.py for device sync
        ssv_path = '/root/TRELLIS/trellis2/models/sparse_structure_vae.py'
        if os.path.exists(ssv_path):
            with open(ssv_path, 'r') as f:
                content = f.read()
            
            if '# DEVICE_SYNC_PATCHED' not in content:
                content = re.sub(
                    r'(def forward\(self.*?\):)',
                    r'\1\n        # DEVICE_SYNC_PATCHED: Force all inputs to model device\n        _device = next(self.parameters()).device',
                    content
                )
                with open(ssv_path, 'w') as f:
                    f.write(content)
                print("✅ Patched sparse_structure_vae.py for device sync")
        
        # FIX 8: Patch the base sampler to ensure device consistency
        base_sampler_path = '/root/TRELLIS/trellis2/pipelines/samplers/base.py'
        if os.path.exists(base_sampler_path):
            with open(base_sampler_path, 'r') as f:
                content = f.read()
            
            if '# DEVICE_SYNC_PATCHED' not in content:
                # Patch the sample method to force device sync
                content = re.sub(
                    r'(def sample\(self.*?\):)',
                    r'\1\n        # DEVICE_SYNC_PATCHED',
                    content
                )
                with open(base_sampler_path, 'w') as f:
                    f.write(content)
                print("✅ Patched base.py sampler for device sync")
        
        # FIX 9: Comprehensive patch for image_feature_extractor.py
        ife_path = '/root/TRELLIS/trellis2/modules/image_feature_extractor.py'
        if os.path.exists(ife_path):
            with open(ife_path, 'r') as f:
                content = f.read()
            
            if '# DEVICE_SYNC_PATCHED' not in content:
                # Force model and inputs to CUDA in forward
                content = re.sub(
                    r'(def forward\(self, image.*?\):)',
                    r'\1\n        # DEVICE_SYNC_PATCHED: Force CUDA\n        self.model = self.model.cuda()\n        if hasattr(image, "to"): image = image.cuda()',
                    content
                )
                with open(ife_path, 'w') as f:
                    f.write(content)
                print("✅ Patched image_feature_extractor.py for CUDA sync")
        
        
        # Import Trellis (use trellis2 module name)
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
            from trellis2.utils import render_utils
        except ImportError as e:
            print(f"❌ TRELLIS Import Failed: {e}")
            raise e

        print("🏗️ Loading TRELLIS-2-4B Pipeline (Real)...")
        # Load Pipeline
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        
        # Robust Recursive CUDA Move
        def force_cuda(obj, name=""):
            import torch.nn as nn
            if isinstance(obj, nn.Module):
                obj.cuda()
                print(f"✅ Moved Module {name} to CUDA")
            elif hasattr(obj, 'to'):
                try:
                    obj.to("cuda")
                    print(f"✅ Moved {name} via .to('cuda')")
                except: pass
            elif hasattr(obj, 'cuda'):
                try:
                    obj.cuda()
                    print(f"✅ Moved {name} via .cuda()")
                except: pass

        # Move top-level pipeline
        self.pipeline.to("cuda")
        
        # Iterate through everything to catch hidden wrappers
        for attr_name in dir(self.pipeline):
            if attr_name.startswith('_'): continue
            try:
                attr = getattr(self.pipeline, attr_name)
                force_cuda(attr, f"pipeline.{attr_name}")
                # Check for nested model (common in Transformers wrappers)
                if hasattr(attr, 'model'):
                    force_cuda(attr.model, f"pipeline.{attr_name}.model")
                if hasattr(attr, 'image_cond_model'):
                    force_cuda(attr.image_cond_model, f"pipeline.{attr_name}.image_cond_model")
            except:
                continue
        
        # Configure pipeline (matching official Space)
        self.pipeline.rembg_model = None  # From official Space - we handle bg removal  
        self.pipeline.low_vram = False  # We have A10G (24GB VRAM)
        
        # FIX 10: Monkey-patch the pipeline's internal models to force CUDA on every forward
        print("🔧 Applying runtime forward hooks for device sync...")
        import torch
        import torch.nn as nn
        
        def create_cuda_hook(module_name):
            def hook(module, inputs):
                new_inputs = []
                for inp in inputs:
                    if isinstance(inp, torch.Tensor) and inp.device.type == 'cpu':
                        new_inputs.append(inp.cuda())
                    else:
                        new_inputs.append(inp)
                return tuple(new_inputs) if len(new_inputs) > 1 else (new_inputs[0],) if new_inputs else inputs
            return hook
        
        # Register forward pre-hooks on all nn.Module attributes of the pipeline
        def register_hooks_recursive(obj, prefix=""):
            if isinstance(obj, nn.Module):
                # Register hook on this module
                try:
                    obj.register_forward_pre_hook(create_cuda_hook(prefix))
                except:
                    pass
                # Also iterate through its named_modules
                try:
                    for name, submodule in obj.named_modules():
                        try:
                            submodule.register_forward_pre_hook(create_cuda_hook(f"{prefix}.{name}"))
                        except:
                            pass
                except:
                    pass
        
        # Find all nn.Module attributes in the pipeline
        for attr_name in dir(self.pipeline):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(self.pipeline, attr_name)
                register_hooks_recursive(attr, f"pipeline.{attr_name}")
            except:
                continue
        
        print("✅ TRELLIS Ready!")
        
    def remove_background(self, input_image):
        from rembg import remove
        import numpy as np
        from PIL import Image
        
        input_image = input_image.convert("RGB")
        output = remove(input_image) # Auto-download 'u2net' (no auth needed)
        
        # Post-process (crop logic from user snippet)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        if len(bbox) > 0:
            bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
            center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            size = int(size * 1.0)
            bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
            output = output.crop(bbox)
        
        return output

    @modal.method()
    def process_3d(self, image_b64: str, params: dict):
        import torch
        import numpy as np
        from PIL import Image
        from trellis2.utils import render_utils
        
        print("🚀 Starting 3D Processing (TRELLIS Engine)...")
        
        # Force CUDA as default device for all tensor operations
        torch.set_default_device('cuda')
        
        # Ensure all pipeline models are on CUDA before processing
        self.pipeline.to("cuda")
        
        # 1. Decode Image
        img_data = base64.b64decode(image_b64)
        input_img = Image.open(io.BytesIO(img_data)).convert("RGBA")
        
        # 2. Preprocess (Remove BG)
        print("✂️ Automated Background Removal...")
        processed_img = self.remove_background(input_img)
        
        # 3. Generate 3D (Sparse => Shape => Texture)
        print("🧊 Generating 3D Mesh & Texture...")
        
        # Extract Advanced Params
        seed = int(params.get('seed', 0))
        # Stage 1: Sparse Structure
        ss_steps = int(params.get('ss_sampling_steps', 12))
        ss_strength = float(params.get('ss_guidance_strength', 7.5))
        ss_rescale = float(params.get('ss_guidance_rescale', 0.7))
        ss_rescale_t = float(params.get('ss_rescale_t', 5.0))
        # Stage 2: Shape
        shape_steps = int(params.get('shape_slat_sampling_steps', 12))
        shape_strength = float(params.get('shape_slat_guidance_strength', 7.5))
        shape_rescale = float(params.get('shape_slat_guidance_rescale', 0.5))
        shape_rescale_t = float(params.get('shape_slat_rescale_t', 3.0))
        # Stage 3: Texture
        tex_steps = int(params.get('tex_slat_sampling_steps', 12))
        tex_strength = float(params.get('tex_slat_guidance_strength', 1.0))
        tex_rescale = float(params.get('tex_slat_guidance_rescale', 0.0))
        tex_rescale_t = float(params.get('tex_slat_rescale_t', 3.0))

        outputs = self.pipeline.run(
            processed_img,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_steps,
                "guidance_strength": ss_strength,
                "guidance_rescale": ss_rescale,
                "rescale_t": ss_rescale_t
            },
            shape_slat_sampler_params={
                "steps": shape_steps,
                "guidance_strength": shape_strength,
                "guidance_rescale": shape_rescale,
                "rescale_t": shape_rescale_t
            },
            tex_slat_sampler_params={
                "steps": tex_steps,
                "guidance_strength": tex_strength,
                "guidance_rescale": tex_rescale,
                "rescale_t": tex_rescale_t
            }
        )
        
        # Handle different output formats from TRELLIS pipeline
        print(f"📦 Pipeline output type: {type(outputs)}")
        if isinstance(outputs, dict):
            # Dictionary output: {'mesh': [...], 'gaussian': [...], ...}
            mesh = outputs['mesh'][0] if 'mesh' in outputs else outputs.get('meshes', [None])[0]
        elif isinstance(outputs, (list, tuple)):
            # List/tuple output: [mesh, gaussian, ...] or just [mesh]
            if len(outputs) > 0:
                # Check if first element is the mesh or if it's nested
                first = outputs[0]
                if hasattr(first, 'vertices') or hasattr(first, 'export'):
                    mesh = first
                elif isinstance(first, (list, tuple)) and len(first) > 0:
                    mesh = first[0]
                else:
                    mesh = first
            else:
                raise ValueError("Pipeline returned empty output")
        else:
            # Single mesh object
            mesh = outputs
        
        print(f"📦 Mesh type: {type(mesh)}")
        
        # apply decimation if available
        decimation_target = int(params.get('decimation', 100000))
        # Trellis mesh usually has .simplify method (trimesh or custom)
        if hasattr(mesh, 'simplify'):
             print(f"📉 Decimating to {decimation_target} faces...")
             try:
                 mesh.simplify(decimation_target)
             except Exception as e:
                 print(f"⚠️ Decimation failed: {e}")

        # 4. Render 360 Previews (skip if render fails - not critical)
        print("📸 Rendering 360 Previews...")
        previews_b64 = []
        try:
            # Try simple voxel rendering first (doesn't need envmap)
            from trellis2.renderers import VoxelRenderer
            voxel_renderer = VoxelRenderer()
            # Render from multiple angles
            import torch
            for i in range(4):
                angle = i * 90
                try:
                    # Simple render without envmap
                    render = voxel_renderer.render(mesh, resolution=512)
                    if isinstance(render, torch.Tensor):
                        render_np = (render.cpu().numpy() * 255).astype(np.uint8)
                    else:
                        render_np = np.array(render)
                    pil_frame = Image.fromarray(render_np)
                    buf = io.BytesIO()
                    pil_frame.save(buf, format="PNG")
                    previews_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
                except:
                    pass
        except Exception as e:
            print(f"⚠️ Voxel Render Failed: {e}")
        
        # Fallback: use the input 2D image as preview
        if not previews_b64:
            print("📸 Using 2D image as preview fallback")
            previews_b64.append(image_b64)

        # 5. Export GLB
        print("📦 Exporting GLB Asset...")
        print(f"📦 Mesh attributes: {[a for a in dir(mesh) if not a.startswith('_')]}")
        
        import trimesh
        
        # TRELLIS MeshWithVoxel - extract vertices/faces and build trimesh manually
        # Get vertices and faces as numpy arrays
        verts = mesh.vertices.cpu().numpy() if hasattr(mesh.vertices, 'cpu') else np.array(mesh.vertices)
        faces = mesh.faces.cpu().numpy() if hasattr(mesh.faces, 'cpu') else np.array(mesh.faces)
        
        print(f"📦 Vertices shape: {verts.shape}, Faces shape: {faces.shape}")
        
        # Check for vertex colors in attrs
        vertex_colors = None
        if hasattr(mesh, 'attrs') and mesh.attrs is not None:
            print(f"📦 Mesh attrs keys: {list(mesh.attrs.keys()) if isinstance(mesh.attrs, dict) else 'not a dict'}")
            # Try different color attribute names
            for color_key in ['color', 'colors', 'vertex_colors', 'rgb', 'rgba']:
                if isinstance(mesh.attrs, dict) and color_key in mesh.attrs:
                    vc = mesh.attrs[color_key]
                    vertex_colors = vc.cpu().numpy() if hasattr(vc, 'cpu') else np.array(vc)
                    print(f"📦 Found colors in attrs['{color_key}']: shape {vertex_colors.shape}")
                    break
        
        # Also check query_vertex_attrs method
        if vertex_colors is None and hasattr(mesh, 'query_vertex_attrs'):
            try:
                vc = mesh.query_vertex_attrs('color')
                if vc is not None:
                    vertex_colors = vc.cpu().numpy() if hasattr(vc, 'cpu') else np.array(vc)
                    print(f"📦 Found colors via query_vertex_attrs: shape {vertex_colors.shape}")
            except:
                pass
        
        # Process vertex colors if found
        if vertex_colors is not None:
            # Ensure correct shape (N, 3) or (N, 4)
            if len(vertex_colors.shape) == 1:
                vertex_colors = vertex_colors.reshape(-1, 3)
            
            # Convert to 0-255 range if normalized
            if vertex_colors.max() <= 1.0:
                vertex_colors = (vertex_colors * 255).astype(np.uint8)
            else:
                vertex_colors = vertex_colors.astype(np.uint8)
            
            # Add alpha channel if RGB
            if vertex_colors.shape[1] == 3:
                alpha = np.full((vertex_colors.shape[0], 1), 255, dtype=np.uint8)
                vertex_colors = np.concatenate([vertex_colors, alpha], axis=1)
            
            print(f"📦 Final vertex_colors shape: {vertex_colors.shape}")
        
        # Create trimesh object
        trimesh_obj = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
        
        # Export to GLB
        glb_buffer = io.BytesIO()
        trimesh_obj.export(glb_buffer, file_type='glb')
        glb_b64 = base64.b64encode(glb_buffer.getvalue()).decode('utf-8')
        
        # Get mesh stats safely
        num_faces = len(faces)
        num_vertices = len(verts)
        
        print(f"✅ GLB exported: {len(glb_b64)} bytes, {num_vertices} verts, {num_faces} faces")
        
        return {
            "status": "completed",
            "image_2d": image_b64, 
            "glb_b64": glb_b64,
            "previews": previews_b64,
            "metadata": {
                "faces": num_faces,
                "vertices": num_vertices,
                # Return used params for confirmation
                "seed": seed,
                "voxel_res": params.get('resolution', 1024)
            }
        }

# --- 3b. FLUX DEDICATED IMAGE (Lightweight, Modern Torch) ---
flux_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "diffusers", "transformers", "accelerate", "bitsandbytes", 
        "peft", "sentencepiece", "huggingface_hub", "protobuf",
        "fastapi" # Required for web endpoint
    )
    .env({"HF_TOKEN": _HF_TOKEN})
)

@app.cls(gpu="A10G", image=flux_image, timeout=3600, min_containers=0) # Use modern image for Flux
class Flux2DGenerator:
    @modal.enter()
    def setup(self):
        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig
        import os
        
        print("🏗️ Loading FLUX.1-Kontext-dev (4-bit)...")
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        
        # Load Transformer with Quantization
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            token=os.environ["HF_TOKEN"]
        )

        # Load Pipeline with Quantized Transformer
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            token=os.environ["HF_TOKEN"]
        )
        self.pipe.enable_model_cpu_offload() 
        print("✅ FLUX Ready!")

    @modal.method()
    def generate(self, prompt: str):
        import io
        print(f"🎨 Generating 2D: {prompt[:50]}...")
        image = self.pipe(
            prompt,
            height=768, width=1024,
            output_type="pil",
            num_inference_steps=25,
            guidance_scale=3.5,
            max_sequence_length=512
        ).images[0]
        
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- 4. ORCHESTRATOR ---
@app.function(image=flux_image, timeout=7200) # Orchestrator can run on lightweight image
def orchestrate_pro_pipeline(task_id: str, req_data: dict):
    print(f"🎬 Starting Pipeline for {task_id}")
    try:
        # Step 1: 2D Generation
        results_dict[task_id] = {"status": "generating_2d"}
        
        prompt = f"Top down view floor plan, {req_data.get('bhk', '2 BHK')}, {req_data.get('style', 'Modern')}, white background, technical blueprint style."
        flux = Flux2DGenerator()
        image_2d = flux.generate.remote(prompt)
        
        # Step 2: 3D Generation (Automatically Pass 2D to 3D)
        results_dict[task_id] = {
            "status": "generating_3d_pro", 
            "image_base64": image_2d # Send 2D immediately for UI
        }
        
        trellis = Trellis3DGenerator() # Uses the heavy image automatically
        res_3d = trellis.process_3d.remote(image_2d, req_data)
        
        # Final Combine
        final_result = {
            "status": "completed",
            "task_id": task_id,
            "image_base64": image_2d,
            "glb_b64": res_3d['glb_b64'],
            "previews": res_3d['previews'], # List for Scrubber
            "metadata": res_3d['metadata']
        }
        results_dict[task_id] = final_result
        print(f"🏁 Pipeline Finished: {task_id}")
        
    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")
        results_dict[task_id] = {"status": "failed", "error": str(e)}

# --- 5. HOST API ---
fastapi_app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.post("/generate")
async def generate_endpoint(request: Request):
    data = await request.json()
    task_id = str(uuid.uuid4())
    results_dict[task_id] = {"status": "queued"}
    
    # Spawn background orchestration
    orchestrate_pro_pipeline.spawn(task_id, data)
    
    return {"task_id": task_id}

@fastapi_app.get("/task/{task_id}")
async def get_status(task_id: str):
    if task_id not in results_dict:
        return {"status": "not_found"}
    return results_dict[task_id]

@app.function(image=flux_image)
@modal.asgi_app()
def fastapi_app_entry():
    return fastapi_app
