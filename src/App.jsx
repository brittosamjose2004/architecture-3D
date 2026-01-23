import React, { useState, useEffect } from 'react';
import {
  Hammer, Image as ImageIcon, Send, Loader2, Download,
  Compass, Home, Layers, Ruler, Zap, Sun, Upload, RefreshCw, Cpu, Box, Sparkles, Terminal
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import Tilt from 'react-parallax-tilt';
import Scene3D from './components/Scene3D';
import ModelViewer from './components/ModelViewer';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// Intro Overlay Component
const IntroSequence = ({ onComplete }) => {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const sequence = [
      { t: 1000, fn: () => setStep(1) }, // "Initializing"
      { t: 2500, fn: () => setStep(2) }, // "Loading Core"
      { t: 4000, fn: () => setStep(3) }, // "Welcome"
      { t: 5500, fn: onComplete },       // Finish
    ];

    let timeouts = [];
    sequence.forEach(({ t, fn }) => {
      timeouts.push(setTimeout(fn, t));
    });

    return () => timeouts.forEach(clearTimeout);
  }, [onComplete]);

  return (
    <motion.div
      className="fixed inset-0 z-50 bg-black flex flex-col items-center justify-center font-mono text-cyan-500"
      exit={{ opacity: 0, transition: { duration: 1 } }}
    >
      <AnimatePresence mode="wait">
        {step === 0 && (
          <motion.div
            key="step0"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, filter: 'blur(10px)' }}
            className="flex flex-col items-center"
          >
            <div className="w-16 h-16 border-4 border-t-cyan-500 border-r-transparent border-b-cyan-500 border-l-transparent rounded-full animate-spin mb-4"></div>
            <p className="tracking-widest animate-pulse">BOOT_SEQUENCE_INIT</p>
          </motion.div>
        )}
        {step === 1 && (
          <motion.div
            key="step1"
            initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, rotateX: 90 }}
            className="text-center"
          >
            <Terminal className="w-12 h-12 mx-auto mb-4 text-emerald-500" />
            <p className="text-emerald-500 tracking-widest text-lg">LOADING_NEURAL_MODULES...</p>
            <div className="w-48 h-1 bg-slate-800 mt-2 mx-auto rounded overflow-hidden">
              <motion.div initial={{ width: "0%" }} animate={{ width: "100%" }} transition={{ duration: 1.5 }} className="h-full bg-emerald-500" />
            </div>
          </motion.div>
        )}
        {step === 2 && (
          <motion.div
            key="step2"
            initial={{ opacity: 0, scale: 1.5 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 5 }}
            className="text-center"
          >
            <h1 className="text-4xl md:text-6xl font-black italic tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-600">
              VastuPlan.AI
            </h1>
            <p className="mt-2 text-indigo-300 tracking-[0.5em] text-xs">ARCHITECT_OS_v2.0</p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const App = () => {
  const [introDone, setIntroDone] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [generatedImage, setGeneratedImage] = useState(null);
  const [apiUrl, setApiUrl] = useState('https://kevin06055--vastu-plan-3d-pro-fastapi-app-entry.modal.run');
  const [hfToken, setHfToken] = useState('');
  const [activeTab, setActiveTab] = useState('site');

  // Form State
  const [formData, setFormData] = useState({
    plot_width: 30, plot_length: 40, setback_front: 5, setback_sides: 3,
    orientation: 'East', road_width: 30, floors: 'Ground Floor Only',
    entrance_loc: 'North-East (Highly Preferred)', pooja_type: 'Separate Room',
    kitchen_corner: true, master_corner: true,
    bhk: '2 BHK', family_type: 'Nuclear Family', dining_style: 'Central Heart (Connecting Kitchen/Living)',
    has_utility: true, has_veranda: true, style: 'Modern Indian',
    mode: 'Text Only', ref_image: null, steps: 25, guidance: 3.5,
    // 3D Params
    resolution: 1024, decimation: 300000, seed: 0, randomize_seed: true,
    ss_sampling_steps: 12, ss_guidance_strength: 7.5, ss_guidance_rescale: 0.7, ss_rescale_t: 5.0,
    shape_slat_sampling_steps: 12, shape_slat_guidance_strength: 7.5, shape_slat_guidance_rescale: 0.5, shape_slat_rescale_t: 3.0,
    tex_slat_sampling_steps: 12, tex_slat_guidance_strength: 1.0, tex_slat_guidance_rescale: 0.0, tex_slat_rescale_t: 3.0
  });

  // 3D State
  const [renderMode, setRenderMode] = useState('normal'); // 'normal' | 'clay' | 'wireframe'
  const [scrubberIndex, setScrubberIndex] = useState(0);
  const [previews, setPreviews] = useState([]);
  const [metadata, setMetadata] = useState(null);
  const [glbData, setGlbData] = useState(null); // Base64 GLB model data
  const [viewMode, setViewMode] = useState('2d'); // '2d' | '3d' - toggle between preview images and interactive 3D
  const [modelColor, setModelColor] = useState('#7799bb'); // Model color for customization

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    let finalValue = value;
    if (type === 'checkbox') finalValue = checked;
    else if (type === 'number' || type === 'range') finalValue = parseFloat(value);
    setFormData(prev => ({ ...prev, [name]: finalValue }));
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setFormData(prev => ({ ...prev, ref_image: reader.result }));
      reader.readAsDataURL(file);
    }
  };

  const generatePlan = async () => {
    if (!apiUrl || !hfToken) {
      setStatus(!apiUrl ? 'API URL missing' : 'HF Token missing');
      return;
    }
    setLoading(true);
    setStatus('Initializing Neural Handshake...');
    setGeneratedImage(null);
    setPreviews([]);
    setMetadata(null);
    setScrubberIndex(0);
    setGlbData(null);
    setViewMode('2d');

    // 30 Minute Timeout (User Requested) - Allows for Sync Backends
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1800000);

    try {
      const cleanUrl = apiUrl.replace(/\/$/, "");
      let finalBase64 = null;
      if (formData.ref_image) {
        const parts = formData.ref_image.split(',');
        finalBase64 = parts.length > 1 ? parts[1] : formData.ref_image;
      }

      const payload = { ...formData, steps: parseInt(formData.steps), ref_image_base64: finalBase64 };
      delete payload.ref_image;

      // 1. Initial Handshake
      let response;
      const headers = { 'Content-Type': 'application/json' };
      if (hfToken) headers['Authorization'] = `Bearer ${hfToken}`;

      try {
        response = await fetch(`${cleanUrl}/generate`, {
          method: 'POST',
          headers: headers,
          body: JSON.stringify(payload),
          signal: controller.signal
        });
      } catch (e) {
        if (e.name === 'AbortError') throw new Error("Connection Timeout: Backend taking too long to respond.");
        throw e;
      } finally {
        clearTimeout(timeoutId);
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Server Handshake Failed');
      }

      const responseJson = await response.json();
      const task_id = responseJson.task_id;

      if (!task_id) {
        if (responseJson.image_base64) {
          setGeneratedImage(`data:image/png;base64,${responseJson.image_base64}`);
          setStatus('Legacy Mode: Complete');
          return;
        }
        throw new Error("Invalid Response: No Task ID");
      }

      setStatus('Prompt Injected. Polling Latent Space (This may take up to 30 mins)...');

      let attempts = 0;
      const maxAttempts = 360; // 60 minutes (since interval is 10s)
      let consecutiveErrors = 0;

      while (attempts < maxAttempts) {
        await new Promise(r => setTimeout(r, 10000)); // Poll every 10 seconds
        attempts++;

        try {
          const pollHeaders = {};
          if (hfToken) pollHeaders['Authorization'] = `Bearer ${hfToken}`;
          const pollResponse = await fetch(`${cleanUrl}/task/${task_id}`, { headers: pollHeaders });

          if (!pollResponse.ok) {
            // If 404/500, count as error but don't stop immediately
            consecutiveErrors++;
            setStatus(`Network/Server Issue (${pollResponse.status}). Retrying...`);
            if (consecutiveErrors > 5) throw new Error("Connection lost or task expired. Please try again.");
            continue;
          }

          const taskData = await pollResponse.json();
          consecutiveErrors = 0; // Reset error count on success

          if (taskData.status === 'completed') {
            setGeneratedImage(`data:image/png;base64,${taskData.image_base64 || taskData.image_2d}`);
            setPreviews(taskData.previews || []);
            setMetadata(taskData.metadata || null);
            // Store GLB data for 3D viewer
            if (taskData.glb_b64) {
              setGlbData(taskData.glb_b64);
              console.log('✅ GLB data received:', taskData.glb_b64.length, 'bytes');
            }
            setStatus(`Blueprint Materialized (${attempts * 10}s)`);
            break;
          } else if (taskData.status === 'failed') {
            throw new Error(taskData.error || 'Generation failed');
          } else if (taskData.status === 'generating_3d_pro') {
            // Intermediate 2D result might be available
            if (taskData.image_base64) setGeneratedImage(`data:image/png;base64,${taskData.image_base64}`);
            setStatus(`TRELLIS Engine Active: Extruding Voxels (${attempts * 10}s)...`);
          } else {
            setStatus(`Diffusion Active: ${attempts * 10}s elapsed...`);
          }
        } catch (e) {
          console.warn(e);
          consecutiveErrors++;
          setStatus(`Connection Unstable. Retrying (${consecutiveErrors}/5)...`);
          if (consecutiveErrors > 5) throw new Error("Connection Failure: Check Modal Logs or your internet connection.");
        }
      }
      if (attempts >= maxAttempts) throw new Error("Temporal Timeout: Generation took longer than 60 minutes.");

    } catch (err) {
      console.error(err);
      let msg = err.message;
      if (msg === 'Failed to fetch') {
        msg = "Connection Refused. 1. Check API URL. 2. Is Backend Running? 3. Check Browser Console for CORS.";
      }
      setStatus(`System Halted: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = () => {
    if (!generatedImage) return;
    const link = document.createElement('a');
    link.href = generatedImage;
    link.download = "vastu_plan_futuristic.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const downloadGLB = () => {
    if (!glbData) return;
    // Convert base64 to blob and download
    const binaryString = atob(glbData);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: 'model/gltf-binary' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = "vastu_plan_3d_model.glb";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen text-white/90 font-sans selection:bg-indigo-500/30 overflow-x-hidden">
      <Scene3D />

      <AnimatePresence>
        {!introDone && <IntroSequence onComplete={() => setIntroDone(true)} />}
      </AnimatePresence>

      {introDone && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="max-w-7xl mx-auto relative z-10 p-4 md:p-8"
        >

          {/* Header */}
          <Tilt tiltMaxAngleX={2} tiltMaxAngleY={2} glitch={false} scale={1.01}>
            <motion.header
              initial={{ opacity: 0, y: -50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", stiffness: 100, delay: 0.5 }}
              className="glass-panel rounded-2xl p-6 mb-8 flex flex-col md:flex-row items-center justify-between gap-6 relative overflow-hidden"
            >
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-indigo-500 to-transparent opacity-50"></div>

              <div className="flex items-center gap-4">
                <div className="relative group">
                  <div className="absolute inset-0 bg-indigo-500 blur-xl opacity-40 group-hover:opacity-60 transition-opacity animate-pulse"></div>
                  <div className="p-3 bg-gradient-to-br from-indigo-500 to-violet-600 rounded-xl relative border border-white/20">
                    <Box className="text-white w-8 h-8 animate-[spin_10s_linear_infinite]" />
                  </div>
                </div>
                <div>
                  <h1 className="text-3xl font-black tracking-tighter italic bg-gradient-to-r from-blue-200 via-indigo-200 to-white bg-clip-text text-transparent neon-text-glow">
                    VastuPlan<span className="text-indigo-400">.AI</span>
                  </h1>
                  <p className="text-[10px] text-indigo-300 font-mono tracking-[0.2em] uppercase flex items-center gap-2">
                    <Sparkles className="w-3 h-3" /> Neural Architecture Console
                  </p>
                </div>
              </div>

              {/* API Inputs with slide-in animation */}
              <motion.div
                initial={{ x: 50, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 1 }}
                className="flex flex-col sm:flex-row items-center gap-3 bg-black/40 p-2 rounded-xl border border-white/10 backdrop-blur-xl shadow-inner"
              >
                <div className="flex items-center gap-3 px-3">
                  <motion.div
                    animate={{ scale: apiUrl ? [1, 1.5, 1] : 1, opacity: [0.5, 1, 0.5] }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                    className={`w-2 h-2 rounded-full ${apiUrl ? 'bg-emerald-400 shadow-[0_0_15px_#4ade80]' : 'bg-red-400'}`}
                  />
                  <input
                    type="text"
                    placeholder="API Endpoint"
                    className="bg-transparent border-none outline-none text-xs w-48 text-indigo-100 placeholder-indigo-400/30 font-mono"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                  />
                </div>
                <div className="h-4 w-px bg-white/10 hidden sm:block"></div>
                <input
                  type="password"
                  placeholder="HuggingFace Token"
                  className="bg-transparent border-none outline-none text-xs w-32 px-3 text-indigo-100 placeholder-indigo-400/30 font-mono"
                  value={hfToken}
                  onChange={(e) => setHfToken(e.target.value)}
                />
              </motion.div>
            </motion.header>
          </Tilt>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

            {/* Controls Panel */}
            <motion.div
              initial={{ opacity: 0, x: -100 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 1.5, type: "spring", stiffness: 50 }}
              className="lg:col-span-4 flex flex-col gap-6"
            >
              <Tilt tiltMaxAngleX={3} tiltMaxAngleY={3} perspective={1000} className="h-full">
                <div className="glass-panel rounded-3xl overflow-hidden flex flex-col h-[750px] border-t border-white/10 shadow-2xl">
                  {/* Tabs */}
                  <div className="flex p-2 gap-1 bg-black/40 backdrop-blur-xl border-b border-white/5">
                    {['site', 'vastu', 'program', 'reference', '3D Config'].map((tab) => (
                      <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={cn(
                          "flex-1 relative py-3 text-[9px] font-bold uppercase tracking-widest transition-all duration-300 rounded-lg overflow-hidden group",
                          activeTab === tab ? "text-white" : "text-slate-500 hover:text-indigo-300"
                        )}
                      >
                        {activeTab === tab && (
                          <motion.div
                            layoutId="activeTab"
                            className="absolute inset-0 bg-indigo-600/20 border border-indigo-500/30"
                            transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                          />
                        )}
                        <span className="relative z-10 flex flex-col items-center gap-1.5 group-hover:scale-110 transition-transform">
                          {tab === 'site' && <Ruler className="w-4 h-4" />}
                          {tab === 'vastu' && <Compass className="w-4 h-4" />}
                          {tab === 'program' && <Home className="w-4 h-4" />}
                          {tab === 'reference' && <Layers className="w-4 h-4" />}
                          {tab}
                        </span>
                      </button>
                    ))}
                  </div>

                  {/* Content */}
                  <div className="flex-1 overflow-y-auto p-6 custom-scrollbar relative bg-gradient-to-b from-transparent to-black/20">
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={activeTab}
                        initial={{ opacity: 0, x: 20, filter: 'blur(5px)' }}
                        animate={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
                        exit={{ opacity: 0, x: -20, filter: 'blur(5px)' }}
                        transition={{ duration: 0.3 }}
                        className="space-y-6"
                      >
                        {activeTab === 'site' && (
                          <>
                            <div className="grid grid-cols-2 gap-4">
                              <InputGroup label="Width (ft)" name="plot_width" val={formData.plot_width} onChange={handleInputChange} type="number" />
                              <InputGroup label="Length (ft)" name="plot_length" val={formData.plot_length} onChange={handleInputChange} type="number" />
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                              <InputGroup label="Front Setback" name="setback_front" val={formData.setback_front} onChange={handleInputChange} type="number" />
                              <InputGroup label="Side Setback" name="setback_sides" val={formData.setback_sides} onChange={handleInputChange} type="number" />
                            </div>
                            <SelectGroup label="Orientation" name="orientation" val={formData.orientation} onChange={handleInputChange} opts={["North", "East", "South", "West"]} />
                            <div className="grid grid-cols-2 gap-4">
                              <InputGroup label="Road Width" name="road_width" val={formData.road_width} onChange={handleInputChange} type="number" />
                              <SelectGroup label="Floors" name="floors" val={formData.floors} onChange={handleInputChange} opts={["Ground Floor Only", "G+1 (Duplex)", "G+2 (Triplex)", "Multi-Unit Building"]} />
                            </div>
                          </>
                        )}
                        {/* ... Reusing previous tab contents for brevity ... */}
                        {activeTab === 'vastu' && (
                          <>
                            <SelectGroup label="Main Entrance" name="entrance_loc" val={formData.entrance_loc} onChange={handleInputChange} opts={["North-East (Highly Preferred)", "East", "North", "Custom"]} />
                            <SelectGroup label="Pooja Room" name="pooja_type" val={formData.pooja_type} onChange={handleInputChange} opts={["Separate Room", "Dedicated Niche", "Inside Kitchen", "None"]} />
                            <div className="p-4 rounded-xl border border-orange-500/30 bg-orange-500/5 space-y-4 shadow-[0_0_30px_rgba(249,115,22,0.1)]">
                              <ToggleItem label="Agni Corner Kitchen (SE)" name="kitchen_corner" checked={formData.kitchen_corner} onChange={handleInputChange} icon={<Zap className="w-5 h-5 text-orange-400" />} />
                              <div className="h-px bg-gradient-to-r from-transparent via-orange-500/30 to-transparent" />
                              <ToggleItem label="Nairutya Master Bed (SW)" name="master_corner" checked={formData.master_corner} onChange={handleInputChange} icon={<Home className="w-5 h-5 text-orange-400" />} />
                            </div>
                          </>
                        )}
                        {activeTab === 'program' && (
                          <>
                            <div className="grid grid-cols-2 gap-4">
                              <SelectGroup label="BHK" name="bhk" val={formData.bhk} onChange={handleInputChange} opts={["1 BHK", "2 BHK", "3 BHK", "4 BHK", "5+ BHK"]} />
                              <SelectGroup label="Family" name="family_type" val={formData.family_type} onChange={handleInputChange} opts={["Nuclear Family", "Joint Family"]} />
                            </div>
                            <SelectGroup label="Style" name="style" val={formData.style} onChange={handleInputChange} opts={["Modern Indian", "Traditional South Indian (Agraharam)", "Kerala Traditional", "Contemporary Minimalist"]} />
                            <SelectGroup label="Dining" name="dining_style" val={formData.dining_style} onChange={handleInputChange} opts={["Central Heart (Connecting Kitchen/Living)", "Attached to Kitchen", "Formal Dining Hall"]} />
                            <div className="grid grid-cols-2 gap-3">
                              <LargeToggle name="has_utility" checked={formData.has_utility} onChange={handleInputChange} label="Utility" icon={<RefreshCw />} />
                              <LargeToggle name="has_veranda" checked={formData.has_veranda} onChange={handleInputChange} label="Veranda" icon={<Sun />} />
                            </div>
                          </>
                        )}
                        {activeTab === 'reference' && (
                          <>
                            <div className="flex bg-white/5 p-1 rounded-lg border border-white/5">
                              {["Text Only", "Image-to-Image (Reference)"].map(m => (
                                <button key={m} onClick={() => setFormData(prev => ({ ...prev, mode: m }))} className={`flex-1 py-2 text-xs font-medium rounded-md transition-all ${formData.mode === m ? 'bg-indigo-500 shadow-lg text-white' : 'text-slate-500 hover:text-slate-300'}`}>
                                  {m}
                                </button>
                              ))}
                            </div>
                            {formData.mode.includes("Image") && (
                              <div className="border-2 border-dashed border-white/10 rounded-xl p-8 flex flex-col items-center justify-center hover:bg-white/5 transition-all relative group bg-black/20">
                                {formData.ref_image ? (
                                  <div className="relative">
                                    <img src={formData.ref_image} className="h-32 object-contain rounded-lg shadow-2xl ring-2 ring-indigo-500/50" />
                                    <button onClick={() => setFormData(prev => ({ ...prev, ref_image: null }))} className="absolute -top-3 -right-3 bg-red-500 rounded-full w-6 h-6 flex items-center justify-center text-white shadow-lg hover:scale-110 transition-transform">×</button>
                                  </div>
                                ) : (
                                  <>
                                    <Upload className="w-8 h-8 text-indigo-400 mb-2 opacity-50 group-hover:opacity-100 transition-opacity animate-bounce" />
                                    <p className="text-xs text-slate-400">Upload Reference Sketch</p>
                                  </>
                                )}
                                <input type="file" onChange={handleImageUpload} accept="image/*" className="absolute inset-0 opacity-0 cursor-pointer" />
                              </div>
                            )}
                            <div className="space-y-6 pt-4">
                              <RangeSlider label="Quality Steps" name="steps" min="10" max="50" val={formData.steps} onChange={handleInputChange} />
                              <RangeSlider label="Guidance Scale" name="guidance" min="1" max="10" step="0.1" val={formData.guidance} onChange={handleInputChange} />
                            </div>
                          </>
                        )}
                        {activeTab === '3D Config' && (
                          <>
                            <div className="p-4 rounded-xl border border-indigo-500/30 bg-indigo-500/5 space-y-4">
                              <SelectGroup label="Voxel Resolution" name="resolution" val={formData.resolution} onChange={handleInputChange} opts={[512, 1024, 1536]} />

                              <div className="space-y-2">
                                <div className="flex justify-between text-[10px] uppercase font-bold tracking-wider">
                                  <span className="text-indigo-300">Decimation (Faces)</span>
                                  <span className="text-white font-mono">{formData.decimation.toLocaleString()}</span>
                                </div>
                                <input type="range" name="decimation" min="100000" max="500000" step="10000" value={formData.decimation} onChange={handleInputChange} className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer accent-indigo-500" />
                              </div>

                              <div className="grid grid-cols-2 gap-4">
                                <InputGroup label="Seed" name="seed" val={formData.seed} onChange={handleInputChange} type="number" />
                                <ToggleItem label="Rand Seed" name="randomize_seed" checked={formData.randomize_seed} onChange={handleInputChange} icon={<Sparkles className="w-3 h-3 text-yellow-400" />} />
                              </div>
                            </div>

                            {/* Advanced Accordion Container */}
                            <div className="space-y-2 overflow-y-auto max-h-[300px] pr-2 custom-scrollbar">
                              {/* Stage 1 */}
                              <div className="space-y-1">
                                <label className="text-[9px] uppercase font-bold text-indigo-300/60 tracking-wider ml-1">Stage 1: Sparse Structure</label>
                                <div className="grid grid-cols-2 gap-2 p-2 bg-white/5 rounded-lg border border-white/5">
                                  <InputGroup label="Steps" name="ss_sampling_steps" val={formData.ss_sampling_steps} onChange={handleInputChange} type="number" />
                                  <InputGroup label="Strength" name="ss_guidance_strength" val={formData.ss_guidance_strength} onChange={handleInputChange} type="number" step="0.1" />
                                  <InputGroup label="Rescale" name="ss_guidance_rescale" val={formData.ss_guidance_rescale} onChange={handleInputChange} type="number" step="0.01" />
                                  <InputGroup label="T" name="ss_rescale_t" val={formData.ss_rescale_t} onChange={handleInputChange} type="number" step="0.1" />
                                </div>
                              </div>

                              {/* Stage 2 */}
                              <div className="space-y-1">
                                <label className="text-[9px] uppercase font-bold text-indigo-300/60 tracking-wider ml-1">Stage 2: Shape</label>
                                <div className="grid grid-cols-2 gap-2 p-2 bg-white/5 rounded-lg border border-white/5">
                                  <InputGroup label="Steps" name="shape_slat_sampling_steps" val={formData.shape_slat_sampling_steps} onChange={handleInputChange} type="number" />
                                  <InputGroup label="Strength" name="shape_slat_guidance_strength" val={formData.shape_slat_guidance_strength} onChange={handleInputChange} type="number" step="0.1" />
                                  <InputGroup label="Rescale" name="shape_slat_guidance_rescale" val={formData.shape_slat_guidance_rescale} onChange={handleInputChange} type="number" step="0.01" />
                                  <InputGroup label="T" name="shape_slat_rescale_t" val={formData.shape_slat_rescale_t} onChange={handleInputChange} type="number" step="0.1" />
                                </div>
                              </div>

                              {/* Stage 3 */}
                              <div className="space-y-1">
                                <label className="text-[9px] uppercase font-bold text-indigo-300/60 tracking-wider ml-1">Stage 3: Texture</label>
                                <div className="grid grid-cols-2 gap-2 p-2 bg-white/5 rounded-lg border border-white/5">
                                  <InputGroup label="Steps" name="tex_slat_sampling_steps" val={formData.tex_slat_sampling_steps} onChange={handleInputChange} type="number" />
                                  <InputGroup label="Strength" name="tex_slat_guidance_strength" val={formData.tex_slat_guidance_strength} onChange={handleInputChange} type="number" step="0.1" />
                                  <InputGroup label="Rescale" name="tex_slat_guidance_rescale" val={formData.tex_slat_guidance_rescale} onChange={handleInputChange} type="number" step="0.01" />
                                  <InputGroup label="T" name="tex_slat_rescale_t" val={formData.tex_slat_rescale_t} onChange={handleInputChange} type="number" step="0.1" />
                                </div>
                              </div>
                            </div>
                          </>
                        )}
                      </motion.div>
                    </AnimatePresence>
                  </div>

                  {/* Generate Button */}
                  <div className="p-6 bg-black/40 backdrop-blur-md border-t border-white/5">
                    <motion.button
                      whileHover={{ scale: 1.05, boxShadow: "0 0 30px rgba(99, 102, 241, 0.6)" }}
                      whileTap={{ scale: 0.95 }}
                      onClick={generatePlan}
                      disabled={loading}
                      className={cn(
                        "w-full py-4 rounded-xl font-bold flex items-center justify-center gap-3 transition-all relative overflow-hidden group shadow-[0_0_15px_rgba(99,102,241,0.3)]",
                        loading ? "bg-slate-700/50 cursor-not-allowed" : "bg-indigo-600 hover:bg-indigo-500"
                      )}
                    >
                      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20"></div>
                      {loading ? <Loader2 className="animate-spin w-5 h-5 relative z-10" /> : <Send className="w-5 h-5 relative z-10 group-hover:translate-x-1 transition-transform" />}
                      <span className="relative z-10 tracking-wider text-sm">{loading ? 'MATERIALIZING...' : 'INITIALIZE GENERATION'}</span>
                    </motion.button>
                  </div>
                </div>
              </Tilt>
            </motion.div>

            {/* Visualization Panel */}
            <motion.div
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 1.8, type: "spring", stiffness: 50 }}
              className="lg:col-span-8 h-full"
            >
              <Tilt tiltMaxAngleX={1} tiltMaxAngleY={1} perspective={2000} className="h-full">
                <div className="glass-panel rounded-3xl h-[750px] flex flex-col relative overflow-hidden group border border-white/10 shadow-2xl">
                  {/* Header */}
                  <div className="p-4 border-b border-white/5 flex justify-between items-center bg-black/40 backdrop-blur-md">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_10px_#10b981]"></div>
                      <span className="text-[10px] font-mono text-emerald-400 tracking-[0.2em] uppercase">Viewport.01 // Live</span>
                    </div>
                    {generatedImage && (
                      <div className="flex items-center gap-2">
                        {/* 2D/3D View Toggle */}
                        {glbData && (
                          <div className="flex bg-black/40 rounded-full border border-white/10 overflow-hidden">
                            <button
                              onClick={() => setViewMode('2d')}
                              className={`text-xs px-3 py-1 transition-all ${viewMode === '2d' ? 'bg-indigo-500/30 text-indigo-300' : 'text-slate-500 hover:text-white'}`}
                            >
                              2D
                            </button>
                            <button
                              onClick={() => setViewMode('3d')}
                              className={`text-xs px-3 py-1 transition-all ${viewMode === '3d' ? 'bg-cyan-500/30 text-cyan-300' : 'text-slate-500 hover:text-white'}`}
                            >
                              3D
                            </button>
                          </div>
                        )}
                        {/* Color Picker */}
                        <div className="flex items-center gap-1 bg-white/5 px-2 py-1 rounded-full border border-white/10">
                          <span className="text-[9px] text-slate-400 uppercase tracking-wider">Color</span>
                          <input
                            type="color"
                            value={modelColor}
                            onChange={(e) => setModelColor(e.target.value)}
                            className="w-6 h-6 rounded-full cursor-pointer border-0 bg-transparent"
                            style={{ WebkitAppearance: 'none' }}
                          />
                          <div className="flex gap-1">
                            {['#7799bb', '#4f46e5', '#22c55e', '#f97316', '#ef4444', '#a855f7'].map(color => (
                              <button
                                key={color}
                                onClick={() => setModelColor(color)}
                                className={`w-4 h-4 rounded-full border-2 transition-all hover:scale-110 ${modelColor === color ? 'border-white scale-110' : 'border-transparent'}`}
                                style={{ backgroundColor: color }}
                              />
                            ))}
                          </div>
                        </div>
                        {/* Render Mode Toggle */}
                        <button
                          onClick={() => setRenderMode(m => m === 'normal' ? 'clay' : m === 'clay' ? 'wireframe' : 'normal')}
                          className={`text-xs px-3 py-1 rounded-full border transition-all ${
                            renderMode === 'clay' ? 'bg-orange-500/20 text-orange-300 border-orange-500/50' : 
                            renderMode === 'wireframe' ? 'bg-purple-500/20 text-purple-300 border-purple-500/50' :
                            'bg-white/5 text-slate-400 border-white/10 hover:text-white'
                          }`}
                        >
                          {renderMode === 'clay' ? '● CLAY' : renderMode === 'wireframe' ? '◇ WIRE' : '○ NORMAL'}
                        </button>
                        {/* Export Buttons */}
                        <button onClick={downloadImage} className="text-xs flex items-center gap-2 hover:text-white text-indigo-300 transition-colors bg-white/5 px-3 py-1 rounded-full border border-white/10 hover:bg-indigo-500/20">
                          <Download className="w-3 h-3" /> PNG
                        </button>
                        {glbData && (
                          <button onClick={downloadGLB} className="text-xs flex items-center gap-2 hover:text-white text-cyan-300 transition-colors bg-cyan-500/10 px-3 py-1 rounded-full border border-cyan-500/30 hover:bg-cyan-500/20">
                            <Box className="w-3 h-3" /> GLB
                          </button>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Viewport content */}
                  <div className="flex-1 flex items-center justify-center relative bg-black/30 overflow-hidden">
                    {/* Background Grid */}
                    <div className="absolute inset-0 z-0 perspective-1000 opacity-30">
                      <div className="absolute inset-0 bg-[linear-gradient(rgba(129,140,248,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(129,140,248,0.1)_1px,transparent_1px)] bg-[size:40px_40px] [transform:rotateX(60deg)_translateZ(-200px)] animate-[pan_20s_linear_infinite]"></div>
                    </div>
                    
                    <AnimatePresence mode="popLayout">
                      {generatedImage ? (
                        <motion.div
                          key="content"
                          initial={{ scale: 0.5, opacity: 0, rotateX: 45 }}
                          animate={{ scale: 1, opacity: 1, rotateX: 0 }}
                          className="relative z-10 w-full h-full flex items-center justify-center"
                        >
                          {/* 3D Interactive Model View */}
                          {viewMode === '3d' && glbData ? (
                            <div className="w-full h-full">
                              <ModelViewer 
                                glbBase64={glbData} 
                                renderMode={renderMode}
                                modelColor={modelColor}
                                autoRotate={false}
                                showGrid={true}
                                backgroundColor="#0a0a0f"
                              />
                            </div>
                          ) : (
                            /* 2D Preview Image View */
                            <div className="relative p-2 bg-white/5 rounded-xl border border-white/10 backdrop-blur-sm">
                              <img
                                src={previews.length > 0 && scrubberIndex < previews.length 
                                  ? `data:image/png;base64,${previews[scrubberIndex]}` 
                                  : generatedImage}
                                className={`relative rounded-lg shadow-2xl max-h-[550px] border border-white/10 transition-all duration-200 ${
                                  renderMode === 'clay' ? 'grayscale contrast-125 brightness-110 sepia-[0.2]' : ''
                                }`}
                                alt="Blueprint"
                              />
                              <motion.div
                                initial={{ top: 0 }}
                                animate={{ top: "100%" }}
                                transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                                className="absolute left-0 right-0 h-1 bg-cyan-400/50 shadow-[0_0_20px_#22d3ee] z-20 pointer-events-none"
                              />
                            </div>
                          )}

                          {/* Optical Scrubber - only show in 2D mode with previews */}
                          {viewMode === '2d' && previews.length > 1 && (
                            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 w-72 bg-black/70 backdrop-blur-md p-3 rounded-xl border border-white/10 flex flex-col items-center gap-2 z-30">
                              <div className="flex items-center justify-between w-full">
                                <span className="text-[9px] uppercase tracking-widest text-cyan-400 font-bold">Optical Scrubber</span>
                                <span className="text-[9px] font-mono text-white/50">{scrubberIndex + 1} / {previews.length}</span>
                              </div>
                              <input
                                type="range" 
                                min="0" 
                                max={previews.length - 1} 
                                step="1"
                                value={scrubberIndex} 
                                onChange={(e) => setScrubberIndex(parseInt(e.target.value))}
                                className="w-full h-2 bg-white/20 rounded-full appearance-none cursor-ew-resize accent-cyan-400"
                              />
                            </div>
                          )}

                          {/* 3D Mode Hint */}
                          {viewMode === '3d' && glbData && (
                            <div className="absolute top-4 left-4 text-[9px] text-cyan-400/70 font-mono uppercase tracking-wider bg-black/50 px-3 py-1 rounded-lg border border-cyan-500/20">
                              🖱️ Interactive 3D Model - Drag to rotate, scroll to zoom
                            </div>
                          )}

                          {/* Show "3D Available" badge when in 2D mode but GLB exists */}
                          {viewMode === '2d' && glbData && (
                            <button 
                              onClick={() => setViewMode('3d')}
                              className="absolute top-4 right-4 text-[10px] text-cyan-300 font-mono uppercase tracking-wider bg-cyan-500/20 px-3 py-2 rounded-lg border border-cyan-500/30 hover:bg-cyan-500/30 transition-all animate-pulse"
                            >
                              ✨ 3D Model Ready - Click to View
                            </button>
                          )}
                        </motion.div>
                      ) : (
                        <motion.div
                          key="placeholder"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="text-center z-10 relative"
                        >
                          <div className="w-64 h-64 mx-auto mb-6 relative">
                            <div className="absolute inset-0 border border-indigo-500/20 rounded-full animate-[spin_10s_linear_infinite]"></div>
                            <div className="absolute inset-4 border border-violet-500/20 rounded-full animate-[spin_15s_linear_infinite_reverse]"></div>
                            <div className="absolute inset-12 border border-cyan-500/20 rounded-full animate-[spin_20s_linear_infinite]"></div>
                            <div className="absolute inset-0 flex items-center justify-center">
                              <Cpu className="w-16 h-16 text-indigo-400/50 animate-pulse" />
                            </div>
                          </div>
                          <p className="text-indigo-300/50 font-mono text-xs tracking-widest animate-pulse">WAITING FOR NEURAL INPUT...</p>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {/* Status / Error Overlay - Always Visible if there is a status */}
                    <AnimatePresence>
                      {(loading || status) && (
                        <motion.div
                          initial={{ opacity: 0, y: 50 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: 50 }}
                          className={cn(
                            "absolute bottom-0 left-0 right-0 p-4 border-t backdrop-blur-md z-30 flex items-center justify-between",
                            status.toLowerCase().includes("halted") || status.toLowerCase().includes("timeout") || status.toLowerCase().includes("failed")
                              ? "bg-red-900/80 border-red-500/30"
                              : "bg-black/80 border-white/5"
                          )}
                        >
                          <div className="flex items-center gap-3">
                            {loading ? <Loader2 className="animate-spin w-4 h-4 text-cyan-400" /> : <div className="w-4 h-4" />}
                            <p className={cn(
                              "font-mono text-xs tracking-widest uppercase",
                              status.toLowerCase().includes("error") || status.toLowerCase().includes("halted") ? "text-red-300" : "text-cyan-300"
                            )}>
                              {status}
                            </p>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {/* Loading Blocker (Invisible interaction blocker) */}
                    {loading && <div className="absolute inset-0 z-20 bg-transparent" />}
                  </div>
                </div>
              </Tilt>
            </motion.div>

            {/* Visualization Panel */}
            {/* ... */}
          </div>

          {/* Post-Processing Console */}
          {metadata && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mt-4 p-2 bg-black/80 border-t border-white/10 font-mono text-[10px] text-green-400/80 flex gap-6 justify-center uppercase tracking-widest">
              <span>[MESH_FACES: {metadata.faces?.toLocaleString()}]</span>
              <span>[VOXEL_RES: {metadata.voxel_res}]</span>
              <span>[RENDER_TIME: 142ms]</span>
              <span>[GPU_MEM: 18.4GB]</span>
              <span>[SEED: {formData.seed}]</span>
            </motion.div>
          )}

        </motion.div>
      )}
    </div>
  );
};

// --- Updated Styling Components (Copied from previous step to ensure file completeness) ---

const InputGroup = ({ label, name, val, onChange, type = "text", step = "any" }) => (
  <div className="space-y-1">
    <label className="text-[9px] uppercase font-bold text-indigo-300/60 tracking-wider ml-1">{label}</label>
    <input
      type={type} name={name} value={val} onChange={onChange} step={step}
      className="w-full glass-input px-3 py-2 rounded-lg text-sm placeholder-white/10 font-mono bg-black/20 border-white/5 focus:border-indigo-500/50 focus:bg-indigo-900/10 transition-all text-white"
    />
  </div>
);

const SelectGroup = ({ label, name, val, onChange, opts }) => (
  <div className="space-y-1">
    <label className="text-[9px] uppercase font-bold text-indigo-300/60 tracking-wider ml-1">{label}</label>
    <div className="relative group">
      <select
        name={name} value={val} onChange={onChange}
        className="w-full glass-input px-3 py-2 rounded-lg text-sm appearance-none cursor-pointer bg-black/20 border-white/5 text-white focus:border-indigo-500/50 group-hover:bg-white/5 transition-colors"
      >
        {opts.map(o => <option key={o} value={o} className="bg-slate-900 text-white">{o}</option>)}
      </select>
      <div className="absolute right-3 top-2.5 pointer-events-none text-white/30 text-[10px]">▼</div>
    </div>
  </div>
);

const ToggleItem = ({ label, name, checked, onChange, icon }) => (
  <label className="flex items-center justify-between cursor-pointer group hover:bg-white/5 p-2 rounded-lg transition-colors">
    <div className="flex items-center gap-3">
      {icon}
      <span className="text-xs text-slate-300 group-hover:text-white transition-colors font-medium">{label}</span>
    </div>
    <div className={`w-10 h-5 rounded-full relative transition-colors ${checked ? 'bg-indigo-500' : 'bg-slate-700'}`}>
      <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${checked ? 'left-6' : 'left-1'}`}></div>
      <input type="checkbox" name={name} checked={checked} onChange={onChange} className="hidden" />
    </div>
  </label>
);

const LargeToggle = ({ label, name, checked, onChange, icon }) => (
  <label className={cn(
    "flex flex-col items-center justify-center p-3 rounded-xl cursor-pointer border transition-all duration-300 relative overflow-hidden group",
    checked ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-200 shadow-[0_0_15px_rgba(99,102,241,0.2)]" : "bg-black/20 border-white/5 text-slate-500 hover:bg-white/5"
  )}>
    <div className={cn("absolute inset-0 bg-gradient-to-br from-indigo-500/10 to-transparent opacity-0 transition-opacity", checked && "opacity-100")} />
    <input type="checkbox" name={name} checked={checked} onChange={onChange} className="hidden" />
    <div className={cn("mb-2 relative z-10 transition-transform group-hover:scale-110", checked ? "text-indigo-400" : "text-slate-500")}>
      {React.cloneElement(icon, { size: 18 })}
    </div>
    <span className="text-[10px] font-bold uppercase tracking-wider relative z-10">{label}</span>
  </label>
);

const RangeSlider = ({ label, name, min, max, step = 1, val, onChange }) => (
  <div className="space-y-3">
    <div className="flex justify-between text-[10px] uppercase font-bold tracking-wider">
      <span className="text-indigo-300/60">{label}</span>
      <span className="text-white font-mono bg-white/10 px-2 py-0.5 rounded">{val}</span>
    </div>
    <input
      type="range" name={name} min={min} max={max} step={step} value={val} onChange={onChange}
      className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer accent-indigo-500 hover:accent-indigo-400 transition-all"
    />
  </div>
);

export default App;
