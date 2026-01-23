import React, { Suspense, useRef, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, Html, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

// Component to load and display a GLB model from base64
function Model({ glbBase64, renderMode, modelColor }) {
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);
  const groupRef = useRef();

  useEffect(() => {
    if (!glbBase64) {
      setModel(null);
      return;
    }

    try {
      // Convert base64 to ArrayBuffer
      const binaryString = atob(glbBase64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const arrayBuffer = bytes.buffer;

      // Load GLB using GLTFLoader
      const loader = new GLTFLoader();
      loader.parse(
        arrayBuffer,
        '',
        (gltf) => {
          console.log('✅ GLB loaded successfully');
          const scene = gltf.scene;
          
          // Fix orientation: rotate -90 degrees on X axis to lay flat (horizontal)
          // This converts from Y-up to proper floor plan orientation
          scene.rotation.x = -Math.PI / 2;
          
          // Compute bounding box after rotation
          const box = new THREE.Box3().setFromObject(scene);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          
          console.log('📦 Model size:', size.x.toFixed(2), 'x', size.y.toFixed(2), 'x', size.z.toFixed(2));
          
          // Center the model
          scene.position.x = -center.x;
          scene.position.y = -center.y;
          scene.position.z = -center.z;
          
          // Scale to fit in view (normalize to ~3 units max)
          const maxDim = Math.max(size.x, size.y, size.z);
          if (maxDim > 0) {
            const scale = 3 / maxDim;
            scene.scale.setScalar(scale);
          }
          
          // Parse the initial color
          const initialColor = new THREE.Color(modelColor || '#7799bb');
          
          // Fix all materials
          scene.traverse((child) => {
            if (child.isMesh) {
              // Make sure geometry has computed normals
              if (child.geometry) {
                child.geometry.computeVertexNormals();
              }
              
              // Get or create a proper material
              const hasVertexColors = child.geometry?.attributes?.color;
              
              child.material = new THREE.MeshStandardMaterial({
                color: hasVertexColors ? 0xffffff : initialColor,
                vertexColors: hasVertexColors,
                roughness: 0.4,
                metalness: 0.1,
                side: THREE.DoubleSide,
                flatShading: false,
              });
              
              child.castShadow = true;
              child.receiveShadow = true;
            }
          });
          
          setModel(scene);
          setError(null);
        },
        (err) => {
          console.error('❌ GLB loading error:', err);
          setError(err.message || 'Failed to load model');
        }
      );
    } catch (e) {
      console.error('❌ Error processing GLB:', e);
      setError(e.message);
    }
  }, [glbBase64]);

  // Apply render mode and color
  useEffect(() => {
    if (!model) return;
    
    const colorValue = new THREE.Color(modelColor || '#7799bb');
    
    model.traverse((child) => {
      if (child.isMesh && child.material) {
        const hasVertexColors = child.geometry?.attributes?.color;
        
        if (renderMode === 'clay') {
          child.material = new THREE.MeshStandardMaterial({
            color: 0xcccccc,
            roughness: 0.6,
            metalness: 0.0,
            side: THREE.DoubleSide,
            flatShading: false,
          });
        } else if (renderMode === 'wireframe') {
          child.material = new THREE.MeshBasicMaterial({
            color: colorValue,
            wireframe: true,
            side: THREE.DoubleSide,
          });
        } else {
          child.material = new THREE.MeshStandardMaterial({
            color: hasVertexColors ? 0xffffff : colorValue,
            vertexColors: hasVertexColors,
            roughness: 0.4,
            metalness: 0.1,
            side: THREE.DoubleSide,
            flatShading: false,
          });
        }
      }
    });
  }, [model, renderMode, modelColor]);

  if (error) {
    return (
      <Html center>
        <div style={{ 
          color: '#f87171', 
          fontSize: '12px', 
          background: 'rgba(0,0,0,0.8)', 
          padding: '16px', 
          borderRadius: '8px',
          border: '1px solid rgba(248,113,113,0.3)'
        }}>
          <p>⚠️ Error loading 3D model</p>
          <p style={{ fontSize: '10px', opacity: 0.7, marginTop: '4px' }}>{error}</p>
        </div>
      </Html>
    );
  }

  if (!model) {
    return (
      <Html center>
        <div style={{ color: '#22d3ee', fontSize: '12px' }}>Loading 3D Model...</div>
      </Html>
    );
  }

  return (
    <group ref={groupRef}>
      <primitive object={model} />
    </group>
  );
}

// Main ModelViewer Component
export default function ModelViewer({ 
  glbBase64, 
  renderMode = 'normal',
  modelColor = '#7799bb',
}) {
  if (!glbBase64) {
    return (
      <div style={{ 
        width: '100%', 
        height: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: '#0f0f1a',
        color: '#6366f1',
        fontSize: '12px',
        fontFamily: 'monospace'
      }}>
        NO_MODEL_DATA
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#0f0f1a' }}>
      <Canvas
        camera={{ 
          position: [5, 4, 5], 
          fov: 50,
          near: 0.1,
          far: 1000
        }}
        gl={{ 
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance'
        }}
        shadows
        style={{ background: '#0f0f1a' }}
      >
        <color attach="background" args={['#0f0f1a']} />
        
        <Suspense fallback={
          <Html center>
            <div style={{ color: '#818cf8', fontSize: '12px', fontFamily: 'monospace' }}>
              LOADING_MESH...
            </div>
          </Html>
        }>
          {/* Lighting Setup */}
          <ambientLight intensity={0.8} />
          <hemisphereLight args={['#ffffff', '#444466', 1.0]} />
          <directionalLight position={[10, 20, 10]} intensity={1.5} castShadow />
          <directionalLight position={[-10, 10, -10]} intensity={0.8} />
          <directionalLight position={[0, -10, 5]} intensity={0.3} />
          <pointLight position={[0, 10, 0]} intensity={0.8} />
          
          {/* Model */}
          <Model glbBase64={glbBase64} renderMode={renderMode} modelColor={modelColor} />
          
          {/* Grid Floor */}
          <Grid
            position={[0, -1.5, 0]}
            args={[20, 20]}
            cellSize={0.5}
            cellThickness={0.5}
            cellColor="#4f46e5"
            sectionSize={2}
            sectionThickness={1}
            sectionColor="#6366f1"
            fadeDistance={25}
            fadeStrength={1}
            infiniteGrid
          />
          
          {/* Environment for reflections */}
          <Environment preset="studio" />
          
          {/* Orbit Controls - key props for proper interaction */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            panSpeed={1}
            rotateSpeed={1}
            zoomSpeed={1.2}
            minDistance={0.5}
            maxDistance={50}
            dampingFactor={0.05}
            enableDamping={true}
          />
        </Suspense>
      </Canvas>

      {/* Controls hint */}
      <div style={{
        position: 'absolute',
        bottom: '8px',
        right: '8px',
        display: 'flex',
        gap: '4px',
        fontSize: '9px',
        fontFamily: 'monospace',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        color: 'rgba(255,255,255,0.4)'
      }}>
        <span style={{ background: 'rgba(255,255,255,0.1)', padding: '4px 8px', borderRadius: '4px' }}>
          🖱️ Left: Rotate
        </span>
        <span style={{ background: 'rgba(255,255,255,0.1)', padding: '4px 8px', borderRadius: '4px' }}>
          🖱️ Right: Pan
        </span>
        <span style={{ background: 'rgba(255,255,255,0.1)', padding: '4px 8px', borderRadius: '4px' }}>
          ⚲ Scroll: Zoom
        </span>
      </div>
    </div>
  );
}
