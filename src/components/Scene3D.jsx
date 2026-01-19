import React, { useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Stars, Float, PerspectiveCamera, Environment, Text } from '@react-three/drei';
import * as THREE from 'three';

function CameraRig({ startIntro }) {
    const { camera } = useThree();
    const vec = new THREE.Vector3();

    useFrame((state) => {
        // Cinematic Fly-in
        // Start at z=50, fly to z=10
        const t = state.clock.getElapsedTime();

        // Smooth easing camera movement
        // We want to move from z=50 to z=10 quickly then slow down
        // Using a lerp-like effect based on time

        const targetZ = 10;
        const startZ = 40;

        // Animate only for the first 4 seconds
        if (t < 4) {
            const progress = Math.min(t / 3.5, 1);
            // Cubic ease out: 1 - pow(1 - x, 3)
            const ease = 1 - Math.pow(1 - progress, 3);

            const currentZ = startZ - (startZ - targetZ) * ease;
            camera.position.set(0, 0, currentZ);

            // Slight rotation drift for realism
            camera.rotation.z = THREE.MathUtils.degToRad(2 * (1 - ease));
        } else {
            // Idle drift after Intro
            camera.position.z = THREE.MathUtils.lerp(camera.position.z, 10 + Math.sin(t * 0.5) * 0.5, 0.02);
            camera.rotation.z = THREE.MathUtils.lerp(camera.rotation.z, 0, 0.05);
        }
    });
    return null;
}

function ArchitecturalStructure() {
    const group = useRef();

    useFrame((state) => {
        const t = state.clock.getElapsedTime();
        group.current.rotation.y = t * 0.15;
        group.current.rotation.x = Math.sin(t * 0.2) * 0.1;
    });

    return (
        <group ref={group}>
            {/* Central Core */}
            <mesh>
                <dodecahedronGeometry args={[1.5, 0]} />
                <meshStandardMaterial color="#4f46e5" wireframe emissive="#4f46e5" emissiveIntensity={0.8} />
            </mesh>

            {/* Orbiting Rings */}
            <mesh rotation={[Math.PI / 3, 0, 0]}>
                <torusGeometry args={[3, 0.02, 16, 100]} />
                <meshStandardMaterial color="#818cf8" emissive="#818cf8" emissiveIntensity={3} />
            </mesh>
            <mesh rotation={[-Math.PI / 3, 0, 0]}>
                <torusGeometry args={[4, 0.02, 16, 100]} />
                <meshStandardMaterial color="#c084fc" emissive="#c084fc" emissiveIntensity={3} />
            </mesh>

            <mesh rotation={[0, Math.PI / 2, 0]}>
                <torusGeometry args={[6, 0.01, 16, 100]} />
                <meshStandardMaterial color="#22d3ee" emissive="#22d3ee" emissiveIntensity={1} />
            </mesh>

            {/* Floating Particles */}
            {Array.from({ length: 40 }).map((_, i) => (
                <Float speed={2} rotationIntensity={2} floatIntensity={4} key={i}>
                    <mesh position={[
                        (Math.random() - 0.5) * 25,
                        (Math.random() - 0.5) * 25,
                        (Math.random() - 0.5) * 20
                    ]}>
                        <octahedronGeometry args={[0.1]} />
                        <meshStandardMaterial color="cyan" emissive="cyan" emissiveIntensity={5} />
                    </mesh>
                </Float>
            ))}
        </group>
    );
}

function GridFloor() {
    return (
        <group rotation={[Math.PI / 2.5, 0, 0]} position={[0, -10, -20]}>
            <gridHelper
                args={[100, 50, 0x4f46e5, 0x1e1b4b]}
                position={[0, 0, 0]}
                rotation={[Math.PI / 2, 0, 0]}
            />
        </group>
    );
}

export default function Scene3D() {
    return (
        <div className="fixed inset-0 z-0 pointer-events-none opacity-80">
            <Canvas gl={{ antialias: true, toneMapping: THREE.ReinhardToneMapping }}>
                <CameraRig />
                <PerspectiveCamera makeDefault position={[0, 0, 50]} fov={60} />

                <ambientLight intensity={0.2} />
                <pointLight position={[10, 10, 10]} intensity={1} color="#818cf8" />
                <pointLight position={[-10, -10, -10]} intensity={1} color="#c084fc" />

                <ArchitecturalStructure />
                <GridFloor />

                <Stars radius={150} depth={50} count={8000} factor={4} saturation={0} fade speed={0.5} />
                <Environment preset="city" />

                <fog attach="fog" args={['#0f172a', 10, 60]} />
            </Canvas>
            <div className="absolute inset-0 bg-gradient-to-t from-[#050b14] via-transparent to-[#050b14] z-10"></div>
        </div>
    );
}
