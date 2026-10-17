import { Suspense, useEffect, useMemo, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { ContactShadows, OrbitControls, SoftShadows, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { LABEL_TO_OBJECT, SENSOR_INFO, TWIN_GLB_URL } from "../api.js";

// Materials in the dashboard theme; the GLB's material names carry the role (see backend twin.py).
const MATERIALS = {
  hand_light: new THREE.MeshStandardMaterial({ color: "#2f2f2f", roughness: 0.42, metalness: 0.15 }),
  hand_dark: new THREE.MeshStandardMaterial({ color: "#3a3a3a", roughness: 0.5, metalness: 0.1 }),
  sensor: new THREE.MeshStandardMaterial({ color: "#ffcc00", roughness: 0.32, metalness: 0.1 }),
  unknown: new THREE.MeshStandardMaterial({ color: "#a0a0a0", roughness: 0.5, transparent: true, opacity: 0.4 }),
};
const OBJECTS = Object.values(LABEL_TO_OBJECT);

// The palm body is one mesh: wrist, palm and back of the hand. It is split per face into three
// material groups. Hand frame: fingers +z, palm faces +x; the wrist is the narrow part below the palm.
const WRIST_TOP_Z = 0.044; // m, where the wrist neck widens into the palm (profiled from the mesh)
const HAND_ROOT_MATERIALS = [
  new THREE.MeshStandardMaterial({ color: "#2f2f2f", roughness: 0.42, metalness: 0.1 }), // palm: black
  new THREE.MeshStandardMaterial({ color: "#ededed", roughness: 0.42, metalness: 0 }), // back: white
  // wrist: brushed metal; envMap (local RoomEnvironment) is set in Twin, metal needs reflections
  new THREE.MeshStandardMaterial({ color: "#b4b8bd", roughness: 0.28, metalness: 0.9, envMapIntensity: 1.1 }),
];

function splitHandRoot(mesh) {
  // Smooth normals first (indexed), then un-index so faces can be regrouped by material.
  if (mesh.geometry.index) mesh.geometry.computeVertexNormals();
  const src = mesh.geometry.index ? mesh.geometry.toNonIndexed() : mesh.geometry;
  const p = src.attributes.position, n = src.attributes.normal;
  const faces = [[], [], []];
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  for (let f = 0; f < p.count; f += 3) {
    a.fromBufferAttribute(p, f);
    b.fromBufferAttribute(p, f + 1);
    c.fromBufferAttribute(p, f + 2);
    const zMid = (a.z + b.z + c.z) / 3;
    const nx = b.clone().sub(a).cross(c.clone().sub(a)).normalize().x; // face normal, not smoothed
    faces[zMid < WRIST_TOP_Z ? 2 : nx < -0.25 ? 1 : 0].push(f);
  }
  const pos = new Float32Array(p.count * 3), nor = new Float32Array(p.count * 3);
  const g = new THREE.BufferGeometry();
  let v = 0;
  faces.forEach((list, group) => {
    g.addGroup(v, list.length * 3, group);
    for (const f of list) {
      for (let k = 0; k < 3; k++, v++) {
        pos.set([p.getX(f + k), p.getY(f + k), p.getZ(f + k)], v * 3);
        nor.set([n.getX(f + k), n.getY(f + k), n.getZ(f + k)], v * 3);
      }
    }
  });
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  mesh.geometry = g;
  mesh.material = HAND_ROOT_MATERIALS;
}
const FLOOR_GAP = 0.004; // shadow plane just below the lowest point of the model

function applyPose(node, p) {
  node.position.set(p[0], p[1], p[2]);
  node.quaternion.set(p[4], p[5], p[6], p[3]); // MuJoCo w,x,y,z -> three x,y,z,w
}

function Twin({ frame, truthObject, estimate, onFloor, mirrored }) {
  const { scene } = useGLTF(TWIN_GLB_URL);
  const gl = useThree((s) => s.gl);
  // Reflections for the metal wrist only, from three's built-in studio room (nothing downloaded).
  useEffect(() => {
    const pmrem = new THREE.PMREMGenerator(gl);
    const env = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    HAND_ROOT_MATERIALS[2].envMap = env;
    HAND_ROOT_MATERIALS[2].needsUpdate = true;
    return () => { env.dispose(); pmrem.dispose(); };
  }, [gl]);
  const root = useMemo(() => {
    const s = scene.clone(true);
    s.traverse((o) => {
      if (!o.isMesh) return;
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals(); // MuJoCo meshes ship without
      const role = o.material?.name;
      o.material = MATERIALS[role] ?? (OBJECTS.includes(role) ? MATERIALS.sensor : MATERIALS.hand_light);
      if (o.name.endsWith("_hand_root") || o.parent?.name.endsWith("_hand_root")) splitHandRoot(o);
      o.castShadow = true;
    });
    if (import.meta.env.DEV) window.__twin = s; // inspect from the browser console
    return s;
  }, [scene]);

  useEffect(() => {
    if (!frame) return;
    for (const [name, pose] of Object.entries(frame.bodies)) {
      if (name === truthObject) continue;
      const node = root.getObjectByName(name);
      if (node) applyPose(node, pose);
    }
    // Show the *estimated* sensor at the object's pose; grey ghost until the classifier has a vote.
    // Live mode has no object until there is an estimate and a position (truthObject = null).
    const shown = !truthObject ? null : estimate ? LABEL_TO_OBJECT[estimate.label] : truthObject;
    for (const obj of OBJECTS) {
      const node = root.getObjectByName(obj);
      if (!node) continue;
      node.visible = obj === shown;
      if (node.visible) {
        applyPose(node, frame.bodies[truthObject]);
        node.traverse((o) => { if (o.isMesh) o.material = estimate ? MATERIALS.sensor : MATERIALS.unknown; });
      }
    }
  }, [root, frame, truthObject, estimate]);

  // Rest the contact shadow under the model once per session (the floor stays put during playback).
  useEffect(() => {
    if (!frame) return;
    root.updateMatrixWorld(true);
    const box = new THREE.Box3();
    root.traverseVisible((o) => { if (o.isMesh) box.expandByObject(o, true); });
    if (!box.isEmpty()) onFloor(box.min.z - FLOOR_GAP);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root, truthObject, frame != null]);

  // The left hand is the right one mirrored across the plane through the hand root (y = 0.1), the
  // same construction as simulation/hand_sim/left_hand.py; three flips the face winding itself.
  return mirrored ? (
    <group position={[0, 0.2, 0]} scale={[1, -1, 1]}>
      <primitive object={root} />
    </group>
  ) : (
    <primitive object={root} />
  );
}

const STAGE = "#f5f5f5"; // background = fog colour, so the floor fades into an endless white space

// Infinite white photo studio: shadow-receiving floor fading into fog, soft key light with shadows.
function Studio({ floorZ }) {
  return (
    <>
      <color attach="background" args={[STAGE]} />
      <fog attach="fog" args={[STAGE, 1.2, 5]} />
      <SoftShadows size={32} samples={16} focus={0.4} />
      <hemisphereLight args={["#ffffff", "#d9d9d9", 1.6]} position={[0, 0, 1]} />
      <directionalLight
        position={[0.55, -0.35, 0.9]}
        intensity={1.9}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-bias={-0.0004}
        shadow-normalBias={0.01}
        shadow-radius={10}
        shadow-camera-left={-0.35}
        shadow-camera-right={0.35}
        shadow-camera-top={0.35}
        shadow-camera-bottom={-0.35}
        shadow-camera-near={0.2}
        shadow-camera-far={2.5}
      >
        <object3D attach="target" position={[0.03, 0.1, 0.05]} />
      </directionalLight>
      <directionalLight position={[-0.6, 0.8, 0.4]} intensity={0.55} />
      {/* Soft back light so the white back of the hand reads white, not grey. */}
      <directionalLight position={[-0.9, -0.1, 0.5]} intensity={1.1} />
      <mesh position={[0.03, 0.1, floorZ]} receiveShadow>
        <circleGeometry args={[6, 96]} />
        <meshStandardMaterial color="#fbfbfb" roughness={1} metalness={0} />
      </mesh>
      {/* Contact shadow for grounding where the model meets the floor (local XZ -> sim XY). */}
      <group position={[0.03, 0.1, floorZ + 0.0005]} rotation={[Math.PI / 2, 0, 0]}>
        <ContactShadows opacity={0.35} scale={0.5} blur={2.2} far={0.12} resolution={1024} color="#2a2a2a" />
      </group>
    </>
  );
}

// session: {object, hand_side} of the replayed session or the live hand; caption: second chip;
// message: centred notice over the viewport (e.g. waiting for the live stream).
export default function TwinView({ frame, session, estimate, caption, message }) {
  const info = estimate ? SENSOR_INFO[estimate.label] : null;
  // Replay sessions come from the right-hand sim; the live left hand is shown mirrored.
  const mirrored = session?.mirrored ?? false;
  const [floorZ, setFloorZ] = useState(-0.03);
  return (
    <div className="viewport">
      <Canvas
        shadows
        flat
        dpr={[1, 2]}
        camera={{ position: [0.36, -0.24, 0.2], up: [0, 0, 1], fov: 30, near: 0.01, far: 20 }}
      >
        <Studio floorZ={floorZ} />
        <Suspense fallback={null}>
          {session && frame && (
            <Twin frame={frame} truthObject={session.object} estimate={estimate} onFloor={setFloorZ} mirrored={mirrored} />
          )}
        </Suspense>
        <OrbitControls
          target={[0.03, 0.1, 0.09]}
          enableDamping
          makeDefault
          minDistance={0.18}
          maxDistance={1.4}
          maxPolarAngle={Math.PI * 0.49}
        />
      </Canvas>

      <div className="overlay">
        <span className="chip">
          <span className="swatch" style={{ background: estimate ? "var(--yellow)" : "#8a8a8a" }} />
          {info ? (
            <>Estimated <strong>{info.name}</strong> · {info.model}</>
          ) : (
            <>Waiting for contact · sensor unidentified</>
          )}
        </span>
        {caption && <span className="chip">{caption}</span>}
      </div>
      <div className="overlay overlay-right">
        <span className="chip">Drag to orbit · scroll to zoom</span>
      </div>
      {(message || !session) && <div className="viewport-msg">{message || "Loading digital twin…"}</div>}
    </div>
  );
}

useGLTF.preload(TWIN_GLB_URL);
