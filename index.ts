import {
  Scene,
  PerspectiveCamera,
  PlaneGeometry,
  Mesh,
  WebGLRenderer,
  Color,
  DoubleSide,
  FloatType,
  RedFormat,
  DataTexture,
  ShaderMaterial,
  VideoTexture,
  Texture,
  GridHelper,
} from "three";
import "@tensorflow/tfjs-backend-webgpu";
import {
  sub,
  relu,
  div,
  max,
  min,
  squeeze,
  tidy,
  browser,
  image,
  expandDims,
  TensorLike,
  Tensor,
  Rank,
  ready,
} from "@tensorflow/tfjs-core";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import { LookingGlassWebXRPolyfill } from "@lookingglass/webxr";
import { VRButton } from "three/addons/webxr/VRButton.js";

new LookingGlassWebXRPolyfill({
  targetX: 0.023710900360862243,
  targetY: -0.010309250276921724,
  targetZ: 0.6353649044950787,
  fovy: (1 * Math.PI) / 180,
  targetDiam: 0.9869841960442293,
  trackballX: -0.04000000000000008,
  trackballY: 2.6020852139652106e-16,
  depthiness: 0.85,
});

await ready();

const canvas = document.getElementById("root")!;

const renderer = new WebGLRenderer({
  antialias: true,
  canvas,
});
renderer.xr.enabled = true;
document.body.appendChild(VRButton.createButton(renderer));

const model = await loadGraphModel("./pydnet.json");

const camera = new PerspectiveCamera(
  90,
  window.innerWidth / window.innerHeight,
  0.01,
  10
);
camera.position.z = 1;

const scene = new Scene();
scene.add(new GridHelper());

const geometry = new PlaneGeometry(1, 1, 400, 200);
const material = new ShaderMaterial({
  uniforms: {
    depthMap: { value: new Texture() },
    colorMap: { value: new Texture() },
  },
  vertexShader: `
  varying vec3 vUv;
  uniform sampler2D depthMap;

  void main() {
    vUv = position; 

    float depth = texture2D(depthMap, vUv.xy * vec2(1.0, -1.0) + vec2(0.5)).r;

    vec4 modelViewPosition = modelViewMatrix * vec4(position.xy, depth * 1.0, 1.0);
    gl_Position = projectionMatrix * modelViewPosition;
  }`,
  fragmentShader: `
  varying vec3 vUv;
  uniform sampler2D colorMap;

  void main() {
    gl_FragColor = texture2D(colorMap, vUv.xy * vec2(1.0) + vec2(0.5));
  }`,
});
material.side = DoubleSide;
material.transparent = true;

const mesh = new Mesh(geometry, material);
scene.add(mesh);

renderer.setSize(window.innerWidth, window.innerHeight);

window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

canvas.addEventListener("click", async () => {
  const mediaStream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
  });
  const video = document.createElement("video");
  video.srcObject = mediaStream;
  video.play();
  material.uniforms.colorMap.value = new VideoTexture(video);
  material.needsUpdate = true;
  renderer.setAnimationLoop(() => startRender(video));
});

renderer.setAnimationLoop(() => renderer.render(scene, camera));

// animation

async function startRender(video: HTMLVideoElement) {
  mesh.scale.x = video.videoWidth / video.videoHeight;
  renderer.render(scene, camera);
  predictDepth(video);
}

async function predictDepth(source: HTMLVideoElement) {
  const raw_input = browser.fromPixels(source);
  const upsampledraw_input = image.resizeBilinear(raw_input, [384, 640]);
  const preprocessedInput = expandDims(upsampledraw_input);
  const divided = div(preprocessedInput, 255.0);
  const result = model.predict(divided) as Tensor<Rank>;
  const output = prepareOutput(result);
  const data = await output.data();

  divided.dispose();
  upsampledraw_input.dispose();
  preprocessedInput.dispose();
  raw_input.dispose();
  result.dispose();
  output.dispose();

  const dataTexture = new DataTexture(data, 640, 384, RedFormat, FloatType);
  material.uniforms.depthMap.value.dispose();
  material.uniforms.depthMap.value = dataTexture;
  dataTexture.needsUpdate = true;
  material.needsUpdate = true;
}

function prepareOutput(
  tensor: TensorLike | Tensor<Rank>
) {
  return tidy(() => {
    tensor = relu(tensor);
    tensor = squeeze(tensor);
    var min_value = min(tensor);
    var max_value = max(tensor);
    tensor = div(sub(tensor, min_value), sub(max_value, min_value));
    return tensor;
  });
}
