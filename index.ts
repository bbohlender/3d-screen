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
  slice,
} from "@tensorflow/tfjs-core";
import { loadGraphModel } from "@tensorflow/tfjs-converter";

await ready();

const canvas = document.getElementById("root")! as HTMLCanvasElement;

const renderer = new WebGLRenderer({
  antialias: true,
  canvas,
});

const model = await loadGraphModel("./pydnet.json");

const camera = new PerspectiveCamera(90, 1, 0.01, 10);
camera.position.z = 0.5;

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

    vec4 modelViewPosition = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * modelViewPosition;
  }`,
  fragmentShader: `
  varying vec3 vUv;
  uniform sampler2D colorMap;
  uniform sampler2D depthMap;

  void main() {
    if(vUv.x > 0.0) {
      gl_FragColor = texture2D(colorMap, vUv.xy + vec2(0.0, 0.5));
    } else {
      gl_FragColor = texture2D(depthMap, vUv.xy * vec2(2.0, -1.0) + vec2(1.0, 0.5));
    }
  }`,
});
material.side = DoubleSide;
material.transparent = true;

const mesh = new Mesh(geometry, material);
scene.add(mesh);

renderer.setSize(window.innerWidth, window.innerHeight);
canvas.width = window.innerWidth * window.devicePixelRatio;
canvas.height = window.innerHeight * window.devicePixelRatio;

window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  canvas.width = window.innerWidth * window.devicePixelRatio;
  canvas.height = window.innerHeight * window.devicePixelRatio;
  camera.updateProjectionMatrix();
});

canvas.addEventListener("click", async () => {
  const video = document.createElement("video");
  video.src = "./video2.mp4";
  video.play();
  material.uniforms.colorMap.value = new VideoTexture(video);
  material.needsUpdate = true;
  renderer.setAnimationLoop(() => startRender(video));
  const stream = canvas.captureStream(30);

  const mediaRecorder = new MediaRecorder(stream, {
    mimeType: "video/mp4; codecs=vp9",
  });

  const recordedChunks = [];

  //ondataavailable will fire in interval of `time || 4000 ms`
  mediaRecorder.start(1000);

  mediaRecorder.ondataavailable = function (event) {
    recordedChunks.push(event.data);
  };

  window.addEventListener("keydown", () => mediaRecorder.stop());

  mediaRecorder.onstop = function (event) {
    var blob = new Blob(recordedChunks, { type: "video/webm" });
    var url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("download", "recordingVideo");
    link.setAttribute("href", url);
    link.click();
  };
});

renderer.setAnimationLoop(() => renderer.render(scene, camera));

// animation

async function startRender(video: HTMLVideoElement) {
  renderer.render(scene, camera);
  predictDepth(video);
}

async function predictDepth(source: HTMLVideoElement) {
  const raw_input = browser.fromPixels(source);
  const sliced = slice(
    raw_input,
    [0, 0, 0],
    [source.videoWidth / 2, source.videoHeight, 3]
  );
  const upsampledraw_input = image.resizeBilinear(sliced, [384, 640]);
  const preprocessedInput = expandDims(upsampledraw_input);
  const divided = div(preprocessedInput, 255.0);
  const result = model.predict(divided) as Tensor<Rank>;
  const output = prepareOutput(result);
  const data = await output.data();

  sliced.dispose();
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

function prepareOutput(tensor: TensorLike | Tensor<Rank>) {
  return tidy(() => {
    tensor = relu(tensor);
    tensor = squeeze(tensor);
    var min_value = min(tensor);
    var max_value = max(tensor);
    tensor = div(sub(tensor, min_value), sub(max_value, min_value));
    return tensor;
  });
}
