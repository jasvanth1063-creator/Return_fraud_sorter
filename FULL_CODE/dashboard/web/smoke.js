/* ==========================================================================
   CURSOR SMOKE: WebGL fluid simulation
   Faithfully ported from https://fwccomet.iiitb.ac.in/hackathon/js/theme.js
   ========================================================================== */

const SMOKE_CONFIG = {
  SIM_RESOLUTION: 128,        // physics grid
  DYE_RESOLUTION: 1024,       // colour sharpness
  DENSITY_DISSIPATION: 0.9,   // how fast smoke fades
  VELOCITY_DISSIPATION: 0.4,  // how fast motion slows down
  PRESSURE: 0.8,
  PRESSURE_ITERATIONS: 20,
  CURL: 12,                   // swirliness
  SPLAT_RADIUS: 0.35,         // size of the smoke at cursor tip
  SPLAT_FORCE: 6000,          // how hard cursor pushes smoke
  SHADING: true,              // gentle glossy look
  OPACITY: 0.65,              // overall smoke opacity
  SOFTNESS: 3.0,              // edge softness in pixels
  COLOR_SPEED: 0.35,          // how fast colour cycles
  INTENSITY_DARK: 0.22,       // colour strength on dark theme
  INTENSITY_LIGHT: 0.32,      // colour strength on light theme
  IDLE_STOP_MS: 7000          // stop rendering this long after last move
};

// Vibrant teal, cyan, purple, amber, green matching our glassmorphism palette
const SMOKE_PALETTE = [
  [0.31, 0.84, 0.72], // teal (#4fd6b8)
  [0.22, 0.74, 0.97], // cyan (#38bdf8)
  [0.65, 0.55, 0.98], // purple (#a78bfa)
  [0.95, 0.57, 0.35], // accent coral (#f2925a)
  [0.10, 0.85, 0.50]  // bright emerald
];

let smokeWrap = null;
let smokeCanvas = null;
let smokeRaf = null;
let lastMoveTime = 0;
let leavingSince = 0;
let fluid = null;
let lastTick = 0;

function getWebGLContext(canvas) {
  const params = { alpha: true, premultipliedAlpha: true, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false };
  let gl = canvas.getContext("webgl2", params);
  const isWebGL2 = !!gl;
  if (!isWebGL2) gl = canvas.getContext("webgl", params) || canvas.getContext("experimental-webgl", params);
  if (!gl) return null;

  let halfFloat, supportLinearFiltering;
  if (isWebGL2) {
    gl.getExtension("EXT_color_buffer_float");
    supportLinearFiltering = gl.getExtension("OES_texture_float_linear");
  } else {
    halfFloat = gl.getExtension("OES_texture_half_float");
    supportLinearFiltering = gl.getExtension("OES_texture_half_float_linear");
  }
  gl.clearColor(0, 0, 0, 0);
  const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat && halfFloat.HALF_FLOAT_OES;
  if (!halfFloatTexType) return null;

  let formatRGBA, formatRG, formatR;
  if (isWebGL2) {
    formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
    formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
    formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);
  } else {
    formatRGBA = formatRG = formatR = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
  }
  if (!formatRGBA || !formatRG || !formatR) return null;
  return { gl, ext: { formatRGBA, formatRG, formatR, halfFloatTexType, supportLinearFiltering } };
}

function getSupportedFormat(gl, internalFormat, format, type) {
  if (!supportRenderTextureFormat(gl, internalFormat, format, type)) {
    switch (internalFormat) {
      case gl.R16F: return getSupportedFormat(gl, gl.RG16F, gl.RG, type);
      case gl.RG16F: return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type);
      default: return null;
    }
  }
  return { internalFormat, format };
}

function supportRenderTextureFormat(gl, internalFormat, format, type) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  return gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
}

function initFluid(canvas, gl, ext) {
  const config = SMOKE_CONFIG;
  if (!ext.supportLinearFiltering) { config.DYE_RESOLUTION = 512; config.SHADING = false; }

  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor(window.innerWidth * dpr);
    const h = Math.floor(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + "px";
    canvas.style.height = window.innerHeight + "px";
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; return true; }
    return false;
  }
  resizeCanvas();

  function compileShader(type, source, keywords) {
    if (keywords) source = keywords.map((k) => "#define " + k + "\n").join("") + source;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(shader));
    return shader;
  }

  function makeProgram(vs, fs) {
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.bindAttribLocation(program, 0, "aPosition");
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(program));
    const uniforms = {};
    const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const name = gl.getActiveUniform(program, i).name;
      uniforms[name] = gl.getUniformLocation(program, name);
    }
    return { program, uniforms, bind() { gl.useProgram(program); } };
  }

  const baseVertex = compileShader(gl.VERTEX_SHADER, `
    precision highp float;
    attribute vec2 aPosition;
    varying vec2 vUv, vL, vR, vT, vB;
    uniform vec2 texelSize;
    void main () {
      vUv = aPosition * 0.5 + 0.5;
      vL = vUv - vec2(texelSize.x, 0.0);
      vR = vUv + vec2(texelSize.x, 0.0);
      vT = vUv + vec2(0.0, texelSize.y);
      vB = vUv - vec2(0.0, texelSize.y);
      gl_Position = vec4(aPosition, 0.0, 1.0);
    }`);

  const clearShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying highp vec2 vUv;
    uniform sampler2D uTexture;
    uniform float value;
    void main () { gl_FragColor = value * texture2D(uTexture, vUv); }`);

  const splatShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uTarget;
    uniform float aspectRatio;
    uniform vec3 color;
    uniform vec2 point;
    uniform float radius;
    void main () {
      vec2 p = vUv - point.xy;
      p.x *= aspectRatio;
      vec3 splat = exp(-dot(p, p) / radius) * color;
      vec3 base = texture2D(uTarget, vUv).xyz;
      gl_FragColor = vec4(base + splat, 1.0);
    }`);

  const advectionShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uVelocity;
    uniform sampler2D uSource;
    uniform vec2 texelSize;
    uniform vec2 dyeTexelSize;
    uniform float dt;
    uniform float dissipation;
    vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
      vec2 st = uv / tsize - 0.5;
      vec2 iuv = floor(st);
      vec2 fuv = fract(st);
      vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
      vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
      vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
      vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
      return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
    }
    void main () {
    #ifdef MANUAL_FILTERING
      vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
      vec4 result = bilerp(uSource, coord, dyeTexelSize);
    #else
      vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
      vec4 result = texture2D(uSource, coord);
    #endif
      float decay = 1.0 + dissipation * dt;
      gl_FragColor = result / decay;
    }`, ext.supportLinearFiltering ? null : ["MANUAL_FILTERING"]);

  const divergenceShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying highp vec2 vUv, vL, vR, vT, vB;
    uniform sampler2D uVelocity;
    void main () {
      float L = texture2D(uVelocity, vL).x;
      float R = texture2D(uVelocity, vR).x;
      float T = texture2D(uVelocity, vT).y;
      float B = texture2D(uVelocity, vB).y;
      vec2 C = texture2D(uVelocity, vUv).xy;
      if (vL.x < 0.0) { L = -C.x; }
      if (vR.x > 1.0) { R = -C.x; }
      if (vT.y > 1.0) { T = -C.y; }
      if (vB.y < 0.0) { B = -C.y; }
      gl_FragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
    }`);

  const curlShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying highp vec2 vUv, vL, vR, vT, vB;
    uniform sampler2D uVelocity;
    void main () {
      float L = texture2D(uVelocity, vL).y;
      float R = texture2D(uVelocity, vR).y;
      float T = texture2D(uVelocity, vT).x;
      float B = texture2D(uVelocity, vB).x;
      gl_FragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
    }`);

  const vorticityShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    varying vec2 vUv, vL, vR, vT, vB;
    uniform sampler2D uVelocity;
    uniform sampler2D uCurl;
    uniform float curl;
    uniform float dt;
    void main () {
      float L = texture2D(uCurl, vL).x;
      float R = texture2D(uCurl, vR).x;
      float T = texture2D(uCurl, vT).x;
      float B = texture2D(uCurl, vB).x;
      float C = texture2D(uCurl, vUv).x;
      vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
      force /= length(force) + 0.0001;
      force *= curl * C;
      force.y *= -1.0;
      vec2 velocity = texture2D(uVelocity, vUv).xy + force * dt;
      velocity = clamp(velocity, -1000.0, 1000.0);
      gl_FragColor = vec4(velocity, 0.0, 1.0);
    }`);

  const pressureShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying highp vec2 vUv, vL, vR, vT, vB;
    uniform sampler2D uPressure;
    uniform sampler2D uDivergence;
    void main () {
      float L = texture2D(uPressure, vL).x;
      float R = texture2D(uPressure, vR).x;
      float T = texture2D(uPressure, vT).x;
      float B = texture2D(uPressure, vB).x;
      float divergence = texture2D(uDivergence, vUv).x;
      gl_FragColor = vec4((L + R + B + T - divergence) * 0.25, 0.0, 0.0, 1.0);
    }`);

  const gradientSubtractShader = compileShader(gl.FRAGMENT_SHADER, `
    precision mediump float;
    varying highp vec2 vUv, vL, vR, vT, vB;
    uniform sampler2D uPressure;
    uniform sampler2D uVelocity;
    void main () {
      float L = texture2D(uPressure, vL).x;
      float R = texture2D(uPressure, vR).x;
      float T = texture2D(uPressure, vT).x;
      float B = texture2D(uPressure, vB).x;
      vec2 velocity = texture2D(uVelocity, vUv).xy - vec2(R - L, T - B);
      gl_FragColor = vec4(velocity, 0.0, 1.0);
    }`);

  const displayShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uTexture;
    uniform vec2 blurStep;
    uniform float uOpacity;

    vec3 soft (vec2 uv) {
      vec3 c = texture2D(uTexture, uv).rgb * 0.25;
      c += texture2D(uTexture, uv + vec2( blurStep.x, 0.0)).rgb * 0.125;
      c += texture2D(uTexture, uv + vec2(-blurStep.x, 0.0)).rgb * 0.125;
      c += texture2D(uTexture, uv + vec2(0.0,  blurStep.y)).rgb * 0.125;
      c += texture2D(uTexture, uv + vec2(0.0, -blurStep.y)).rgb * 0.125;
      c += texture2D(uTexture, uv + blurStep).rgb * 0.0625;
      c += texture2D(uTexture, uv - blurStep).rgb * 0.0625;
      c += texture2D(uTexture, uv + vec2(blurStep.x, -blurStep.y)).rgb * 0.0625;
      c += texture2D(uTexture, uv + vec2(-blurStep.x, blurStep.y)).rgb * 0.0625;
      return c;
    }

    void main () {
      vec3 c = soft(vUv);
    #ifdef SHADING
      vec2 g = blurStep * 1.5;
      float dx = length(texture2D(uTexture, vUv + vec2(g.x, 0.0)).rgb) - length(texture2D(uTexture, vUv - vec2(g.x, 0.0)).rgb);
      float dy = length(texture2D(uTexture, vUv + vec2(0.0, g.y)).rgb) - length(texture2D(uTexture, vUv - vec2(0.0, g.y)).rgb);
      vec3 n = normalize(vec3(dx, dy, length(blurStep)));
      float diffuse = clamp(dot(n, vec3(0.0, 0.0, 1.0)) + 0.85, 0.85, 1.0);
      c *= diffuse;
    #endif
      c = clamp(c, 0.0, 1.0);
      float a = max(c.r, max(c.g, c.b));
      gl_FragColor = vec4(c, a) * uOpacity;
    }`, config.SHADING ? ["SHADING"] : null);

  const clearProgram = makeProgram(baseVertex, clearShader);
  const splatProgram = makeProgram(baseVertex, splatShader);
  const advectionProgram = makeProgram(baseVertex, advectionShader);
  const divergenceProgram = makeProgram(baseVertex, divergenceShader);
  const curlProgram = makeProgram(baseVertex, curlShader);
  const vorticityProgram = makeProgram(baseVertex, vorticityShader);
  const pressureProgram = makeProgram(baseVertex, pressureShader);
  const gradientSubtractProgram = makeProgram(baseVertex, gradientSubtractShader);
  const displayProgram = makeProgram(baseVertex, displayShader);

  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);

  function blit(target) {
    if (target == null) {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
      gl.viewport(0, 0, target.width, target.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  function createFBO(w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return {
      texture, fbo, width: w, height: h, texelSizeX: 1 / w, texelSizeY: 1 / h,
      attach(id) { gl.activeTexture(gl.TEXTURE0 + id); gl.bindTexture(gl.TEXTURE_2D, texture); return id; },
    };
  }

  function createDoubleFBO(w, h, internalFormat, format, type, param) {
    let a = createFBO(w, h, internalFormat, format, type, param);
    let b = createFBO(w, h, internalFormat, format, type, param);
    return {
      width: w, height: h, texelSizeX: a.texelSizeX, texelSizeY: a.texelSizeY,
      get read() { return a; }, get write() { return b; },
      swap() { const t = a; a = b; b = t; },
    };
  }

  function getResolution(resolution) {
    let aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspect < 1) aspect = 1 / aspect;
    const min = Math.round(resolution);
    const max = Math.round(resolution * aspect);
    return gl.drawingBufferWidth > gl.drawingBufferHeight ? { width: max, height: min } : { width: min, height: max };
  }

  let dye, velocity, divergence, curl, pressure;
  function initFramebuffers() {
    const simRes = getResolution(config.SIM_RESOLUTION);
    const dyeRes = getResolution(config.DYE_RESOLUTION);
    const type = ext.halfFloatTexType;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;
    gl.disable(gl.BLEND);
    dye = createDoubleFBO(dyeRes.width, dyeRes.height, ext.formatRGBA.internalFormat, ext.formatRGBA.format, type, filtering);
    velocity = createDoubleFBO(simRes.width, simRes.height, ext.formatRG.internalFormat, ext.formatRG.format, type, filtering);
    divergence = createFBO(simRes.width, simRes.height, ext.formatR.internalFormat, ext.formatR.format, type, gl.NEAREST);
    curl = createFBO(simRes.width, simRes.height, ext.formatR.internalFormat, ext.formatR.format, type, gl.NEAREST);
    pressure = createDoubleFBO(simRes.width, simRes.height, ext.formatR.internalFormat, ext.formatR.format, type, gl.NEAREST);
  }
  initFramebuffers();

  function step(dt, densityDissipation) {
    gl.disable(gl.BLEND);

    curlProgram.bind();
    gl.uniform2f(curlProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(curl);

    vorticityProgram.bind();
    gl.uniform2f(vorticityProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(vorticityProgram.uniforms.uCurl, curl.attach(1));
    gl.uniform1f(vorticityProgram.uniforms.curl, config.CURL);
    gl.uniform1f(vorticityProgram.uniforms.dt, dt);
    blit(velocity.write); velocity.swap();

    divergenceProgram.bind();
    gl.uniform2f(divergenceProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(divergence);

    clearProgram.bind();
    gl.uniform1i(clearProgram.uniforms.uTexture, pressure.read.attach(0));
    gl.uniform1f(clearProgram.uniforms.value, config.PRESSURE);
    blit(pressure.write); pressure.swap();

    pressureProgram.bind();
    gl.uniform2f(pressureProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(pressureProgram.uniforms.uDivergence, divergence.attach(0));
    for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
      gl.uniform1i(pressureProgram.uniforms.uPressure, pressure.read.attach(1));
      blit(pressure.write); pressure.swap();
    }

    gradientSubtractProgram.bind();
    gl.uniform2f(gradientSubtractProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(gradientSubtractProgram.uniforms.uPressure, pressure.read.attach(0));
    gl.uniform1i(gradientSubtractProgram.uniforms.uVelocity, velocity.read.attach(1));
    blit(velocity.write); velocity.swap();

    advectionProgram.bind();
    gl.uniform2f(advectionProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    if (!ext.supportLinearFiltering)
      gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, velocity.texelSizeX, velocity.texelSizeY);
    const velocityId = velocity.read.attach(0);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocityId);
    gl.uniform1i(advectionProgram.uniforms.uSource, velocityId);
    gl.uniform1f(advectionProgram.uniforms.dt, dt);
    gl.uniform1f(advectionProgram.uniforms.dissipation, config.VELOCITY_DISSIPATION);
    blit(velocity.write); velocity.swap();

    if (!ext.supportLinearFiltering)
      gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, dye.texelSizeX, dye.texelSizeY);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionProgram.uniforms.uSource, dye.read.attach(1));
    gl.uniform1f(advectionProgram.uniforms.dissipation, densityDissipation);
    blit(dye.write); dye.swap();
  }

  function render() {
    displayProgram.bind();
    gl.uniform2f(displayProgram.uniforms.blurStep, config.SOFTNESS / gl.drawingBufferWidth, config.SOFTNESS / gl.drawingBufferHeight);
    gl.uniform1f(displayProgram.uniforms.uOpacity, config.OPACITY);
    gl.uniform1i(displayProgram.uniforms.uTexture, dye.read.attach(0));
    blit(null);
  }

  function correctRadius(radius) {
    const aspect = canvas.width / canvas.height;
    return aspect > 1 ? radius * aspect : radius;
  }

  function splat(x, y, dx, dy, color) {
    splatProgram.bind();
    gl.uniform1i(splatProgram.uniforms.uTarget, velocity.read.attach(0));
    gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(splatProgram.uniforms.point, x, y);
    gl.uniform3f(splatProgram.uniforms.color, dx, dy, 0);
    gl.uniform1f(splatProgram.uniforms.radius, correctRadius(config.SPLAT_RADIUS / 100));
    blit(velocity.write); velocity.swap();

    gl.uniform1i(splatProgram.uniforms.uTarget, dye.read.attach(0));
    gl.uniform3f(splatProgram.uniforms.color, color[0], color[1], color[2]);
    blit(dye.write); dye.swap();
  }

  let colorT = Math.random() * SMOKE_PALETTE.length;
  function currentColor(dt) {
    colorT = (colorT + dt * config.COLOR_SPEED) % SMOKE_PALETTE.length;
    const i = Math.floor(colorT);
    const f = colorT - i;
    const a = SMOKE_PALETTE[i];
    const b = SMOKE_PALETTE[(i + 1) % SMOKE_PALETTE.length];
    const s = f * f * (3 - 2 * f);
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const intensity = isDark ? config.INTENSITY_DARK : config.INTENSITY_LIGHT;
    return [0, 1, 2].map((k) => (a[k] + (b[k] - a[k]) * s) * intensity);
  }

  const pointer = { x: 0, y: 0, dx: 0, dy: 0, moved: false, started: false };
  function updatePointer(clientX, clientY) {
    const x = clientX / window.innerWidth;
    const y = 1 - clientY / window.innerHeight;
    if (!pointer.started) { pointer.x = x; pointer.y = y; pointer.started = true; return; }
    const aspect = canvas.width / canvas.height;
    let dx = x - pointer.x;
    let dy = y - pointer.y;
    if (aspect < 1) dx *= aspect;
    if (aspect > 1) dy /= aspect;
    pointer.dx += dx;
    pointer.dy += dy;
    pointer.x = x;
    pointer.y = y;
    pointer.moved = pointer.dx !== 0 || pointer.dy !== 0;
  }
  function resetPointer() { pointer.started = false; pointer.dx = pointer.dy = 0; pointer.moved = false; }

  function frame(dt, now) {
    if (resizeCanvas()) initFramebuffers();

    const color = currentColor(dt);
    if (pointer.moved && !leavingSince) {
      splat(pointer.x, pointer.y, pointer.dx * config.SPLAT_FORCE, pointer.dy * config.SPLAT_FORCE, color);
    }
    pointer.dx = pointer.dy = 0;
    pointer.moved = false;

    const leavingBoost = leavingSince ? 1 + Math.min(6, (now - leavingSince) / 140) : 1;
    step(dt, config.DENSITY_DISSIPATION * leavingBoost);
    render();
  }

  return { frame, updatePointer, resetPointer };
}

function tickSmoke(now) {
  const dt = lastTick ? Math.min((now - lastTick) / 1000, 1 / 60) : 1 / 60;
  lastTick = now;

  fluid.frame(dt, now);

  const idleFor = now - lastMoveTime;
  const cleared = leavingSince ? now - leavingSince > 1200 : idleFor > SMOKE_CONFIG.IDLE_STOP_MS;
  if (cleared) {
    smokeRaf = null;
    lastTick = 0;
    if (smokeWrap) smokeWrap.style.opacity = "0";
    leavingSince = 0;
    return;
  }
  smokeRaf = requestAnimationFrame(tickSmoke);
}

function ensureLoopRunning() {
  if (!smokeRaf && fluid) {
    lastTick = 0;
    smokeRaf = requestAnimationFrame(tickSmoke);
  }
}

function initCursorSmoke() {
  if (
    window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
    window.matchMedia("(max-width: 760px)").matches
  ) {
    return;
  }

  smokeWrap = document.createElement("div");
  smokeWrap.className = "liquid-smoke";
  smokeWrap.setAttribute("aria-hidden", "true");

  smokeCanvas = document.createElement("canvas");
  smokeWrap.appendChild(smokeCanvas);
  document.body.prepend(smokeWrap);

  const ctx = getWebGLContext(smokeCanvas);
  if (!ctx) {
    smokeWrap.remove();
    smokeWrap = smokeCanvas = null;
    return;
  }
  fluid = initFluid(smokeCanvas, ctx.gl, ctx.ext);

  window.addEventListener("pointermove", (e) => {
    fluid.updatePointer(e.clientX, e.clientY);
    lastMoveTime = performance.now();
    leavingSince = 0;
    smokeWrap.style.opacity = "1";
    ensureLoopRunning();
  });

  document.documentElement.addEventListener("mouseleave", () => {
    fluid.resetPointer();
    if (smokeRaf && !leavingSince) leavingSince = performance.now();
  });

  document.documentElement.addEventListener("mouseenter", () => {
    leavingSince = 0;
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initCursorSmoke);
} else {
  initCursorSmoke();
}
