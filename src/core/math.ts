export const PI = Math.PI;
export const TWO_PI = Math.PI * 2;

/** 列主序（Column-Major）4x4 矩阵，存于长度 16 的 Float32Array。右手系 / Y-Up。 */
export type Mat4 = Float32Array;

/** 构造单位矩阵（16 个元素）。 */
export function identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

/** 拷贝 src → dst，返回 dst。 */
export function copy(out: Mat4, src: Mat4): Mat4 {
  out.set(src);
  return out;
}

/**
 * 构造右手系透视矩阵（同数学库惯例）。
 * @param fovYRad 纵向视场角（弧度）
 * @param aspect 宽高比 width/height
 * @param near 近裁剪面
 * @param far 远裁剪面
 */
export function perspective(fovYRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovYRad / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far / (near - far);
  m[11] = -1;
  m[14] = (far * near) / (near - far);
  return m;
}

/** 右手系观察矩阵（相机朝向 -Z）。view = lookAt(eye, target, up)。 */
export function lookAt(eye: Float32Array, target: Float32Array, up: Float32Array): Mat4 {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];
  const zLen = Math.hypot(zx, zy, zz) || 1;
  zx /= zLen;
  zy /= zLen;
  zz /= zLen;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  const xLen = Math.hypot(xx, xy, xz) || 1;
  xx /= xLen;
  xy /= xLen;
  xz /= xLen;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  const m = new Float32Array(16);
  m[0] = xx;
  m[1] = yx;
  m[2] = zx;
  m[3] = 0;
  m[4] = xy;
  m[5] = yy;
  m[6] = zy;
  m[7] = 0;
  m[8] = xz;
  m[9] = yz;
  m[10] = zz;
  m[11] = 0;
  m[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  m[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  m[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  m[15] = 1;
  return m;
}

/**
 * 将旋转四元数 (x,y,z,w) 写入 out 的列主序旋转部分。
 * out 需为 16 长度；旋转外元素保持原值。
 */
export function rotationQuat(out: Mat4, x: number, y: number, z: number, w: number): Mat4 {
  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;

  out[0] = 1 - (yy + zz);
  out[1] = xy + wz;
  out[2] = xz - wy;
  out[4] = xy - wz;
  out[5] = 1 - (xx + zz);
  out[6] = yz + wx;
  out[8] = xz + wy;
  out[9] = yz - wx;
  out[10] = 1 - (xx + yy);
  return out;
}

/** 由 position + quat + scale 构造 TRS 矩阵（列主序），写入 out。 */
export function fromTranslationRotationScale(
  out: Mat4,
  px: number,
  py: number,
  pz: number,
  qx: number,
  qy: number,
  qz: number,
  qw: number,
  sx: number,
  sy: number,
  sz: number,
): Mat4 {
  rotationQuat(out, qx, qy, qz, qw);
  out[0] *= sx;
  out[1] *= sx;
  out[2] *= sx;
  out[4] *= sy;
  out[5] *= sy;
  out[6] *= sy;
  out[8] *= sz;
  out[9] *= sz;
  out[10] *= sz;
  out[12] = px;
  out[13] = py;
  out[14] = pz;
  out[15] = 1;
  return out;
}

/** 绕 Y 轴旋转 θ（弧度）的列主序矩阵，写入 out。 */
export function rotationY(out: Mat4, theta: number): Mat4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  out.fill(0);
  out[0] = c;
  out[2] = s;
  out[5] = 1;
  out[8] = -s;
  out[10] = c;
  out[15] = 1;
  return out;
}

/** 矩阵乘法 C = A * B（列主序）。安全：out 可与 A/B 不同但不得与其字节重叠。 */
export function multiply(out: Mat4, a: Mat4, b: Mat4): Mat4 {
  const a0 = a[0];
  const a1 = a[1];
  const a2 = a[2];
  const a3 = a[3];
  const a4 = a[4];
  const a5 = a[5];
  const a6 = a[6];
  const a7 = a[7];
  const a8 = a[8];
  const a9 = a[9];
  const a10 = a[10];
  const a11 = a[11];
  const a12 = a[12];
  const a13 = a[13];
  const a14 = a[14];
  const a15 = a[15];

  let b0 = b[0];
  let b1 = b[1];
  let b2 = b[2];
  let b3 = b[3];
  out[0] = b0 * a0 + b1 * a4 + b2 * a8 + b3 * a12;
  out[1] = b0 * a1 + b1 * a5 + b2 * a9 + b3 * a13;
  out[2] = b0 * a2 + b1 * a6 + b2 * a10 + b3 * a14;
  out[3] = b0 * a3 + b1 * a7 + b2 * a11 + b3 * a15;

  b0 = b[4];
  b1 = b[5];
  b2 = b[6];
  b3 = b[7];
  out[4] = b0 * a0 + b1 * a4 + b2 * a8 + b3 * a12;
  out[5] = b0 * a1 + b1 * a5 + b2 * a9 + b3 * a13;
  out[6] = b0 * a2 + b1 * a6 + b2 * a10 + b3 * a14;
  out[7] = b0 * a3 + b1 * a7 + b2 * a11 + b3 * a15;

  b0 = b[8];
  b1 = b[9];
  b2 = b[10];
  b3 = b[11];
  out[8] = b0 * a0 + b1 * a4 + b2 * a8 + b3 * a12;
  out[9] = b0 * a1 + b1 * a5 + b2 * a9 + b3 * a13;
  out[10] = b0 * a2 + b1 * a6 + b2 * a10 + b3 * a14;
  out[11] = b0 * a3 + b1 * a7 + b2 * a11 + b3 * a15;

  b0 = b[12];
  b1 = b[13];
  b2 = b[14];
  b3 = b[15];
  out[12] = b0 * a0 + b1 * a4 + b2 * a8 + b3 * a12;
  out[13] = b0 * a1 + b1 * a5 + b2 * a9 + b3 * a13;
  out[14] = b0 * a2 + b1 * a6 + b2 * a10 + b3 * a14;
  out[15] = b0 * a3 + b1 * a7 + b2 * a11 + b3 * a15;
  return out;
}

// ──────────────────────────────────────────────
// Sprint 3A: 数学库扩展
// ──────────────────────────────────────────────

/** 构造正交投影矩阵（列主序），写入 out。适用于 2D / UI 渲染。 */
export function orthographic(out: Mat4, left: number, right: number, bottom: number, top: number, near: number, far: number): Mat4 {
  const lr = 1 / (left - right);
  const bt = 1 / (bottom - top);
  const nf = 1 / (near - far);
  out.fill(0);
  out[0] = -2 * lr;
  out[5] = -2 * bt;
  out[10] = nf;
  out[12] = (left + right) * lr;
  out[13] = (top + bottom) * bt;
  out[14] = near * nf;
  out[15] = 1;
  return out;
}

/**
 * 4×4 矩阵求逆（列主序），写入 out。返回 false 表示奇异矩阵（out 不可信）。
 * 使用伴随矩阵法（adjugate / cofactor）。
 */
export function invert(out: Mat4, m: Mat4): boolean {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return false;
  det = 1 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return true;
}

/** 构造平移矩阵（列主序），写入 out。 */
export function translation(out: Mat4, x: number, y: number, z: number): Mat4 {
  out.fill(0);
  out[0] = 1;
  out[5] = 1;
  out[10] = 1;
  out[12] = x;
  out[13] = y;
  out[14] = z;
  out[15] = 1;
  return out;
}

/** 构造缩放矩阵（列主序），写入 out。 */
export function scaling(out: Mat4, x: number, y: number, z: number): Mat4 {
  out.fill(0);
  out[0] = x;
  out[5] = y;
  out[10] = z;
  out[15] = 1;
  return out;
}

/** 绕 X 轴旋转 θ（弧度）的列主序矩阵，写入 out。 */
export function rotationX(out: Mat4, theta: number): Mat4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  out.fill(0);
  out[0] = 1;
  out[5] = c;
  out[6] = s;
  out[9] = -s;
  out[10] = c;
  out[15] = 1;
  return out;
}

/** 绕 Z 轴旋转 θ（弧度）的列主序矩阵，写入 out。 */
export function rotationZ(out: Mat4, theta: number): Mat4 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  out.fill(0);
  out[0] = c;
  out[1] = s;
  out[4] = -s;
  out[5] = c;
  out[10] = 1;
  out[15] = 1;
  return out;
}