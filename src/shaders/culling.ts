/**
 * GPU 视锥剔除（Frustum Culling）—— Compute Shader + Pipeline。
 *
 * Phase 6B+ 架构（全 GPU-driven，含 compaction）：
 *   CPU 上传 bounding spheres + geometry indices + 每个 geometry 的 base 表 + VP matrix
 *   Compute Shader:
 *     1. 测试每个 instance 的 bounding sphere
 *     2. 如果可见 → 原子递增对应 geometry 的 instanceCount
 *                    原子递增该 geometry 的 compaction slot
 *                    写入 compactedIndices[slotBase + slot] = 组内原始实例索引
 *   Render Pass:
 *     vertex shader 用 compactedIndices[instance_index]（binding 1 带 dynamic offset，
 *     偏移到该 geometry 的 slotBase）间接读 instance buffer
 *     → 只绘制可见 instances，且能正确映射回原始实例数据
 *
 * 关键索引空间约定：
 *   - spheres / geometryIndices 是全局候选索引（0..N-1，与实例 buffer 顺序一致）
 *   - compactedIndices 按 *slot* 空间写入：geometry g 占用 [slotBase_g, slotBase_g + n_g)，
 *     slotBase_g 由 CPU 计算并保证 256 字节对齐（供 dynamic offset 使用）
 *   - 写入的值是**组内**原始索引（idx - candidateBase_g），因为实例 buffer 绑定
 *     （binding 0 的 dynamic offset）已经偏移到该组的起始位置
 *
 * CPU 不读取任何 GPU 结果（零 readback）。
 */

/** Compute Shader: 视锥剔除 + compaction + 间接绘制参数生成 */
export const CS_FRUSTUM_CULL = /* wgsl */ `
struct CullUniforms {
    vp: mat4x4<f32>,
    sphereCount: u32,
    geometryCount: u32,
    _pad1: u32,
    _pad2: u32,
};

// DrawIndexedIndirectArgs 布局（20 字节）。
// instanceCount 由 compute shader 原子填充。
// 其余字段由 CPU 预写。
struct DrawArgs {
    indexCount: u32,
    instanceCount: atomic<u32>,
    firstIndex: u32,
    baseVertex: u32,
    firstInstance: u32,
};

@group(0) @binding(0) var<uniform> uniforms: CullUniforms;
@group(0) @binding(1) var<storage, read> spheres: array<vec4<f32>>;
// 每个 instance 所属的 geometryIndex（由 CPU 上传）。
@group(0) @binding(2) var<storage, read> geometryIndices: array<u32>;
// Compaction mapping：compactedIndices[slotBase + compactedSlot] = 组内原始索引。
// 不可见 instance 不占 slot（instanceCount 由原子计数决定，不会画到它们）。
@group(0) @binding(3) var<storage, read_write> compactedIndices: array<u32>;
// Indirect draw args：每个 geometry 一个 DrawIndexedIndirectArgs。
@group(0) @binding(4) var<storage, read_write> drawArgs: array<DrawArgs>;
// 原子 compaction 计数器：每个 geometry 一个，初始为 0。
@group(0) @binding(5) var<storage, read_write> compactionCounters: array<atomic<u32>>;
// 每个 geometry 的索引基址（由 CPU 计算）：
//   candidateBase: 该组第一个候选实例在全局候选数组中的下标
//   slotBase:      该组第一个 compaction slot 在 compactedIndices 中的下标（256B 对齐）
@group(0) @binding(6) var<storage, read> geometryBases: array<vec2<u32>>;

// Gribb-Hartmann 平面提取。
//
// 矩阵按列主序上传，WGSL 中 vp[i] 取的是第 i 列：
//     vp[c].r == M(r, c)
// 所以数学上的第 r 行 = vec4(vp[0][r], vp[1][r], vp[2][r], vp[3][r])。
//
// 平面 = sign * row + wRow，其中 wRow 是第 4 行（裁剪空间的 w 分量行）。
// 判定式 dot(plane.xyz, c) + plane.w + radius >= 0 表示球体在平面内侧。
//
// CPU 侧参考实现见 src/core/culling.ts 的 extractFrustumPlanes()（两者的数学必须一致）。
fn extractPlane(row: vec4<f32>, wRow: vec4<f32>, sign: f32) -> vec4<f32> {
    var p = sign * row + wRow;
    let len = length(p.xyz);
    if (len > 0.0) { p = p / len; }
    return p;
}

fn intersectSphere(plane: vec4<f32>, center: vec3<f32>, radius: f32) -> bool {
    return dot(plane.xyz, center) + plane.w + radius >= 0.0;
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.sphereCount) { return; }

    let s = spheres[idx];
    let center = s.xyz;
    let radius = s.w;

    // 取出 VP 的四行（注意 vp[i] 是列）。
    let rx = vec4<f32>(uniforms.vp[0].x, uniforms.vp[1].x, uniforms.vp[2].x, uniforms.vp[3].x);
    let ry = vec4<f32>(uniforms.vp[0].y, uniforms.vp[1].y, uniforms.vp[2].y, uniforms.vp[3].y);
    let rz = vec4<f32>(uniforms.vp[0].z, uniforms.vp[1].z, uniforms.vp[2].z, uniforms.vp[3].z);
    let rw = vec4<f32>(uniforms.vp[0].w, uniforms.vp[1].w, uniforms.vp[2].w, uniforms.vp[3].w);

    // 6 个平面：left/right = w ± x，bottom/top = w ± y，near/far = w ± z。
    // （WebGPU 裁剪空间 z ∈ [0, w]，故 near = w + z、far = w - z。）
    var visible = true;
    if (!intersectSphere(extractPlane(rx, rw,  1.0), center, radius)) { visible = false; }
    if (!intersectSphere(extractPlane(rx, rw, -1.0), center, radius)) { visible = false; }
    if (!intersectSphere(extractPlane(ry, rw,  1.0), center, radius)) { visible = false; }
    if (!intersectSphere(extractPlane(ry, rw, -1.0), center, radius)) { visible = false; }
    if (!intersectSphere(extractPlane(rz, rw,  1.0), center, radius)) { visible = false; }
    if (!intersectSphere(extractPlane(rz, rw, -1.0), center, radius)) { visible = false; }

    if (visible) {
        let geoIdx = geometryIndices[idx];
        let bases = geometryBases[geoIdx];
        // 原子递增该 geometry 的 instanceCount（indirect args）。
        atomicAdd(&drawArgs[geoIdx].instanceCount, 1u);
        // 原子递增 compaction slot，得到该 instance 在该 geometry slot 区中的位置。
        let slot = atomicAdd(&compactionCounters[geoIdx], 1u);
        // 写入 compaction mapping：slot → 组内原始索引
        // （组内：实例 buffer 的 binding 0 dynamic offset 已指向该组起始位置）。
        compactedIndices[bases.y + slot] = idx - bases.x;
    }
}
`;
