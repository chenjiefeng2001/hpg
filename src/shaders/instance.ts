/**
 * 统一实例数据模型 —— 用户 WGSL 只需面向这一套语义编写。
 *
 * 无论单物体（instanceCount=1）、自动合批（Instanced），还是未来的
 * GPU 剔除（Indirect），实例数据始终是 `array<InstanceData>`，
 * 通过 @builtin(instance_index) 索引读取 —— 三路策略对用户零感知。
 */

export const INSTANCE_LAYOUT_TEMPLATE = /* wgsl */ `
struct InstanceData {
    modelMatrix: mat4x4<f32>,
    color: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};
`;

/**
 * 材质绑定模板（固定 group=2），与 core/texture.ts 的 createMaterialBindGroupLayout 一一对应。
 *
 *   binding 0 → baseColorTexture（sRGB）
 *   binding 1 → baseColorSampler
 *   binding 2 → material uniform（baseColorFactor + alphaCutoff + alphaMode）
 *
 * alphaMode 常量与 ALPHA_MODE_CODE 一致：0=OPAQUE, 1=MASK, 2=BLEND。
 */
export const MATERIAL_LAYOUT_TEMPLATE = /* wgsl */ `
struct MaterialUniforms {
    baseColorFactor: vec4<f32>,
    alphaCutoff: f32,
    alphaMode: u32,
    _pad: vec2<f32>,
};

@group(2) @binding(0) var baseColorTexture: texture_2d<f32>;
@group(2) @binding(1) var baseColorSampler: sampler;
@group(2) @binding(2) var<uniform> material: MaterialUniforms;
`;

/**
 * 带 base color 贴图的实例顶点：输出 UV（location 0）与实例颜色（location 1）。
 * 需要 canonical 顶点布局（position/normal/uv/tangent 四个 location）。
 */
export const VS_INSTANCED_MATERIAL = /* wgsl */ `
struct InstanceData {
    modelMatrix: mat4x4<f32>,
    color: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;

struct MaterialVertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
) -> MaterialVertexOutput {
    let inst = instances[instanceIdx];
    var out: MaterialVertexOutput;
    out.clip = inst.modelMatrix * vec4<f32>(position, 1.0);
    out.uv = uv;
    out.color = inst.color;
    return out;
}
`;

/**
 * GPU Culled 路径的材质顶点着色器：通过 compactedIndices 间接索引实例数据。
 * 必须配合 registerPipeline({ compaction: true }) 使用。
 */
export const VS_INSTANCED_MATERIAL_COMPACTION = /* wgsl */ `
struct InstanceData {
    modelMatrix: mat4x4<f32>,
    color: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(1) @binding(1) var<storage, read> compactedIndices: array<u32>;

struct MaterialVertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
) -> MaterialVertexOutput {
    let mapped = compactedIndices[instanceIdx];
    let inst = instances[mapped];
    var out: MaterialVertexOutput;
    out.clip = inst.modelMatrix * vec4<f32>(position, 1.0);
    out.uv = uv;
    out.color = inst.color;
    return out;
}
`;

/**
 * 材质片段：采样 base color 贴图 × baseColorFactor × 实例颜色；MASK 时按 alphaCutoff discard。
 *
 * 注意：使用本着色器时，实例颜色应保持白色（baseColorFactor 由 material uniform 提供），
 * 否则颜色会被乘两次 —— sceneToRenderItems() 在附上材质时会自动写入白色。
 */
export const FS_MATERIAL = /* wgsl */ `
${MATERIAL_LAYOUT_TEMPLATE}

@fragment
fn fs_main(
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
) -> @location(0) vec4<f32> {
    let sampled = textureSample(baseColorTexture, baseColorSampler, uv);
    let base = color * material.baseColorFactor * sampled;
    if (material.alphaMode == 1u) {
        if (base.a < material.alphaCutoff) {
            discard;
        }
    }
    return base;
}
`;

/**
 * 标准彩色方块片段（带实例颜色）。
 */
export const FS_COLOR = /* wgsl */ `
@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
`;

/**
 * 展示用顶点：位置属性在 location(0)，来自实例矩阵做世界变换。
 */
export const VS_INSTANCED = /* wgsl */ `
${INSTANCE_LAYOUT_TEMPLATE}

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    let inst = instances[instanceIdx];
    let worldPos = inst.modelMatrix * vec4<f32>(position, 1.0);
    var out: VertexOutput;
    out.clip = worldPos;
    out.color = inst.color;
    return out;
}
`;

/**
 * 平面顶点：无实例颜色，使用 uniform 颜色（group=0 binding(1)）。
 * 适用于纯色几何、线框、UI 等不需要逐实例颜色的场景。
 */
export const VS_FLAT = /* wgsl */ `
struct Uniforms {
    color: vec4<f32>,
};

@group(0) @binding(1) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @location(0) position: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip = vec4<f32>(position, 1.0);
    out.color = uniforms.color;
    return out;
}
`;

/**
 * 深度预通道片段：仅写入深度，不输出颜色。
 * 配合 depthWriteEnabled: true 使用，提前填充 Z-Buffer 减少 overdraw。
 */
export const FS_DEPTH_ONLY = /* wgsl */ `
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
`;

/**
 * GPU Culling 路径的顶点着色器。
 *
 * 与 VS_INSTANCED 的区别：
 *   - 额外绑定 compactedIndices（group=1, binding=1）
 *   - 使用 compactedIndices[instanceIdx] 间接访问 instance 数据
 *   - 只绘制 visible instances（已被 compaction 重排）
 *
 * 使用方式：
 *   - registerPipeline({ vsCode: VS_INSTANCED_COMPACTION, ... })
 *   - bind group(1) 绑定 [instanceBuffer, compactedIndicesBuffer]
 */
export const VS_INSTANCED_COMPACTION = /* wgsl */ `
struct InstanceData {
    modelMatrix: mat4x4<f32>,
    color: vec4<f32>,
};

@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;
// Compaction mapping（binding 1 带 dynamic offset，指向本 geometry 的 slot 区）：
//   compactedIndices[compactedSlot] = 组内原始实例索引
@group(1) @binding(1) var<storage, read> compactedIndices: array<u32>;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(instance_index) instanceIdx: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
) -> VertexOutput {
    // 通过 compaction mapping 间接读取 instance 数据。
    // instanceIdx 是 GPU 的 draw instance 索引（0..visibleCount-1），
    // compactedIndices 将其映射回**本 geometry** 实例区内的原始位置
    // （instances / compactedIndices 两个 binding 都带 dynamic offset）。
    let mappedIdx = compactedIndices[instanceIdx];
    let inst = instances[mappedIdx];
    let worldPos = inst.modelMatrix * vec4<f32>(position, 1.0);
    var out: VertexOutput;
    out.clip = worldPos;
    out.color = inst.color;
    return out;
}
`;