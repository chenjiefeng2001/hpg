/**
 * Material-heavy benchmark — simulates Blender scenes with many materials.
 * Tests: does material count affect parse or import time?
 */

import { parseGltf } from '../src/core/gltf';

let seed = 42;
function rand(): number {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
}

function generateGlb(meshCount: number, materialCount: number, vertsPerMesh: number): ArrayBuffer {
  const meshDescs: { verts: Float32Array; indices: Uint16Array; matIdx: number }[] = [];
  for (let m = 0; m < meshCount; m++) {
    const verts = new Float32Array(vertsPerMesh * 6);
    for (let v = 0; v < vertsPerMesh; v++) {
      verts[v*6] = (rand()-0.5)*0.5;
      verts[v*6+1] = (rand()-0.5)*0.5;
      verts[v*6+2] = (rand()-0.5)*0.5;
      const len = Math.sqrt(verts[v*6]**2 + verts[v*6+1]**2 + verts[v*6+2]**2) || 1;
      verts[v*6+3] = verts[v*6]/len; verts[v*6+4] = verts[v*6+1]/len; verts[v*6+5] = verts[v*6+2]/len;
    }
    const indices = new Uint16Array(vertsPerMesh);
    for (let i = 0; i < vertsPerMesh; i++) indices[i] = i;
    meshDescs.push({ verts, indices, matIdx: m % materialCount });
  }

  let totalBinSize = 0;
  const vertOff: number[] = [], idxOff: number[] = [];
  for (const md of meshDescs) {
    vertOff.push(totalBinSize); totalBinSize += md.verts.byteLength;
    totalBinSize = (totalBinSize + 3) & ~3;
    idxOff.push(totalBinSize); totalBinSize += md.indices.byteLength;
    totalBinSize = (totalBinSize + 3) & ~3;
  }

  const binData = new Uint8Array(totalBinSize);
  for (let i = 0; i < meshDescs.length; i++) {
    binData.set(new Uint8Array(meshDescs[i]!.verts.buffer), vertOff[i]!);
    binData.set(new Uint8Array(meshDescs[i]!.indices.buffer), idxOff[i]!);
  }

  const accessors: any[] = [], bufferViews: any[] = [], primAccessors: {pos:number;norm:number;idx:number}[] = [];
  let acc=0, bv=0;
  for (let i = 0; i < meshDescs.length; i++) {
    const md = meshDescs[i]!;
    bufferViews.push({buffer:0,byteOffset:vertOff[i],byteLength:md.verts.byteLength,byteStride:24,target:34962});
    const vbv=bv++;
    bufferViews.push({buffer:0,byteOffset:idxOff[i],byteLength:md.indices.byteLength,target:34963});
    const ibv=bv++;
    accessors.push({bufferView:vbv,byteOffset:0,componentType:5126,count:md.verts.length/6,type:'VEC3'}); const pa=acc++;
    accessors.push({bufferView:vbv,byteOffset:12,componentType:5126,count:md.verts.length/6,type:'VEC3'}); const na=acc++;
    accessors.push({bufferView:ibv,componentType:5123,count:md.indices.length,type:'SCALAR'}); const ia=acc++;
    primAccessors.push({pos:pa,norm:na,idx:ia});
  }

  const nodes = Array.from({length:meshCount}, (_,i) => ({name:`M${i}`,mesh:i,translation:[(rand()-0.5)*10,(rand()-0.5)*10,(rand()-0.5)*10]}));
  const materials = Array.from({length:materialCount}, (_,i) => ({
    name:`Mat${i}`, pbrMetallicRoughness:{baseColorFactor:[rand(),rand(),rand(),1],metallicFactor:rand(),roughnessFactor:rand()}
  }));

  const gltfJson = {
    asset:{version:'2.0',generator:'bench'}, scene:0,
    scenes:[{nodes:Array.from({length:meshCount},(_,i)=>i)}],
    nodes, materials,
    meshes:meshDescs.map((md,i)=>({name:`M${i}`,primitives:[{attributes:{POSITION:primAccessors[i]!.pos,NORMAL:primAccessors[i]!.norm},indices:primAccessors[i]!.idx,material:md.matIdx}]})),
    accessors, bufferViews, buffers:[{byteLength:totalBinSize}]
  };

  const jsonStr = JSON.stringify(gltfJson);
  const jsonBytes = new TextEncoder().encode(jsonStr);
  const jsonPad = (4-(jsonBytes.byteLength%4))%4;
  const jsonChunkLen = jsonBytes.byteLength + jsonPad;
  const binPad = (4-(totalBinSize%4))%4;
  const binChunkLen = totalBinSize + binPad;
  const totalLen = 12+8+jsonChunkLen+8+binChunkLen;

  const glb = new ArrayBuffer(totalLen);
  const dv = new DataView(glb);
  dv.setUint32(0,0x46546C67,true); dv.setUint32(4,2,true); dv.setUint32(8,totalLen,true);
  let off=12;
  dv.setUint32(off,jsonChunkLen,true); dv.setUint32(off+4,0x4E4F534A,true);
  const jd = new Uint8Array(glb,off+8,jsonChunkLen);
  jd.set(jsonBytes);
  for (let i=jsonBytes.byteLength;i<jsonChunkLen;i++) jd[i]=0x20;
  off+=8+jsonChunkLen;
  dv.setUint32(off,binChunkLen,true); dv.setUint32(off+4,0x004E4942,true);
  new Uint8Array(glb,off+8,binChunkLen).set(binData);
  return glb;
}

function median(a:number[]){const s=[...a].sort((x,y)=>x-y);return s[s.length>>1]??0;}

console.log('hpg — Material-Count Benchmark\n');
console.log('='.repeat(95));
console.log(
  'Config'.padEnd(40) +
  'GLB Size'.padStart(10) +
  'Meshes'.padStart(8) +
  'Mats'.padStart(6) +
  'Parse(ms)'.padStart(12) +
  'p95(ms)'.padStart(10)
);
console.log('-'.repeat(95));

const tests = [
  { label:'100 meshes, 1 mat',    mesh:100, mat:1,   verts:100 },
  { label:'100 meshes, 50 mat',   mesh:100, mat:50,  verts:100 },
  { label:'100 meshes, 100 mat',  mesh:100, mat:100, verts:100 },
  { label:'500 meshes, 1 mat',    mesh:500, mat:1,   verts:100 },
  { label:'500 meshes, 100 mat',  mesh:500, mat:100, verts:100 },
  { label:'500 meshes, 500 mat',  mesh:500, mat:500, verts:100 },
  { label:'1000 meshes, 1 mat',   mesh:1000,mat:1,   verts:100 },
  { label:'1000 meshes, 500 mat', mesh:1000,mat:500, verts:100 },
  { label:'1000 meshes, 1000 mat',mesh:1000,mat:1000,verts:100 },
];

for (const t of tests) {
  seed = 42;
  const glb = generateGlb(t.mesh, t.mat, t.verts);
  // warmup
  for (let w=0;w<3;w++) parseGltf(glb);
  const times: number[] = [];
  for (let s=0;s<20;s++) {
    const t0=performance.now(); parseGltf(glb); times.push(performance.now()-t0);
  }
  console.log(
    t.label.padEnd(40) +
    `${(glb.byteLength/1024).toFixed(0)} KB`.padStart(10) +
    String(t.mesh).padStart(8) +
    String(t.mat).padStart(6) +
    `${median(times).toFixed(2)}`.padStart(12) +
    `${times.sort((a,b)=>a-b)[Math.floor(times.length*0.95)].toFixed(2)}`.padStart(10)
  );
}
console.log('-'.repeat(95));
console.log('Conclusion: Does material count affect parse time independently of mesh/vertex count?');
