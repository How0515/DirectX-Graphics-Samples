//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author(s):  Alex Nankervis
//             James Stanard
//

// From Core
#include "GraphicsCore.h"
#include "BufferManager.h"
#include "Camera.h"
#include "CommandContext.h"
#include "TemporalEffects.h"
#include "SSAO.h"
#include "SystemTime.h"
#include "ShadowCamera.h"
#include "ParticleEffects.h"
#include "SponzaRenderer.h"
#include "Renderer.h"

// From Model
#include "ModelH3D.h"

// From ModelViewer
#include "LightManager.h"

#include "CompiledShaders/DepthViewerVS.h"
#include "CompiledShaders/DepthViewerPS.h"
#include "CompiledShaders/ModelViewerVS.h"
#include "CompiledShaders/ModelViewerPS.h"

using namespace Math;
using namespace Graphics;

namespace Sponza
{
    void RenderLightShadows(GraphicsContext& gfxContext, const Camera& camera);

    enum eObjectFilter { kOpaque = 0x1, kCutout = 0x2, kTransparent = 0x4, kAll = 0xF, kNone = 0x0 };
    void RenderObjects( GraphicsContext& Context, const Matrix4& ViewProjMat, const Vector3& viewerPos, eObjectFilter Filter = kAll );

    GraphicsPSO m_DepthPSO = { (L"Sponza: Depth PSO") };
    GraphicsPSO m_CutoutDepthPSO = { (L"Sponza: Cutout Depth PSO") };
    GraphicsPSO m_ModelPSO = { (L"Sponza: Color PSO") };
    GraphicsPSO m_CutoutModelPSO = { (L"Sponza: Cutout Color PSO") };
    GraphicsPSO m_ShadowPSO(L"Sponza: Shadow PSO");
    GraphicsPSO m_CutoutShadowPSO(L"Sponza: Cutout Shadow PSO");

    ModelH3D m_Model;
    std::vector<bool> m_pMaterialIsCutout;

    Vector3 m_SunDirection;
    ShadowCamera m_SunShadow;

    ExpVar m_AmbientIntensity("Sponza/Lighting/Ambient Intensity", 0.1f, -16.0f, 16.0f, 0.1f);
    ExpVar m_SunLightIntensity("Sponza/Lighting/Sun Light Intensity", 4.0f, 0.0f, 16.0f, 0.1f);
    NumVar m_SunOrientation("Sponza/Lighting/Sun Orientation", -0.5f, -100.0f, 100.0f, 0.1f );
    NumVar m_SunInclination("Sponza/Lighting/Sun Inclination", 0.75f, 0.0f, 1.0f, 0.01f );
    NumVar ShadowDimX("Sponza/Lighting/Shadow Dim X", 5000, 1000, 10000, 100 );
    NumVar ShadowDimY("Sponza/Lighting/Shadow Dim Y", 3000, 1000, 10000, 100 );
    NumVar ShadowDimZ("Sponza/Lighting/Shadow Dim Z", 3000, 1000, 10000, 100 );

    // ---- Procedural scene geometry (floor, walls, box) ----------------------
    static const UINT kNumProcSurfaces = 5;

    struct ProcSurface
    {
        ByteAddressBuffer vb;         // 56-byte stride, position at offset 0
        ByteAddressBuffer ib;         // 16-bit indices
        UINT              vertexCount;
        UINT              indexCount;
        UINT              materialID; // ≥100 → handled in hit shader
        DescriptorHandle  srvHandle;  // 6 consecutive entries in Renderer::s_TextureHeap
    };

    ProcSurface            m_procSurfaces[kNumProcSurfaces];
    ComPtr<ID3D12Resource> m_procDiffuseTextures[kNumProcSurfaces];
}

// Creates a 1×1 R8G8B8A8_UNORM texture with a solid colour and writes its SRV.
// The resource is kept alive via outTexture (caller must keep it alive).
static void CreateSolidColorTextureSRV(
    D3D12_CPU_DESCRIPTOR_HANDLE destSRV,
    float r, float g, float b,
    ComPtr<ID3D12Resource>& outTexture)
{
    using namespace Graphics;

    uint8_t rgba[4] = {
        static_cast<uint8_t>(r * 255.f + 0.5f),
        static_cast<uint8_t>(g * 255.f + 0.5f),
        static_cast<uint8_t>(b * 255.f + 0.5f),
        255u
    };

    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width            = 1;
    texDesc.Height           = 1;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels        = 1;
    texDesc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    D3D12_HEAP_PROPERTIES defHeap = {}; defHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    g_Device->CreateCommittedResource(&defHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&outTexture));

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    UINT numRows; UINT64 rowBytes, totalBytes;
    g_Device->GetCopyableFootprints(&texDesc, 0, 1, 0, &fp, &numRows, &rowBytes, &totalBytes);

    D3D12_HEAP_PROPERTIES upHeap = {}; upHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC   upDesc = {};
    upDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    upDesc.Width            = totalBytes;
    upDesc.Height           = 1; upDesc.DepthOrArraySize = 1; upDesc.MipLevels = 1;
    upDesc.SampleDesc.Count = 1; upDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> upBuf;
    g_Device->CreateCommittedResource(&upHeap, D3D12_HEAP_FLAG_NONE, &upDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upBuf));

    void* pMap = nullptr;
    upBuf->Map(0, nullptr, &pMap);
    memcpy(static_cast<uint8_t*>(pMap) + fp.Offset, rgba, 4);
    upBuf->Unmap(0, nullptr);

    GraphicsContext& ctx = GraphicsContext::Begin(L"SolidColorTexUpload");
    D3D12_TEXTURE_COPY_LOCATION dst = {outTexture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    D3D12_TEXTURE_COPY_LOCATION src = {upBuf.Get(),       D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,  {fp}};
    ctx.GetCommandList()->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        outTexture.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    ctx.GetCommandList()->ResourceBarrier(1, &barrier);
    ctx.Finish(true);  // wait for GPU to finish before upBuf goes out of scope

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension           = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels     = 1;
    g_Device->CreateShaderResourceView(outTexture.Get(), &srvDesc, destSRV);
}

void Sponza::Startup( Camera& Camera )
{
    DXGI_FORMAT ColorFormat = g_SceneColorBuffer.GetFormat();
    DXGI_FORMAT NormalFormat = g_SceneNormalBuffer.GetFormat();
    DXGI_FORMAT DepthFormat = g_SceneDepthBuffer.GetFormat();
    //DXGI_FORMAT ShadowFormat = g_ShadowBuffer.GetFormat();

    D3D12_INPUT_ELEMENT_DESC vertElem[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "BITANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };

    // Depth-only (2x rate)
    m_DepthPSO.SetRootSignature(Renderer::m_RootSig);
    m_DepthPSO.SetRasterizerState(RasterizerDefault);
    m_DepthPSO.SetBlendState(BlendNoColorWrite);
    m_DepthPSO.SetDepthStencilState(DepthStateReadWrite);
    m_DepthPSO.SetInputLayout(_countof(vertElem), vertElem);
    m_DepthPSO.SetPrimitiveTopologyType(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    m_DepthPSO.SetRenderTargetFormats(0, nullptr, DepthFormat);
    m_DepthPSO.SetVertexShader(g_pDepthViewerVS, sizeof(g_pDepthViewerVS));
    m_DepthPSO.Finalize();

    // Depth-only shading but with alpha testing
    m_CutoutDepthPSO = m_DepthPSO;
    m_CutoutDepthPSO.SetPixelShader(g_pDepthViewerPS, sizeof(g_pDepthViewerPS));
    m_CutoutDepthPSO.SetRasterizerState(RasterizerTwoSided);
    m_CutoutDepthPSO.Finalize();

    // Depth-only but with a depth bias and/or render only backfaces
    m_ShadowPSO = m_DepthPSO;
    m_ShadowPSO.SetRasterizerState(RasterizerShadow);
    m_ShadowPSO.SetRenderTargetFormats(0, nullptr, g_ShadowBuffer.GetFormat());
    m_ShadowPSO.Finalize();

    // Shadows with alpha testing
    m_CutoutShadowPSO = m_ShadowPSO;
    m_CutoutShadowPSO.SetPixelShader(g_pDepthViewerPS, sizeof(g_pDepthViewerPS));
    m_CutoutShadowPSO.SetRasterizerState(RasterizerShadowTwoSided);
    m_CutoutShadowPSO.Finalize();

    DXGI_FORMAT formats[2] = { ColorFormat, NormalFormat };

    // Full color pass
    m_ModelPSO = m_DepthPSO;
    m_ModelPSO.SetBlendState(BlendDisable);
    m_ModelPSO.SetDepthStencilState(DepthStateTestEqual);
    m_ModelPSO.SetRenderTargetFormats(2, formats, DepthFormat);
    m_ModelPSO.SetVertexShader( g_pModelViewerVS, sizeof(g_pModelViewerVS) );
    m_ModelPSO.SetPixelShader( g_pModelViewerPS, sizeof(g_pModelViewerPS) );
    m_ModelPSO.Finalize();

    m_CutoutModelPSO = m_ModelPSO;
    m_CutoutModelPSO.SetRasterizerState(RasterizerTwoSided);
    m_CutoutModelPSO.Finalize();

    ASSERT(m_Model.Load(L"FlightHelmet/FlightHelmet.h3d"), "Failed to load model");
    ASSERT(m_Model.GetMeshCount() > 0, "Model contains no meshes");

    // The caller of this function can override which materials are considered cutouts
    m_pMaterialIsCutout.resize(m_Model.GetMaterialCount());
    for (uint32_t i = 0; i < m_Model.GetMaterialCount(); ++i)
    {
        const ModelH3D::Material& mat = m_Model.GetMaterial(i);
        if (std::string(mat.texDiffusePath).find("thorn") != std::string::npos ||
            std::string(mat.texDiffusePath).find("plant") != std::string::npos ||
            std::string(mat.texDiffusePath).find("chain") != std::string::npos)
        {
            m_pMaterialIsCutout[i] = true;
        }
        else
        {
            m_pMaterialIsCutout[i] = false;
        }
    }

    ParticleEffects::InitFromJSON(L"Sponza/particles.json");

    float modelRadius = Length(m_Model.GetBoundingBox().GetDimensions()) * 0.5f;
    const Vector3 eye = m_Model.GetBoundingBox().GetCenter() + Vector3(modelRadius * 0.5f, 0.0f, 0.0f);
    Camera.SetEyeAtUp( eye, Vector3(kZero), Vector3(kYUnitVector) );

    Lighting::CreateRandomLights(m_Model.GetBoundingBox().GetMin(), m_Model.GetBoundingBox().GetMax());

    // ---- Build procedural scene geometry ------------------------------------
    {
        using namespace Graphics;

        const float kFloorY   = -0.05f;
        const float kRoomHalf =  1.5f;
        const float kRoomTop  =  2.0f;
        const float kBoxXMin  =  0.44f, kBoxXMax = 0.56f;
        const float kBoxYMin  = kFloorY, kBoxYMax = 0.45f;
        const float kBoxZMin  = -0.04f, kBoxZMax = 0.04f;

        // Surface diffuse colours (must match DiffuseHitShaderLib.hlsl's GetProceduralColor).
        struct SurfCol { float r, g, b; } cols[kNumProcSurfaces] = {
            {0.725f, 0.710f, 0.680f},   // floor
            {0.630f, 0.065f, 0.050f},   // red wall
            {0.140f, 0.450f, 0.091f},   // green wall
            {0.850f, 0.850f, 0.850f},   // white back wall
            {0.650f, 0.650f, 0.650f},   // box
        };
        const UINT matIDs[kNumProcSurfaces] = {100, 101, 102, 103, 104};

        // FillQuad: writes 4 vertices (pos+normal+tangent+bitangent) and 6 indices.
        auto FillQuad = [](
            SceneVertex* v, uint16_t* idx, uint16_t base,
            const float p0[3], const float p1[3],
            const float p2[3], const float p3[3],
            const float n[3],  const float t[3], const float bt[3])
        {
            for (int k = 0; k < 4; ++k) {
                memset(&v[k], 0, sizeof(SceneVertex));
                memcpy(v[k].normal,    n,  12);
                memcpy(v[k].tangent,   t,  12);
                memcpy(v[k].bitangent, bt, 12);
            }
            memcpy(v[0].position, p0, 12); memcpy(v[1].position, p1, 12);
            memcpy(v[2].position, p2, 12); memcpy(v[3].position, p3, 12);
            idx[0]=base; idx[1]=base+1; idx[2]=base+2;
            idx[3]=base; idx[4]=base+2; idx[5]=base+3;
        };

        SceneVertex tmpV[20] = {};
        uint16_t    tmpI[30] = {};
        UINT vcnt[kNumProcSurfaces], icnt[kNumProcSurfaces];

        // 0 – Floor (n=0,1,0)
        { const float p0[3]={-kRoomHalf,kFloorY,-kRoomHalf}, p1[3]={ kRoomHalf,kFloorY,-kRoomHalf},
                      p2[3]={ kRoomHalf,kFloorY, kRoomHalf},  p3[3]={-kRoomHalf,kFloorY, kRoomHalf};
          const float n[3]={0,1,0}, t[3]={1,0,0}, bt[3]={0,0,-1};
          FillQuad(tmpV, tmpI, 0, p0,p1,p2,p3, n,t,bt);
          vcnt[0]=4; icnt[0]=6;
          m_procSurfaces[0].vb.Create(L"ProcFloor VB", 4, sizeof(SceneVertex), tmpV);
          m_procSurfaces[0].ib.Create(L"ProcFloor IB", 6, sizeof(uint16_t),    tmpI); }

        // 1 – Red wall  (X=-kRoomHalf, n=+1,0,0)
        { const float p0[3]={-kRoomHalf,kFloorY,-kRoomHalf}, p1[3]={-kRoomHalf,kFloorY, kRoomHalf},
                      p2[3]={-kRoomHalf,kRoomTop, kRoomHalf}, p3[3]={-kRoomHalf,kRoomTop,-kRoomHalf};
          const float n[3]={1,0,0}, t[3]={0,1,0}, bt[3]={0,0,1};
          FillQuad(tmpV, tmpI, 0, p0,p1,p2,p3, n,t,bt);
          vcnt[1]=4; icnt[1]=6;
          m_procSurfaces[1].vb.Create(L"ProcRedWall VB", 4, sizeof(SceneVertex), tmpV);
          m_procSurfaces[1].ib.Create(L"ProcRedWall IB", 6, sizeof(uint16_t),    tmpI); }

        // 2 – Green wall (X=+kRoomHalf, n=-1,0,0)
        { const float p0[3]={ kRoomHalf,kFloorY, kRoomHalf},  p1[3]={ kRoomHalf,kFloorY,-kRoomHalf},
                      p2[3]={ kRoomHalf,kRoomTop,-kRoomHalf},  p3[3]={ kRoomHalf,kRoomTop, kRoomHalf};
          const float n[3]={-1,0,0}, t[3]={0,1,0}, bt[3]={0,0,-1};
          FillQuad(tmpV, tmpI, 0, p0,p1,p2,p3, n,t,bt);
          vcnt[2]=4; icnt[2]=6;
          m_procSurfaces[2].vb.Create(L"ProcGreenWall VB", 4, sizeof(SceneVertex), tmpV);
          m_procSurfaces[2].ib.Create(L"ProcGreenWall IB", 6, sizeof(uint16_t),    tmpI); }

        // 3 – White back wall (Z=+kRoomHalf, n=0,0,-1)
        { const float p0[3]={-kRoomHalf,kFloorY, kRoomHalf},  p1[3]={ kRoomHalf,kFloorY, kRoomHalf},
                      p2[3]={ kRoomHalf,kRoomTop, kRoomHalf},  p3[3]={-kRoomHalf,kRoomTop, kRoomHalf};
          const float n[3]={0,0,-1}, t[3]={1,0,0}, bt[3]={0,1,0};
          FillQuad(tmpV, tmpI, 0, p0,p1,p2,p3, n,t,bt);
          vcnt[3]=4; icnt[3]=6;
          m_procSurfaces[3].vb.Create(L"ProcWhiteWall VB", 4, sizeof(SceneVertex), tmpV);
          m_procSurfaces[3].ib.Create(L"ProcWhiteWall IB", 6, sizeof(uint16_t),    tmpI); }

        // 4 – Thin box occluder (5 faces, one BLAS)
        { memset(tmpV, 0, sizeof(tmpV)); memset(tmpI, 0, sizeof(tmpI));
          { const float p0[3]={kBoxXMin,kBoxYMax,kBoxZMin}, p1[3]={kBoxXMax,kBoxYMax,kBoxZMin},
                        p2[3]={kBoxXMax,kBoxYMax,kBoxZMax},  p3[3]={kBoxXMin,kBoxYMax,kBoxZMax};
            const float n[3]={0,1,0},  t[3]={1,0,0},  bt[3]={0,0,-1};
            FillQuad(tmpV+0, tmpI+0, 0, p0,p1,p2,p3, n,t,bt); }     // top
          { const float p0[3]={kBoxXMin,kBoxYMax,kBoxZMin}, p1[3]={kBoxXMin,kBoxYMin,kBoxZMin},
                        p2[3]={kBoxXMax,kBoxYMin,kBoxZMin},  p3[3]={kBoxXMax,kBoxYMax,kBoxZMin};
            const float n[3]={0,0,-1}, t[3]={1,0,0},  bt[3]={0,1,0};
            FillQuad(tmpV+4, tmpI+6, 4, p0,p1,p2,p3, n,t,bt); }     // front
          { const float p0[3]={kBoxXMax,kBoxYMax,kBoxZMax}, p1[3]={kBoxXMax,kBoxYMin,kBoxZMax},
                        p2[3]={kBoxXMin,kBoxYMin,kBoxZMax},  p3[3]={kBoxXMin,kBoxYMax,kBoxZMax};
            const float n[3]={0,0,1},  t[3]={-1,0,0}, bt[3]={0,1,0};
            FillQuad(tmpV+8,  tmpI+12, 8,  p0,p1,p2,p3, n,t,bt); }  // back
          { const float p0[3]={kBoxXMin,kBoxYMax,kBoxZMax}, p1[3]={kBoxXMin,kBoxYMin,kBoxZMax},
                        p2[3]={kBoxXMin,kBoxYMin,kBoxZMin},  p3[3]={kBoxXMin,kBoxYMax,kBoxZMin};
            const float n[3]={-1,0,0}, t[3]={0,1,0},  bt[3]={0,0,-1};
            FillQuad(tmpV+12, tmpI+18, 12, p0,p1,p2,p3, n,t,bt); }  // left
          { const float p0[3]={kBoxXMax,kBoxYMax,kBoxZMin}, p1[3]={kBoxXMax,kBoxYMin,kBoxZMin},
                        p2[3]={kBoxXMax,kBoxYMin,kBoxZMax},  p3[3]={kBoxXMax,kBoxYMax,kBoxZMax};
            const float n[3]={1,0,0},  t[3]={0,1,0},  bt[3]={0,0,1};
            FillQuad(tmpV+16, tmpI+24, 16, p0,p1,p2,p3, n,t,bt); }  // right
          vcnt[4]=20; icnt[4]=30;
          m_procSurfaces[4].vb.Create(L"ProcBox VB", 20, sizeof(SceneVertex), tmpV);
          m_procSurfaces[4].ib.Create(L"ProcBox IB", 30, sizeof(uint16_t),    tmpI); }

        // Finalise per-surface metadata and allocate SRV descriptors.
        UINT descSize = Renderer::s_TextureHeap.GetDescriptorSize();
        D3D12_CPU_DESCRIPTOR_HANDLE defaultSRVs[5] = {
            GetDefaultTexture(kBlackOpaque2D),       // t1: specular (none)
            GetDefaultTexture(kBlackTransparent2D),  // t2: emissive (none)
            GetDefaultTexture(kDefaultNormalMap),    // t3: normal (flat)
            GetDefaultTexture(kBlackTransparent2D),  // t4: lightmap (none)
            GetDefaultTexture(kBlackCubeMap),         // t5: reflection (none)
        };
        UINT oneCounts[5] = {1,1,1,1,1};
        UINT five = 5;

        for (UINT i = 0; i < kNumProcSurfaces; ++i)
        {
            m_procSurfaces[i].vertexCount = vcnt[i];
            m_procSurfaces[i].indexCount  = icnt[i];
            m_procSurfaces[i].materialID  = matIDs[i];

            // Allocate 6 consecutive descriptors: [0]=diffuse … [5]=reflection
            DescriptorHandle hBase = Renderer::s_TextureHeap.Alloc(6);
            m_procSurfaces[i].srvHandle = hBase;

            // Slot 0: solid-colour diffuse
            D3D12_CPU_DESCRIPTOR_HANDLE hSlot0 = hBase;
            CreateSolidColorTextureSRV(hSlot0, cols[i].r, cols[i].g, cols[i].b,
                m_procDiffuseTextures[i]);

            // Slots 1-5: defaults (copy 5 descriptors in one call)
            D3D12_CPU_DESCRIPTOR_HANDLE hSlot1 = hBase + descSize;
            g_Device->CopyDescriptors(1, &hSlot1, &five,
                5, defaultSRVs, oneCounts, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
    }
}

const ModelH3D& Sponza::GetModel()
{
    return Sponza::m_Model;
}

void Sponza::Cleanup( void )
{
    m_Model.Clear();
    Lighting::Shutdown();
    TextureManager::Shutdown();
}

void Sponza::RenderObjects( GraphicsContext& gfxContext, const Matrix4& ViewProjMat, const Vector3& viewerPos, eObjectFilter Filter )
{
    struct VSConstants
    {
        Matrix4 modelToProjection;
        Matrix4 modelToShadow;
        XMFLOAT3 viewerPos;
    } vsConstants;
    vsConstants.modelToProjection = ViewProjMat;
    vsConstants.modelToShadow = m_SunShadow.GetShadowMatrix();
    XMStoreFloat3(&vsConstants.viewerPos, viewerPos);

    gfxContext.SetDynamicConstantBufferView(Renderer::kMeshConstants, sizeof(vsConstants), &vsConstants);

    __declspec(align(16)) uint32_t materialIdx = 0xFFFFFFFFul;

    uint32_t VertexStride = m_Model.GetVertexStride();

    for (uint32_t meshIndex = 0; meshIndex < m_Model.GetMeshCount(); meshIndex++)
    {
        const ModelH3D::Mesh& mesh = m_Model.GetMesh(meshIndex);

        uint32_t indexCount = mesh.indexCount;
        uint32_t startIndex = mesh.indexDataByteOffset / sizeof(uint16_t);
        uint32_t baseVertex = mesh.vertexDataByteOffset / VertexStride;

        if (mesh.materialIndex != materialIdx)
        {
            if ( m_pMaterialIsCutout[mesh.materialIndex] && !(Filter & kCutout) ||
                !m_pMaterialIsCutout[mesh.materialIndex] && !(Filter & kOpaque) )
                continue;

            materialIdx = mesh.materialIndex;
            gfxContext.SetDescriptorTable(Renderer::kMaterialSRVs, m_Model.GetSRVs(materialIdx));

            gfxContext.SetDynamicConstantBufferView(Renderer::kCommonCBV, sizeof(uint32_t), &materialIdx);
        }

        gfxContext.DrawIndexed(indexCount, startIndex, baseVertex);
    }
}

// Draws all procedural surfaces using whatever PSO is currently bound.
// withMaterials=true → also binds per-surface SRV table + kCommonCBV (for colour pass).
// withMaterials=false → depth/shadow pass (no material descriptors needed).
static void RenderProceduralSurfaces(GraphicsContext& ctx, bool withMaterials)
{
    using namespace Sponza;
    for (UINT i = 0; i < kNumProcSurfaces; ++i)
    {
        auto& s = m_procSurfaces[i];
        ctx.SetVertexBuffer(0, s.vb.VertexBufferView(0, s.vertexCount * (UINT)sizeof(SceneVertex), sizeof(SceneVertex)));
        ctx.SetIndexBuffer(s.ib.IndexBufferView(0, s.indexCount * (UINT)sizeof(uint16_t), false));
        if (withMaterials)
        {
            ctx.SetDescriptorTable(Renderer::kMaterialSRVs, s.srvHandle);
            uint32_t matIdx = 0;
            ctx.SetDynamicConstantBufferView(Renderer::kCommonCBV, sizeof(uint32_t), &matIdx);
        }
        ctx.DrawIndexed(s.indexCount, 0, 0);
    }
}

UINT Sponza::GetNumProceduralSurfaces()  { return kNumProcSurfaces; }

Sponza::ProceduralSurfaceDesc Sponza::GetProceduralSurface(UINT index)
{
    ASSERT(index < kNumProcSurfaces);
    auto& s = m_procSurfaces[index];
    return { s.vb.GetGpuVirtualAddress(), s.ib.GetGpuVirtualAddress(),
             s.vertexCount, s.indexCount, s.materialID };
}

void Sponza::RenderLightShadows(GraphicsContext& gfxContext, const Camera& camera)
{
    using namespace Lighting;

    ScopedTimer _prof(L"RenderLightShadows", gfxContext);

    static uint32_t LightIndex = 0;
    if (LightIndex >= MaxLights)
        return;

    m_LightShadowTempBuffer.BeginRendering(gfxContext);
    {
        gfxContext.SetPipelineState(m_ShadowPSO);
        RenderObjects(gfxContext, m_LightShadowMatrix[LightIndex], camera.GetPosition(), kOpaque);
        gfxContext.SetPipelineState(m_CutoutShadowPSO);
        RenderObjects(gfxContext, m_LightShadowMatrix[LightIndex], camera.GetPosition(), kCutout);
    }
    //m_LightShadowTempBuffer.EndRendering(gfxContext);

    gfxContext.TransitionResource(m_LightShadowTempBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE);
    gfxContext.TransitionResource(m_LightShadowArray, D3D12_RESOURCE_STATE_COPY_DEST);

    gfxContext.CopySubresource(m_LightShadowArray, LightIndex, m_LightShadowTempBuffer, 0);

    gfxContext.TransitionResource(m_LightShadowArray, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    ++LightIndex;
}

void Sponza::RenderScene(
    GraphicsContext& gfxContext,
    const Camera& camera,
    const D3D12_VIEWPORT& viewport,
    const D3D12_RECT& scissor,
    bool skipDiffusePass,
    bool skipShadowMap)
{
    Renderer::UpdateGlobalDescriptors();

    uint32_t FrameIndex = TemporalEffects::GetFrameIndexMod2();

    float costheta = cosf(m_SunOrientation);
    float sintheta = sinf(m_SunOrientation);
    float cosphi = cosf(m_SunInclination * 3.14159f * 0.5f);
    float sinphi = sinf(m_SunInclination * 3.14159f * 0.5f);
    m_SunDirection = Normalize(Vector3( costheta * cosphi, sinphi, sintheta * cosphi ));

    __declspec(align(16)) struct
    {
        Vector3 sunDirection;
        Vector3 sunLight;
        Vector3 ambientLight;
        float ShadowTexelSize[4];

        float InvTileDim[4];
        uint32_t TileCount[4];
        uint32_t FirstLightIndex[4];

		uint32_t FrameIndexMod2;
    } psConstants;

    psConstants.sunDirection = m_SunDirection;
    psConstants.sunLight = Vector3(1.0f, 1.0f, 1.0f) * m_SunLightIntensity;
    psConstants.ambientLight = Vector3(1.0f, 1.0f, 1.0f) * m_AmbientIntensity;
    psConstants.ShadowTexelSize[0] = 1.0f / g_ShadowBuffer.GetWidth();
    psConstants.InvTileDim[0] = 1.0f / Lighting::LightGridDim;
    psConstants.InvTileDim[1] = 1.0f / Lighting::LightGridDim;
    psConstants.TileCount[0] = Math::DivideByMultiple(g_SceneColorBuffer.GetWidth(), Lighting::LightGridDim);
    psConstants.TileCount[1] = Math::DivideByMultiple(g_SceneColorBuffer.GetHeight(), Lighting::LightGridDim);
    psConstants.FirstLightIndex[0] = Lighting::m_FirstConeLight;
    psConstants.FirstLightIndex[1] = Lighting::m_FirstConeShadowedLight;
	psConstants.FrameIndexMod2 = FrameIndex;

    // Set the default state for command lists
    auto& pfnSetupGraphicsState = [&](void)
    {
        gfxContext.SetRootSignature(Renderer::m_RootSig);
        gfxContext.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, Renderer::s_TextureHeap.GetHeapPointer());
        gfxContext.SetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        gfxContext.SetIndexBuffer(m_Model.GetIndexBuffer());
        gfxContext.SetVertexBuffer(0, m_Model.GetVertexBuffer());
    };

    pfnSetupGraphicsState();

    RenderLightShadows(gfxContext, camera);

    {
        ScopedTimer _prof(L"Z PrePass", gfxContext);

        gfxContext.SetDynamicConstantBufferView(Renderer::kMaterialConstants, sizeof(psConstants), &psConstants);

        {
            ScopedTimer _prof2(L"Opaque", gfxContext);
            {
                gfxContext.TransitionResource(g_SceneDepthBuffer, D3D12_RESOURCE_STATE_DEPTH_WRITE, true);
                gfxContext.ClearDepth(g_SceneDepthBuffer);
                gfxContext.SetPipelineState(m_DepthPSO);
                gfxContext.SetDepthStencilTarget(g_SceneDepthBuffer.GetDSV());
                gfxContext.SetViewportAndScissor(viewport, scissor);
            }
            RenderObjects(gfxContext, camera.GetViewProjMatrix(), camera.GetPosition(), kOpaque );
        }

        {
            ScopedTimer _prof2(L"Cutout", gfxContext);
            {
                gfxContext.SetPipelineState(m_CutoutDepthPSO);
            }
            RenderObjects(gfxContext, camera.GetViewProjMatrix(), camera.GetPosition(), kCutout );
        }
        // Procedural surfaces: contribute to depth prepass (for SSAO)
        gfxContext.SetPipelineState(m_DepthPSO);
        RenderProceduralSurfaces(gfxContext, false);
    }

    SSAO::Render(gfxContext, camera);

    if (!skipDiffusePass)
    {
        Lighting::FillLightGrid(gfxContext, camera);

        if (!SSAO::DebugDraw)
        {
            ScopedTimer _prof(L"Main Render", gfxContext);
            {
                gfxContext.TransitionResource(g_SceneColorBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);
                gfxContext.TransitionResource(g_SceneNormalBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, true);
                gfxContext.ClearColor(g_SceneColorBuffer);
            }
        }
    }

    if (!skipShadowMap)
    {
        if (!SSAO::DebugDraw)
        {
            pfnSetupGraphicsState();
            {
                ScopedTimer _prof2(L"Render Shadow Map", gfxContext);

                m_SunShadow.UpdateMatrix(-m_SunDirection, Vector3(0, -500.0f, 0), Vector3(ShadowDimX, ShadowDimY, ShadowDimZ),
                    (uint32_t)g_ShadowBuffer.GetWidth(), (uint32_t)g_ShadowBuffer.GetHeight(), 16);

                g_ShadowBuffer.BeginRendering(gfxContext);
                gfxContext.SetPipelineState(m_ShadowPSO);
                RenderObjects(gfxContext, m_SunShadow.GetViewProjMatrix(), camera.GetPosition(), kOpaque);
                gfxContext.SetPipelineState(m_CutoutShadowPSO);
                RenderObjects(gfxContext, m_SunShadow.GetViewProjMatrix(), camera.GetPosition(), kCutout);
                // Procedural surfaces: cast shadows
                gfxContext.SetPipelineState(m_ShadowPSO);
                RenderProceduralSurfaces(gfxContext, false);
                g_ShadowBuffer.EndRendering(gfxContext);
            }
        }
    }

    if (!skipDiffusePass)
    {
        if (!SSAO::DebugDraw)
        {
            if (SSAO::AsyncCompute)
            {
                gfxContext.Flush();
                pfnSetupGraphicsState();

                // Make the 3D queue wait for the Compute queue to finish SSAO
                g_CommandManager.GetGraphicsQueue().StallForProducer(g_CommandManager.GetComputeQueue());
            }

            {
                ScopedTimer _prof2(L"Render Color", gfxContext);

                gfxContext.TransitionResource(g_SSAOFullScreen, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

                gfxContext.SetDescriptorTable(Renderer::kCommonSRVs, Renderer::m_CommonTextures);
                gfxContext.SetDynamicConstantBufferView(Renderer::kMaterialConstants, sizeof(psConstants), &psConstants);

                {
                    gfxContext.SetPipelineState(m_ModelPSO);
                    gfxContext.TransitionResource(g_SceneDepthBuffer, D3D12_RESOURCE_STATE_DEPTH_READ);
                    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[]{ g_SceneColorBuffer.GetRTV(), g_SceneNormalBuffer.GetRTV() };
                    gfxContext.SetRenderTargets(ARRAYSIZE(rtvs), rtvs, g_SceneDepthBuffer.GetDSV_DepthReadOnly());
                    gfxContext.SetViewportAndScissor(viewport, scissor);
                }
                RenderObjects( gfxContext, camera.GetViewProjMatrix(), camera.GetPosition(), Sponza::kOpaque );

                gfxContext.SetPipelineState(m_CutoutModelPSO);
                RenderObjects( gfxContext, camera.GetViewProjMatrix(), camera.GetPosition(), Sponza::kCutout );
                // Procedural surfaces: colour pass (floor, walls, box)
                gfxContext.SetPipelineState(m_ModelPSO);
                RenderProceduralSurfaces(gfxContext, true);
            }
        }
    }
}
