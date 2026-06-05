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
// Author(s):  James Stanard
//

#pragma once

#include <d3d12.h>

class GraphicsContext;
class ShadowCamera;
class ModelH3D;
class ExpVar;

namespace Math
{
    class Camera;
    class Vector3;
    class Matrix4;
}

namespace Sponza
{
    void Startup( Math::Camera& camera );
    void Cleanup( void );

    void RenderScene(
        GraphicsContext& gfxContext,
        const Math::Camera& camera,
        const D3D12_VIEWPORT& viewport,
        const D3D12_RECT& scissor,
        bool skipDiffusePass = false,
        bool skipShadowMap = false );

    const ModelH3D& GetModel();

    extern Math::Vector3 m_SunDirection;
    extern Math::Matrix4 m_ModelTransform;
    extern ShadowCamera m_SunShadow;
    extern ExpVar m_AmbientIntensity;
    extern ExpVar m_SunLightIntensity;
    extern Math::Vector3 m_PointLightPos;
    extern Math::Vector3 m_PointLightColor;

    // -------------------------------------------------------------------------
    // Procedural scene geometry (floor, walls, box occluder).
    // Used for both rasterization draw calls AND raytracing BLAS construction.
    // Vertex layout: position(3f) texcoord(2f) normal(3f) tangent(3f) bitangent(3f) = 56 bytes.
    // -------------------------------------------------------------------------
    struct SceneVertex
    {
        float position[3];   // offset  0
        float texcoord[2];   // offset 12
        float normal[3];     // offset 20
        float tangent[3];    // offset 32
        float bitangent[3];  // offset 44
    };
    static_assert(sizeof(SceneVertex) == 56, "SceneVertex must be 56 bytes");

    // Descriptor used by ModelViewer.cpp to build BLAS entries.
    struct ProceduralSurfaceDesc
    {
        D3D12_GPU_VIRTUAL_ADDRESS vertexBufferVA;  // position at byte offset 0, stride 56
        D3D12_GPU_VIRTUAL_ADDRESS indexBufferVA;
        UINT                      vertexCount;
        UINT                      indexCount;
        UINT                      materialID;   // ≥100: handled procedurally in hit shader
    };

    UINT                 GetNumProceduralSurfaces();
    ProceduralSurfaceDesc GetProceduralSurface(UINT index);
}
