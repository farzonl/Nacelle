#pragma once

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <unknwn.h>
#include <directx/d3d12.h>
#include <directx/d3d12sdklayers.h>
#include <directx/d3dx12.h>
#include <directx/dxcore_interface.h>
#else
#include <wsl/winadapter.h>
#include <directx/d3d12.h>
#include <directx/d3d12sdklayers.h>
#include <directx/d3dx12.h>
#include <directx/dxcore_interface.h>
#include <dxguids/dxguids.h>
#endif

#ifdef interface
#undef interface
#endif

#if !defined(_WIN32) && defined(__CRT_UUID_DECL) && !defined(NACELLE_ID3D10BLOB_UUID_DEFINED)
#define NACELLE_ID3D10BLOB_UUID_DEFINED
__CRT_UUID_DECL(ID3D10Blob, 0x8ba5fb08, 0x5195, 0x40e2, 0xac, 0x58, 0x0d, 0x98, 0x9c, 0x3a, 0x01, 0x02)
#endif

#include "present_surface.h"

#ifdef __cplusplus
extern "C" {
#endif

HRESULT NacelleCreateAdapterFactory(REFIID riid, void** ppvFactory);

HRESULT NacelleCreateDevice(
    IUnknown* pAdapter,
    D3D_FEATURE_LEVEL minimumFeatureLevel,
    REFIID riid,
    void** ppDevice);

HRESULT NacelleGetDebugInterface(REFIID riid, void** ppvDebug);

HRESULT NacelleCreateSwapChain(
    ID3D12CommandQueue* pQueue,
    uint32_t width,
    uint32_t height,
    uint32_t bufferCount,
    DXGI_FORMAT format,
    PFN_NacellePresentCallback callback,
    void* userData,
    INacelleSwapChain** ppSwapChain);

HRESULT NacelleCreateBlob(SIZE_T size, ID3DBlob** ppBlob);

HRESULT NacelleSerializeRootSignature(
    const D3D12_ROOT_SIGNATURE_DESC* pRootSignature,
    D3D_ROOT_SIGNATURE_VERSION Version,
    ID3DBlob** ppBlob,
    ID3DBlob** ppErrorBlob);

HRESULT NacelleSerializeVersionedRootSignature(
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignature,
    ID3DBlob** ppBlob,
    ID3DBlob** ppErrorBlob);

#if !defined(_WIN32) || defined(NACELLE_EXPORT_D3D12_SYMBOLS)
HRESULT STDAPICALLTYPE DXCoreCreateAdapterFactory(REFIID riid, void** ppvFactory);

HRESULT WINAPI D3D12CreateDevice(
    IUnknown* pAdapter,
    D3D_FEATURE_LEVEL MinimumFeatureLevel,
    REFIID riid,
    void** ppDevice);

HRESULT WINAPI D3D12GetDebugInterface(REFIID riid, void** ppvDebug);

HRESULT WINAPI D3D12SerializeRootSignature(
    const D3D12_ROOT_SIGNATURE_DESC* pRootSignature,
    D3D_ROOT_SIGNATURE_VERSION Version,
    ID3DBlob** ppBlob,
    ID3DBlob** ppErrorBlob);

HRESULT WINAPI D3D12SerializeVersionedRootSignature(
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignature,
    ID3DBlob** ppBlob,
    ID3DBlob** ppErrorBlob);
#endif

#ifdef __cplusplus
} // extern "C"

#if !defined(_WIN32)
template <class T>
inline HRESULT DXCoreCreateAdapterFactory(T** ppvFactory) {
    return DXCoreCreateAdapterFactory(__uuidof(T), reinterpret_cast<void**>(ppvFactory));
}
#endif
#endif
