#pragma once

#include <cstdint>

#if !defined(_WIN32)
#include <wsl/winadapter.h>
#else
#include <unknwn.h>
#endif

#include <directx/d3d12.h>
#include <directx/dxgiformat.h>

typedef void (*PFN_NacellePresentCallback)(
    const void* rgbaOrBgraPixels,
    uint32_t width,
    uint32_t height,
    uint32_t rowPitchBytes,
    DXGI_FORMAT format,
    void* userData);

// {8F7C4A12-51E9-4D30-9B6A-2C18E0D59001}
static const GUID IID_INacelleSwapChain = {
    0x8f7c4a12, 0x51e9, 0x4d30, {0x9b, 0x6a, 0x2c, 0x18, 0xe0, 0xd5, 0x90, 0x01}
};

struct INacelleSwapChain : public IUnknown {
    virtual HRESULT STDMETHODCALLTYPE GetBuffer(uint32_t Buffer, REFIID riid, void** ppSurface) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetCurrentBackBufferIndex() = 0;
    virtual HRESULT STDMETHODCALLTYPE Present(uint32_t SyncInterval, uint32_t Flags) = 0;
    virtual HRESULT STDMETHODCALLTYPE ResizeBuffers(
        uint32_t BufferCount,
        uint32_t Width,
        uint32_t Height,
        DXGI_FORMAT NewFormat,
        uint32_t SwapChainFlags = 0) = 0;
    virtual HRESULT STDMETHODCALLTYPE MapFrontBufferPtr(
        const void** ppPixels,
        uint32_t* pWidth,
        uint32_t* pHeight,
        uint32_t* pRowPitch) = 0;
    virtual void STDMETHODCALLTYPE SetPresentCallback(PFN_NacellePresentCallback Callback, void* UserData) = 0;
};

#if !defined(_WIN32) && defined(__CRT_UUID_DECL)
__CRT_UUID_DECL(INacelleSwapChain, 0x8f7c4a12, 0x51e9, 0x4d30, 0x9b, 0x6a, 0x2c, 0x18, 0xe0, 0xd5, 0x90, 0x01)
#endif
