#include "../common/platform.h"

#include <cstring>

#include "com_base.h"
#include "../../include/nacelle/nacelle.h"

#if !defined(_WIN32)
// Define DXCORE_ADAPTER_ATTRIBUTE_* GUID globals declared in directx/dxcore_interface.h.
const GUID DXCORE_ADAPTER_ATTRIBUTE_D3D11_GRAPHICS = {
    0x8c47866b, 0x7583, 0x450d, {0xf0, 0xf0, 0x6b, 0xad, 0xa8, 0x95, 0xaf, 0x4b}
};
const GUID DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS = {
    0x0c9ece4d, 0x2f6e, 0x4f01, {0x8c, 0x96, 0xe8, 0x9e, 0x33, 0x1b, 0x47, 0xb1}
};
const GUID DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE = {
    0x248e2800, 0xa793, 0x4724, {0xab, 0xaa, 0x23, 0xa6, 0xde, 0x1b, 0xe0, 0x90}
};
#endif

namespace nacelle {

namespace {

constexpr LUID kNacelleAdapterLuid{0x4E434C31, 1};
constexpr char kNacelleDriverDesc[] = "Nacelle Software Adapter";

class NacelleDXCoreAdapterFactory;

class NacelleDXCoreAdapter : public ComObject<IDXCoreAdapter> {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        if (InlineIsEqualGUID(riid, __uuidof(IUnknown)) ||
            InlineIsEqualGUID(riid, __uuidof(IDXCoreAdapter))) {
            *ppvObject = static_cast<IDXCoreAdapter*>(this);
            AddRef();
            return S_OK;
        }
        *ppvObject = nullptr;
        return E_NOINTERFACE;
    }

    bool STDMETHODCALLTYPE IsValid() override { return true; }

    bool STDMETHODCALLTYPE IsAttributeSupported(REFGUID attributeGUID) override {
        return InlineIsEqualGUID(attributeGUID, DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS) ||
               InlineIsEqualGUID(attributeGUID, DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE);
    }

    bool STDMETHODCALLTYPE IsPropertySupported(DXCoreAdapterProperty property) override {
        switch (property) {
        case DXCoreAdapterProperty::InstanceLuid:
        case DXCoreAdapterProperty::DriverVersion:
        case DXCoreAdapterProperty::DriverDescription:
        case DXCoreAdapterProperty::HardwareID:
        case DXCoreAdapterProperty::KmdModelVersion:
        case DXCoreAdapterProperty::ComputePreemptionGranularity:
        case DXCoreAdapterProperty::GraphicsPreemptionGranularity:
        case DXCoreAdapterProperty::DedicatedAdapterMemory:
        case DXCoreAdapterProperty::DedicatedSystemMemory:
        case DXCoreAdapterProperty::SharedSystemMemory:
        case DXCoreAdapterProperty::AcgCompatible:
        case DXCoreAdapterProperty::IsHardware:
        case DXCoreAdapterProperty::IsIntegrated:
        case DXCoreAdapterProperty::IsDetachable:
        case DXCoreAdapterProperty::HardwareIDParts:
            return true;
        default:
            return false;
        }
    }

    HRESULT STDMETHODCALLTYPE GetPropertySize(DXCoreAdapterProperty property, size_t* bufferSize) override {
        if (!bufferSize) return E_INVALIDARG;
        switch (property) {
        case DXCoreAdapterProperty::InstanceLuid:
            *bufferSize = sizeof(LUID);
            return S_OK;
        case DXCoreAdapterProperty::DriverVersion:
            *bufferSize = sizeof(uint64_t);
            return S_OK;
        case DXCoreAdapterProperty::DriverDescription:
            *bufferSize = sizeof(kNacelleDriverDesc);
            return S_OK;
        case DXCoreAdapterProperty::HardwareID:
            *bufferSize = sizeof(DXCoreHardwareID);
            return S_OK;
        case DXCoreAdapterProperty::HardwareIDParts:
            *bufferSize = sizeof(DXCoreHardwareIDParts);
            return S_OK;
        case DXCoreAdapterProperty::KmdModelVersion:
            *bufferSize = sizeof(uint32_t);
            return S_OK;
        case DXCoreAdapterProperty::ComputePreemptionGranularity:
        case DXCoreAdapterProperty::GraphicsPreemptionGranularity:
            *bufferSize = sizeof(uint16_t);
            return S_OK;
        case DXCoreAdapterProperty::DedicatedAdapterMemory:
        case DXCoreAdapterProperty::DedicatedSystemMemory:
        case DXCoreAdapterProperty::SharedSystemMemory:
            *bufferSize = sizeof(uint64_t);
            return S_OK;
        case DXCoreAdapterProperty::AcgCompatible:
        case DXCoreAdapterProperty::IsHardware:
        case DXCoreAdapterProperty::IsIntegrated:
        case DXCoreAdapterProperty::IsDetachable:
            *bufferSize = sizeof(bool);
            return S_OK;
        default:
            return E_INVALIDARG;
        }
    }

    HRESULT STDMETHODCALLTYPE GetProperty(
        DXCoreAdapterProperty property, size_t bufferSize, void* propertyData) override {
        if (!propertyData) return E_INVALIDARG;
        size_t req = 0;
        HRESULT hr = GetPropertySize(property, &req);
        if (FAILED(hr)) return hr;
        if (bufferSize < req) return E_INVALIDARG;

        switch (property) {
        case DXCoreAdapterProperty::InstanceLuid:
            std::memcpy(propertyData, &kNacelleAdapterLuid, sizeof(LUID));
            return S_OK;
        case DXCoreAdapterProperty::DriverVersion: {
            uint64_t ver = (1ULL << 48);
            std::memcpy(propertyData, &ver, sizeof(ver));
            return S_OK;
        }
        case DXCoreAdapterProperty::DriverDescription:
            std::memcpy(propertyData, kNacelleDriverDesc, sizeof(kNacelleDriverDesc));
            return S_OK;
        case DXCoreAdapterProperty::HardwareID: {
            DXCoreHardwareID hw{0x1414, 0x008C, 0, 0};
            std::memcpy(propertyData, &hw, sizeof(hw));
            return S_OK;
        }
        case DXCoreAdapterProperty::HardwareIDParts: {
            DXCoreHardwareIDParts parts{0x1414, 0x008C, 0, 0, 0};
            std::memcpy(propertyData, &parts, sizeof(parts));
            return S_OK;
        }
        case DXCoreAdapterProperty::KmdModelVersion: {
            uint32_t kmd = 3000; // WDDM 3.0
            std::memcpy(propertyData, &kmd, sizeof(kmd));
            return S_OK;
        }
        case DXCoreAdapterProperty::ComputePreemptionGranularity:
        case DXCoreAdapterProperty::GraphicsPreemptionGranularity: {
            uint16_t gran = 100;
            std::memcpy(propertyData, &gran, sizeof(gran));
            return S_OK;
        }
        case DXCoreAdapterProperty::DedicatedAdapterMemory:
        case DXCoreAdapterProperty::DedicatedSystemMemory: {
            uint64_t zero = 0;
            std::memcpy(propertyData, &zero, sizeof(zero));
            return S_OK;
        }
        case DXCoreAdapterProperty::SharedSystemMemory: {
            uint64_t shared = 8ULL << 30;
            std::memcpy(propertyData, &shared, sizeof(shared));
            return S_OK;
        }
        case DXCoreAdapterProperty::AcgCompatible:
        case DXCoreAdapterProperty::IsIntegrated: {
            bool b = true;
            std::memcpy(propertyData, &b, sizeof(b));
            return S_OK;
        }
        case DXCoreAdapterProperty::IsHardware:
        case DXCoreAdapterProperty::IsDetachable: {
            bool b = false;
            std::memcpy(propertyData, &b, sizeof(b));
            return S_OK;
        }
        default:
            return E_INVALIDARG;
        }
    }

    bool STDMETHODCALLTYPE IsSetStateSupported(DXCoreAdapterState) override { return false; }
    bool STDMETHODCALLTYPE IsQueryStateSupported(DXCoreAdapterState) override { return false; }

    HRESULT STDMETHODCALLTYPE QueryState(
        DXCoreAdapterState, size_t, const void*, size_t, void*) override {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE SetState(
        DXCoreAdapterState, size_t, const void*, size_t, const void*) override {
        return E_NOTIMPL;
    }

    HRESULT STDMETHODCALLTYPE GetFactory(REFIID riid, void** ppvFactory) override {
        return NacelleCreateAdapterFactory(riid, ppvFactory);
    }
};

class NacelleDXCoreAdapterList : public ComObject<IDXCoreAdapterList> {
public:
    explicit NacelleDXCoreAdapterList(bool includeAdapter) : hasAdapter_(includeAdapter) {}

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        if (InlineIsEqualGUID(riid, __uuidof(IUnknown)) ||
            InlineIsEqualGUID(riid, __uuidof(IDXCoreAdapterList))) {
            *ppvObject = static_cast<IDXCoreAdapterList*>(this);
            AddRef();
            return S_OK;
        }
        *ppvObject = nullptr;
        return E_NOINTERFACE;
    }

    HRESULT STDMETHODCALLTYPE GetAdapter(uint32_t index, REFIID riid, void** ppvAdapter) override {
        if (!ppvAdapter) return E_INVALIDARG;
        if (!hasAdapter_ || index != 0) return E_INVALIDARG;
        auto* adapter = new NacelleDXCoreAdapter();
        HRESULT hr = adapter->QueryInterface(riid, ppvAdapter);
        adapter->Release();
        return hr;
    }

    uint32_t STDMETHODCALLTYPE GetAdapterCount() override {
        return hasAdapter_ ? 1u : 0u;
    }

    bool STDMETHODCALLTYPE IsStale() override { return false; }

    HRESULT STDMETHODCALLTYPE GetFactory(REFIID riid, void** ppvFactory) override {
        return NacelleCreateAdapterFactory(riid, ppvFactory);
    }

    HRESULT STDMETHODCALLTYPE Sort(uint32_t, const DXCoreAdapterPreference*) override {
        return S_OK;
    }

    bool STDMETHODCALLTYPE IsAdapterPreferenceSupported(DXCoreAdapterPreference) override {
        return true;
    }

private:
    bool hasAdapter_ = true;
};

class NacelleDXCoreAdapterFactory : public ComObject<IDXCoreAdapterFactory1> {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        if (InlineIsEqualGUID(riid, __uuidof(IUnknown)) ||
            InlineIsEqualGUID(riid, __uuidof(IDXCoreAdapterFactory)) ||
            InlineIsEqualGUID(riid, __uuidof(IDXCoreAdapterFactory1))) {
            *ppvObject = static_cast<IDXCoreAdapterFactory1*>(this);
            AddRef();
            return S_OK;
        }
        *ppvObject = nullptr;
        return E_NOINTERFACE;
    }

    HRESULT STDMETHODCALLTYPE CreateAdapterList(
        uint32_t numAttributes, const GUID* filterAttributes, REFIID riid, void** ppvAdapterList) override {
        if (!ppvAdapterList) return E_INVALIDARG;
        bool supported = true;
        if (numAttributes > 0 && filterAttributes) {
            NacelleDXCoreAdapter probe;
            for (uint32_t i = 0; i < numAttributes; ++i) {
                if (!probe.IsAttributeSupported(filterAttributes[i])) {
                    supported = false;
                    break;
                }
            }
        }
        auto* list = new NacelleDXCoreAdapterList(supported);
        HRESULT hr = list->QueryInterface(riid, ppvAdapterList);
        list->Release();
        return hr;
    }

    HRESULT STDMETHODCALLTYPE CreateAdapterListByWorkload(
        DXCoreWorkload,
        DXCoreRuntimeFilterFlags,
        DXCoreHardwareTypeFilterFlags hardwareTypeFilterFlags,
        REFIID riid,
        void** ppvAdapterList) override {
        if (!ppvAdapterList) return E_INVALIDARG;
        (void)hardwareTypeFilterFlags;
        auto* list = new NacelleDXCoreAdapterList(true);
        HRESULT hr = list->QueryInterface(riid, ppvAdapterList);
        list->Release();
        return hr;
    }

    HRESULT STDMETHODCALLTYPE GetAdapterByLuid(const LUID& adapterLUID, REFIID riid, void** ppvAdapter) override {
        if (!ppvAdapter) return E_INVALIDARG;
        if (adapterLUID.LowPart != kNacelleAdapterLuid.LowPart ||
            adapterLUID.HighPart != kNacelleAdapterLuid.HighPart) {
            return E_INVALIDARG;
        }
        auto* adapter = new NacelleDXCoreAdapter();
        HRESULT hr = adapter->QueryInterface(riid, ppvAdapter);
        adapter->Release();
        return hr;
    }

    bool STDMETHODCALLTYPE IsNotificationTypeSupported(DXCoreNotificationType) override {
        return false;
    }

    HRESULT STDMETHODCALLTYPE RegisterEventNotification(
        IUnknown*, DXCoreNotificationType, PFN_DXCORE_NOTIFICATION_CALLBACK, void*, uint32_t* eventCookie) override {
        if (eventCookie) *eventCookie = 1;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE UnregisterEventNotification(uint32_t) override {
        return S_OK;
    }
};

class NacelleDebug : public ComObject<ID3D12Debug> {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        if (InlineIsEqualGUID(riid, __uuidof(IUnknown)) ||
            InlineIsEqualGUID(riid, __uuidof(ID3D12Debug))) {
            *ppvObject = static_cast<ID3D12Debug*>(this);
            AddRef();
            return S_OK;
        }
        *ppvObject = nullptr;
        return E_NOINTERFACE;
    }

    void STDMETHODCALLTYPE EnableDebugLayer() override {}
};

} // namespace

} // namespace nacelle

extern "C" {

HRESULT NacelleCreateAdapterFactory(REFIID riid, void** ppvFactory) {
    if (!ppvFactory) return E_INVALIDARG;
    auto* factory = new nacelle::NacelleDXCoreAdapterFactory();
    HRESULT hr = factory->QueryInterface(riid, ppvFactory);
    factory->Release();
    return hr;
}

HRESULT NacelleGetDebugInterface(REFIID riid, void** ppvDebug) {
    if (!ppvDebug) return E_INVALIDARG;
    auto* dbg = new nacelle::NacelleDebug();
    HRESULT hr = dbg->QueryInterface(riid, ppvDebug);
    dbg->Release();
    return hr;
}

#if !defined(_WIN32) || defined(NACELLE_EXPORT_D3D12_SYMBOLS)
HRESULT STDAPICALLTYPE DXCoreCreateAdapterFactory(REFIID riid, void** ppvFactory) {
    return NacelleCreateAdapterFactory(riid, ppvFactory);
}

HRESULT WINAPI D3D12CreateDevice(
    IUnknown* pAdapter,
    D3D_FEATURE_LEVEL MinimumFeatureLevel,
    REFIID riid,
    void** ppDevice) {
    return NacelleCreateDevice(pAdapter, MinimumFeatureLevel, riid, ppDevice);
}

HRESULT WINAPI D3D12GetDebugInterface(REFIID riid, void** ppvDebug) {
    return NacelleGetDebugInterface(riid, ppvDebug);
}
#endif

} // extern "C"
