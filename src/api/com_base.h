#pragma once

#include "../common/platform.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nacelle {

struct GuidHash {
    size_t operator()(const GUID& g) const noexcept {
        uint64_t a = 0, b = 0;
        std::memcpy(&a, &g, 8);
        std::memcpy(&b, reinterpret_cast<const uint8_t*>(&g) + 8, 8);
        return std::hash<uint64_t>{}(a) ^ (std::hash<uint64_t>{}(b) * 0x9e3779b97f4a7c15ULL);
    }
};

struct GuidEq {
    bool operator()(const GUID& a, const GUID& b) const noexcept {
        return std::memcmp(&a, &b, sizeof(GUID)) == 0;
    }
};

class ObjectPrivateData {
public:
    HRESULT getPrivateData(REFGUID guid, UINT* pDataSize, void* pData) {
        if (!pDataSize) return E_INVALIDARG;
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(guid);
        if (it == entries_.end()) {
            *pDataSize = 0;
            return DXGI_ERROR_NOT_FOUND;
        }
        const Entry& e = it->second;
        if (!pData) {
            *pDataSize = e.size;
            return S_OK;
        }
        if (*pDataSize < e.size) {
            *pDataSize = e.size;
            return DXGI_ERROR_MORE_DATA;
        }
        *pDataSize = e.size;
        if (e.isInterface) {
            IUnknown* unk = e.iface;
            if (unk) unk->AddRef();
            std::memcpy(pData, &unk, sizeof(IUnknown*));
        } else if (e.size > 0) {
            std::memcpy(pData, e.bytes.data(), e.size);
        }
        return S_OK;
    }

    HRESULT setPrivateData(REFGUID guid, UINT dataSize, const void* pData) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!pData) {
            releaseEntry(guid);
            return S_OK;
        }
        releaseEntry(guid);
        Entry e;
        e.isInterface = false;
        e.size = dataSize;
        e.bytes.resize(dataSize);
        if (dataSize > 0) std::memcpy(e.bytes.data(), pData, dataSize);
        entries_.emplace(guid, std::move(e));
        return S_OK;
    }

    HRESULT setPrivateDataInterface(REFGUID guid, const IUnknown* pData) {
        std::lock_guard<std::mutex> lock(mu_);
        releaseEntry(guid);
        if (!pData) return S_OK;
        Entry e;
        e.isInterface = true;
        e.size = sizeof(IUnknown*);
        e.iface = const_cast<IUnknown*>(pData);
        e.iface->AddRef();
        entries_.emplace(guid, std::move(e));
        return S_OK;
    }

    HRESULT setName(LPCWSTR name) {
        std::lock_guard<std::mutex> lock(mu_);
        name_ = name ? wideToUtf8(name) : std::string();
        return S_OK;
    }

    std::string getNameUtf8() const {
        std::lock_guard<std::mutex> lock(mu_);
        return name_;
    }

    ~ObjectPrivateData() {
        for (auto& [_, e] : entries_) {
            if (e.isInterface && e.iface) e.iface->Release();
        }
    }

private:
    struct Entry {
        bool isInterface = false;
        UINT size = 0;
        std::vector<uint8_t> bytes;
        IUnknown* iface = nullptr;
    };

    void releaseEntry(REFGUID guid) {
        auto it = entries_.find(guid);
        if (it != entries_.end()) {
            if (it->second.isInterface && it->second.iface) {
                it->second.iface->Release();
            }
            entries_.erase(it);
        }
    }

    mutable std::mutex mu_;
    std::unordered_map<GUID, Entry, GuidHash, GuidEq> entries_;
    std::string name_;
};

template <typename... Interfaces>
struct InterfaceList {};

template <typename PrimaryInterface, typename... ExtraInterfaces>
class ComObject : public PrimaryInterface, public ExtraInterfaces... {
public:
    virtual ~ComObject() = default;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        *ppvObject = nullptr;
        if (InlineIsEqualGUID(riid, __uuidof(IUnknown))) {
            *ppvObject = static_cast<IUnknown*>(static_cast<PrimaryInterface*>(this));
            AddRef();
            return S_OK;
        }
        if (tryMatch<PrimaryInterface>(riid, ppvObject) ||
            (tryMatch<ExtraInterfaces>(riid, ppvObject) || ...)) {
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }

    ULONG STDMETHODCALLTYPE AddRef() override {
        return refCount_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    ULONG STDMETHODCALLTYPE Release() override {
        ULONG r = refCount_.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (r == 0) delete this;
        return r;
    }

protected:
    template <typename T>
    bool tryMatch(REFIID riid, void** ppv) {
        if (InlineIsEqualGUID(riid, __uuidof(T))) {
            *ppv = static_cast<T*>(this);
            return true;
        }
        return false;
    }

    std::atomic<ULONG> refCount_{1};
};

/// Helper mixin implementing ID3D12Object methods on any ComObject deriving from ID3D12Object.
template <typename PrimaryInterface, typename... ExtraInterfaces>
class D3D12ObjectBase : public ComObject<PrimaryInterface, ExtraInterfaces...> {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        if (InlineIsEqualGUID(riid, __uuidof(ID3D12Object))) {
            *ppvObject = static_cast<ID3D12Object*>(static_cast<PrimaryInterface*>(this));
            this->AddRef();
            return S_OK;
        }
        return ComObject<PrimaryInterface, ExtraInterfaces...>::QueryInterface(riid, ppvObject);
    }

    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* pDataSize, void* pData) override {
        return privateData_.getPrivateData(guid, pDataSize, pData);
    }

    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT DataSize, const void* pData) override {
        return privateData_.setPrivateData(guid, DataSize, pData);
    }

    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* pData) override {
        return privateData_.setPrivateDataInterface(guid, pData);
    }

    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR Name) override {
        return privateData_.setName(Name);
    }

    std::string debugName() const { return privateData_.getNameUtf8(); }

protected:
    ObjectPrivateData privateData_;
};

/// Helper mixin implementing ID3D12DeviceChild + ID3D12Object methods.
template <typename PrimaryInterface, typename... ExtraInterfaces>
class D3D12DeviceChildBase : public D3D12ObjectBase<PrimaryInterface, ExtraInterfaces...> {
public:
    explicit D3D12DeviceChildBase(ID3D12Device* device) : device_(device) {
        if (device_) device_->AddRef();
    }

    ~D3D12DeviceChildBase() override {
        if (device_) device_->Release();
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        if (InlineIsEqualGUID(riid, __uuidof(ID3D12DeviceChild))) {
            *ppvObject = static_cast<ID3D12DeviceChild*>(static_cast<PrimaryInterface*>(this));
            this->AddRef();
            return S_OK;
        }
        return D3D12ObjectBase<PrimaryInterface, ExtraInterfaces...>::QueryInterface(riid, ppvObject);
    }

    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** ppvDevice) override {
        if (!ppvDevice) return E_POINTER;
        *ppvDevice = nullptr;
        if (!device_) return E_FAIL;
        return device_->QueryInterface(riid, ppvDevice);
    }

    ID3D12Device* parentDevice() const { return device_; }

protected:
    ID3D12Device* device_ = nullptr;
};

/// Helper mixin for ID3D12Pageable objects.
template <typename PrimaryInterface, typename... ExtraInterfaces>
class D3D12PageableBase : public D3D12DeviceChildBase<PrimaryInterface, ExtraInterfaces...> {
public:
    using D3D12DeviceChildBase<PrimaryInterface, ExtraInterfaces...>::D3D12DeviceChildBase;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        if (InlineIsEqualGUID(riid, __uuidof(ID3D12Pageable))) {
            *ppvObject = static_cast<ID3D12Pageable*>(static_cast<PrimaryInterface*>(this));
            this->AddRef();
            return S_OK;
        }
        return D3D12DeviceChildBase<PrimaryInterface, ExtraInterfaces...>::QueryInterface(riid, ppvObject);
    }
};

} // namespace nacelle
