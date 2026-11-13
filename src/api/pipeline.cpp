#include "pipeline.h"

#include <cstring>
#include <sstream>

#include <llvm/Object/DXContainer.h>
#include <llvm/Support/MemoryBufferRef.h>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include "../../include/nacelle/nacelle.h"
#include "../compiler/jit_engine.h"

namespace nacelle {

namespace {
constexpr uint32_t kNacelleRootSigMagic = 0x4E525453; // 'NRTS'

struct BinaryWriter {
    std::vector<uint8_t> buf;
    template <typename T>
    void write(const T& v) {
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }
};

struct BinaryReader {
    const uint8_t* ptr = nullptr;
    size_t size = 0;
    size_t pos = 0;
    template <typename T>
    bool read(T& out) {
        if (pos + sizeof(T) > size) return false;
        std::memcpy(&out, ptr + pos, sizeof(T));
        pos += sizeof(T);
        return true;
    }
};

void computeLayoutHash(RootSignatureLayout& layout) {
    BinaryWriter w;
    for (const auto& p : layout.params) {
        w.write(p.type);
        w.write(p.visibility);
        w.write(p.shaderRegister);
        w.write(p.registerSpace);
        w.write(p.num32BitValues);
        w.write(p.constDwordOffset);
        uint32_t rc = static_cast<uint32_t>(p.ranges.size());
        w.write(rc);
        for (const auto& r : p.ranges) {
            w.write(r.type);
            w.write(r.numDescriptors);
            w.write(r.baseShaderRegister);
            w.write(r.registerSpace);
            w.write(r.offsetInTable);
        }
    }
    uint32_t sc = static_cast<uint32_t>(layout.staticSamplers.size());
    w.write(sc);
    for (const auto& s : layout.staticSamplers) {
        w.write(s.shaderRegister);
        w.write(s.registerSpace);
        w.write(s.visibility);
        w.write(s.desc);
    }
    XXH128_hash_t h = XXH3_128bits(w.buf.data(), w.buf.size());
    layout.hashLo = h.low64;
    layout.hashHi = h.high64;
}

D3D12_DESCRIPTOR_RANGE_TYPE toD3D12RangeType(uint32_t llvmRangeType) {
    switch (llvmRangeType) {
    case 0: return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    case 1: return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    case 2: return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    case 3: return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
    default: return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    }
}

void staticSamplerToDesc(const llvm::dxbc::RTS0::v1::StaticSampler& ss, D3D12_SAMPLER_DESC& out) {
    out.Filter = static_cast<D3D12_FILTER>(ss.Filter);
    out.AddressU = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(ss.AddressU);
    out.AddressV = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(ss.AddressV);
    out.AddressW = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(ss.AddressW);
    out.MipLODBias = ss.MipLODBias;
    out.MaxAnisotropy = ss.MaxAnisotropy;
    out.ComparisonFunc = static_cast<D3D12_COMPARISON_FUNC>(ss.ComparisonFunc);
    out.MinLOD = ss.MinLOD;
    out.MaxLOD = ss.MaxLOD;
    std::memset(out.BorderColor, 0, sizeof(out.BorderColor));
    if (ss.BorderColor == 1) { // OPAQUE_BLACK
        out.BorderColor[3] = 1.0f;
    } else if (ss.BorderColor == 2) { // OPAQUE_WHITE
        out.BorderColor[0] = out.BorderColor[1] = out.BorderColor[2] = out.BorderColor[3] = 1.0f;
    }
}
} // namespace

RootBindingTarget RootSignatureLayout::resolveBinding(uint32_t resClass, uint32_t space, uint32_t reg) const {
    // resClass: 0=SRV, 1=UAV, 2=CBV, 3=Sampler
    for (uint32_t i = 0; i < static_cast<uint32_t>(params.size()); ++i) {
        const auto& p = params[i];
        if (p.type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS && resClass == 2) {
            if (p.registerSpace == space && p.shaderRegister == reg) {
                return {RootBindingTarget::RootConstants, i, p.constDwordOffset, 0};
            }
        } else if (p.type == D3D12_ROOT_PARAMETER_TYPE_CBV && resClass == 2) {
            if (p.registerSpace == space && p.shaderRegister == reg) {
                return {RootBindingTarget::RootDescriptor, i, 0, 0};
            }
        } else if (p.type == D3D12_ROOT_PARAMETER_TYPE_SRV && resClass == 0) {
            if (p.registerSpace == space && p.shaderRegister == reg) {
                return {RootBindingTarget::RootDescriptor, i, 0, 0};
            }
        } else if (p.type == D3D12_ROOT_PARAMETER_TYPE_UAV && resClass == 1) {
            if (p.registerSpace == space && p.shaderRegister == reg) {
                return {RootBindingTarget::RootDescriptor, i, 0, 0};
            }
        } else if (p.type == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
            D3D12_DESCRIPTOR_RANGE_TYPE wantType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            if (resClass == 1) wantType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            else if (resClass == 2) wantType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
            else if (resClass == 3) wantType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;

            for (const auto& r : p.ranges) {
                if (r.type != wantType || r.registerSpace != space) continue;
                uint64_t endReg = (r.numDescriptors == UINT32_MAX)
                                      ? UINT64_MAX
                                      : (uint64_t(r.baseShaderRegister) + r.numDescriptors);
                if (reg >= r.baseShaderRegister && reg < endReg) {
                    uint32_t offset = r.offsetInTable + (reg - r.baseShaderRegister);
                    return {RootBindingTarget::DescriptorTableOffset, i, offset, 0};
                }
            }
        }
    }

    if (resClass == 3) {
        for (uint32_t s = 0; s < static_cast<uint32_t>(staticSamplers.size()); ++s) {
            if (staticSamplers[s].registerSpace == space && staticSamplers[s].shaderRegister == reg) {
                return {RootBindingTarget::StaticSampler, 0, 0, s};
            }
        }
    }
    return {};
}

NacelleBlob::NacelleBlob(size_t size) : data_(size, 0) {}

NacelleBlob::NacelleBlob(const void* data, size_t size) : data_(size) {
    if (data && size > 0) std::memcpy(data_.data(), data, size);
}

NacelleRootSignature::NacelleRootSignature(ID3D12Device* device, RootSignatureLayout layout)
    : D3D12DeviceChildBase(device), layout_(std::move(layout)) {}

HRESULT STDMETHODCALLTYPE NacelleRootSignature::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12RootSignature))) {
        *ppvObject = static_cast<ID3D12RootSignature*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12DeviceChildBase::QueryInterface(riid, ppvObject);
}

HRESULT NacelleRootSignature::parseBlob(const void* blob, size_t size, RootSignatureLayout& outLayout) {
    if (!blob || size < 4) return E_INVALIDARG;
    outLayout = RootSignatureLayout{};

    uint32_t magic = 0;
    std::memcpy(&magic, blob, 4);

    if (magic == kNacelleRootSigMagic) {
        BinaryReader r{static_cast<const uint8_t*>(blob), size, 4};
        uint32_t numParams = 0, numSamplers = 0;
        if (!r.read(numParams) || !r.read(numSamplers)) return E_INVALIDARG;
        outLayout.params.resize(numParams);
        uint32_t nextConstDword = 0;
        for (uint32_t i = 0; i < numParams; ++i) {
            auto& p = outLayout.params[i];
            uint32_t numRanges = 0;
            if (!r.read(p.type) || !r.read(p.visibility) || !r.read(p.shaderRegister) ||
                !r.read(p.registerSpace) || !r.read(p.num32BitValues) || !r.read(numRanges)) {
                return E_INVALIDARG;
            }
            if (p.type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                p.constDwordOffset = nextConstDword;
                nextConstDword += p.num32BitValues;
            }
            p.ranges.resize(numRanges);
            for (uint32_t j = 0; j < numRanges; ++j) {
                if (!r.read(p.ranges[j])) return E_INVALIDARG;
            }
        }
        outLayout.staticSamplers.resize(numSamplers);
        for (uint32_t i = 0; i < numSamplers; ++i) {
            if (!r.read(outLayout.staticSamplers[i])) return E_INVALIDARG;
        }
        computeLayoutHash(outLayout);
        return S_OK;
    }

    // Try DXContainer (RTS0 chunk from dxc) using LLVM's DXContainer parser
    llvm::MemoryBufferRef memRef(
        llvm::StringRef(static_cast<const char*>(blob), size), "rootsig");
    auto containerOrErr = llvm::object::DXContainer::create(memRef);
    if (!containerOrErr) {
        llvm::consumeError(containerOrErr.takeError());
        return E_INVALIDARG;
    }
    const auto& rsOpt = containerOrErr->getRootSignature();
    if (!rsOpt.has_value()) return E_INVALIDARG;

    const auto& rs = *rsOpt;
    uint32_t nextConstDword = 0;
    for (const auto& hdr : rs.param_headers()) {
        auto paramViewOrErr = rs.getParameter(hdr);
        if (!paramViewOrErr) {
            llvm::consumeError(paramViewOrErr.takeError());
            return E_INVALIDARG;
        }
        RootSignatureLayout::Param p;
        p.visibility = static_cast<D3D12_SHADER_VISIBILITY>(hdr.ShaderVisibility);
        if (auto* rc = llvm::dyn_cast<llvm::object::DirectX::RootConstantView>(&*paramViewOrErr)) {
            auto valOrErr = rc->read();
            if (!valOrErr) { llvm::consumeError(valOrErr.takeError()); return E_INVALIDARG; }
            p.type = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            p.shaderRegister = valOrErr->ShaderRegister;
            p.registerSpace = valOrErr->RegisterSpace;
            p.num32BitValues = valOrErr->Num32BitValues;
            p.constDwordOffset = nextConstDword;
            nextConstDword += p.num32BitValues;
        } else if (auto* rd = llvm::dyn_cast<llvm::object::DirectX::RootDescriptorView>(&*paramViewOrErr)) {
            auto valOrErr = rd->read(rs.getVersion());
            if (!valOrErr) { llvm::consumeError(valOrErr.takeError()); return E_INVALIDARG; }
            if (hdr.ParameterType == llvm::to_underlying(llvm::dxbc::RootParameterType::CBV))
                p.type = D3D12_ROOT_PARAMETER_TYPE_CBV;
            else if (hdr.ParameterType == llvm::to_underlying(llvm::dxbc::RootParameterType::SRV))
                p.type = D3D12_ROOT_PARAMETER_TYPE_SRV;
            else
                p.type = D3D12_ROOT_PARAMETER_TYPE_UAV;
            p.shaderRegister = valOrErr->ShaderRegister;
            p.registerSpace = valOrErr->RegisterSpace;
        } else if (auto* dt = llvm::dyn_cast<llvm::object::DirectX::DescriptorTableView>(&*paramViewOrErr)) {
            p.type = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            uint32_t runningOffset = 0;
            if (rs.getVersion() == 1) {
                auto tbl = dt->read<llvm::dxbc::RTS0::v1::DescriptorRange>();
                if (!tbl) { llvm::consumeError(tbl.takeError()); return E_INVALIDARG; }
                for (const auto& r : *tbl) {
                    RootSignatureLayout::Param::Range rng;
                    rng.type = toD3D12RangeType(r.RangeType);
                    rng.numDescriptors = r.NumDescriptors;
                    rng.baseShaderRegister = r.BaseShaderRegister;
                    rng.registerSpace = r.RegisterSpace;
                    rng.offsetInTable = (r.OffsetInDescriptorsFromTableStart == UINT32_MAX)
                                            ? runningOffset
                                            : r.OffsetInDescriptorsFromTableStart;
                    runningOffset = rng.offsetInTable + r.NumDescriptors;
                    p.ranges.push_back(rng);
                }
            } else {
                auto tbl = dt->read<llvm::dxbc::RTS0::v2::DescriptorRange>();
                if (!tbl) { llvm::consumeError(tbl.takeError()); return E_INVALIDARG; }
                for (const auto& r : *tbl) {
                    RootSignatureLayout::Param::Range rng;
                    rng.type = toD3D12RangeType(r.RangeType);
                    rng.numDescriptors = r.NumDescriptors;
                    rng.baseShaderRegister = r.BaseShaderRegister;
                    rng.registerSpace = r.RegisterSpace;
                    rng.offsetInTable = (r.OffsetInDescriptorsFromTableStart == UINT32_MAX)
                                            ? runningOffset
                                            : r.OffsetInDescriptorsFromTableStart;
                    runningOffset = rng.offsetInTable + r.NumDescriptors;
                    p.ranges.push_back(rng);
                }
            }
        }
        outLayout.params.push_back(std::move(p));
    }

    for (const auto& ss : rs.samplers()) {
        RootSignatureLayout::StaticSampler s;
        s.shaderRegister = ss.ShaderRegister;
        s.registerSpace = ss.RegisterSpace;
        s.visibility = static_cast<D3D12_SHADER_VISIBILITY>(ss.ShaderVisibility);
        staticSamplerToDesc(ss, s.desc);
        outLayout.staticSamplers.push_back(s);
    }
    computeLayoutHash(outLayout);
    return S_OK;
}

NacellePipelineState::NacellePipelineState(ID3D12Device* device,
                                     const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc,
                                     NacelleRootSignature* rootSig,
                                     std::shared_ptr<CompiledComputePSO> compiled)
    : D3D12PageableBase(device),
      isCompute_(true),
      rootSig_(rootSig),
      csDesc_(desc),
      computePSO_(std::move(compiled)) {
    if (rootSig_) rootSig_->AddRef();
}

NacellePipelineState::NacellePipelineState(ID3D12Device* device,
                                     const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
                                     NacelleRootSignature* rootSig,
                                     std::shared_ptr<CompiledGraphicsPSO> compiled)
    : D3D12PageableBase(device),
      isCompute_(false),
      rootSig_(rootSig),
      gfxDesc_(desc),
      graphicsPSO_(std::move(compiled)) {
    if (rootSig_) rootSig_->AddRef();
    if (desc.InputLayout.NumElements > 0 && desc.InputLayout.pInputElementDescs) {
        inputElements_.assign(desc.InputLayout.pInputElementDescs,
                              desc.InputLayout.pInputElementDescs + desc.InputLayout.NumElements);
        semanticNames_.resize(inputElements_.size());
        for (size_t i = 0; i < inputElements_.size(); ++i) {
            semanticNames_[i] = inputElements_[i].SemanticName ? inputElements_[i].SemanticName : "";
            inputElements_[i].SemanticName = semanticNames_[i].c_str();
        }
        gfxDesc_.InputLayout.pInputElementDescs = inputElements_.data();
    }
}

NacellePipelineState::~NacellePipelineState() {
    if (rootSig_) rootSig_->Release();
}

HRESULT STDMETHODCALLTYPE NacellePipelineState::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12PipelineState))) {
        *ppvObject = static_cast<ID3D12PipelineState*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12PageableBase::QueryInterface(riid, ppvObject);
}

HRESULT STDMETHODCALLTYPE NacellePipelineState::GetCachedBlob(ID3DBlob** ppBlob) {
    if (!ppBlob) return E_INVALIDARG;
    uint64_t hashes[2]{};
    if (isCompute_ && computePSO_) {
        hashes[0] = computePSO_->cacheHashLo;
        hashes[1] = computePSO_->cacheHashHi;
    } else if (graphicsPSO_) {
        hashes[0] = graphicsPSO_->cacheHashLo;
        hashes[1] = graphicsPSO_->cacheHashHi;
    }
    *ppBlob = new NacelleBlob(hashes, sizeof(hashes));
    return S_OK;
}

NacellePipelineLibrary::NacellePipelineLibrary(ID3D12Device* device, const void* blob, size_t size)
    : D3D12DeviceChildBase(device) {
    if (!blob || size < 8) return;
    BinaryReader r{static_cast<const uint8_t*>(blob), size, 0};
    uint32_t magic = 0, count = 0;
    if (!r.read(magic) || magic != 0x504C4942 /* 'PLIB' */ || !r.read(count)) return;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t nameLen = 0, dataLen = 0;
        StoredEntry e;
        if (!r.read(nameLen) || r.pos + nameLen > r.size) return;
        std::string name(reinterpret_cast<const char*>(r.ptr + r.pos), nameLen);
        r.pos += nameLen;
        if (!r.read(e.isCompute) || !r.read(e.hashLo) || !r.read(e.hashHi) || !r.read(dataLen)) return;
        if (r.pos + dataLen > r.size) return;
        e.serializedObj.assign(r.ptr + r.pos, r.ptr + r.pos + dataLen);
        r.pos += dataLen;
        if (!e.serializedObj.empty()) {
            importCachedPSOBlob(e.hashLo, e.hashHi, e.serializedObj.data(), e.serializedObj.size());
        }
        entries_.emplace(std::move(name), std::move(e));
    }
}

HRESULT STDMETHODCALLTYPE NacellePipelineLibrary::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12PipelineLibrary))) {
        *ppvObject = static_cast<ID3D12PipelineLibrary*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12DeviceChildBase::QueryInterface(riid, ppvObject);
}

HRESULT STDMETHODCALLTYPE NacellePipelineLibrary::StorePipeline(LPCWSTR pName, ID3D12PipelineState* pPipeline) {
    if (!pName || !pPipeline) return E_INVALIDARG;
    auto* pso = static_cast<NacellePipelineState*>(pPipeline);
    std::string key = wideToUtf8(pName);
    StoredEntry e;
    e.isCompute = pso->isCompute();
    if (e.isCompute && pso->computePSO()) {
        e.hashLo = pso->computePSO()->cacheHashLo;
        e.hashHi = pso->computePSO()->cacheHashHi;
    } else if (pso->graphicsPSO()) {
        e.hashLo = pso->graphicsPSO()->cacheHashLo;
        e.hashHi = pso->graphicsPSO()->cacheHashHi;
    }
    if ((e.hashLo || e.hashHi) && !exportCachedPSOBlob(e.hashLo, e.hashHi, e.serializedObj)) {
        waitForAllAsyncJITTasks();
        exportCachedPSOBlob(e.hashLo, e.hashHi, e.serializedObj);
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (entries_.count(key)) return E_INVALIDARG;
    entries_.emplace(std::move(key), std::move(e));
    return S_OK;
}

HRESULT STDMETHODCALLTYPE NacellePipelineLibrary::LoadGraphicsPipeline(
    LPCWSTR pName, const D3D12_GRAPHICS_PIPELINE_STATE_DESC* pDesc, REFIID riid, void** ppPipelineState) {
    if (!pName || !pDesc || !ppPipelineState) return E_INVALIDARG;
    std::string key = wideToUtf8(pName);
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!entries_.count(key)) return E_INVALIDARG;
    }
    return parentDevice()->CreateGraphicsPipelineState(pDesc, riid, ppPipelineState);
}

HRESULT STDMETHODCALLTYPE NacellePipelineLibrary::LoadComputePipeline(
    LPCWSTR pName, const D3D12_COMPUTE_PIPELINE_STATE_DESC* pDesc, REFIID riid, void** ppPipelineState) {
    if (!pName || !pDesc || !ppPipelineState) return E_INVALIDARG;
    std::string key = wideToUtf8(pName);
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (!entries_.count(key)) return E_INVALIDARG;
    }
    return parentDevice()->CreateComputePipelineState(pDesc, riid, ppPipelineState);
}

SIZE_T STDMETHODCALLTYPE NacellePipelineLibrary::GetSerializedSize() {
    std::lock_guard<std::mutex> lock(mu_);
    size_t sz = 8;
    for (const auto& [name, e] : entries_) {
        sz += 4 + name.size() + sizeof(bool) + 16 + 4 + e.serializedObj.size();
    }
    return sz;
}

HRESULT STDMETHODCALLTYPE NacellePipelineLibrary::Serialize(void* pData, SIZE_T DataSizeInBytes) {
    if (!pData) return E_INVALIDARG;
    std::lock_guard<std::mutex> lock(mu_);
    BinaryWriter w;
    uint32_t magic = 0x504C4942;
    uint32_t count = static_cast<uint32_t>(entries_.size());
    w.write(magic);
    w.write(count);
    for (const auto& [name, e] : entries_) {
        uint32_t nlen = static_cast<uint32_t>(name.size());
        uint32_t dlen = static_cast<uint32_t>(e.serializedObj.size());
        w.write(nlen);
        w.buf.insert(w.buf.end(), name.begin(), name.end());
        w.write(e.isCompute);
        w.write(e.hashLo);
        w.write(e.hashHi);
        w.write(dlen);
        w.buf.insert(w.buf.end(), e.serializedObj.begin(), e.serializedObj.end());
    }
    if (DataSizeInBytes < w.buf.size()) return E_INVALIDARG;
    std::memcpy(pData, w.buf.data(), w.buf.size());
    return S_OK;
}

extern "C" {

HRESULT NacelleCreateBlob(SIZE_T Size, ID3DBlob** ppBlob) {
    if (!ppBlob) return E_INVALIDARG;
    *ppBlob = new NacelleBlob(Size);
    return S_OK;
}

static HRESULT serializeVersionedInternal(
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignature,
    ID3DBlob** ppBlob) {
    if (!pRootSignature || !ppBlob) return E_INVALIDARG;
    const D3D12_ROOT_PARAMETER1* params = nullptr;
    UINT numParams = 0;
    const D3D12_STATIC_SAMPLER_DESC* samplers = nullptr;
    UINT numSamplers = 0;
    std::vector<D3D12_ROOT_PARAMETER1> convertedParams;
    std::vector<std::vector<D3D12_DESCRIPTOR_RANGE1>> convertedRanges;

    if (pRootSignature->Version == D3D_ROOT_SIGNATURE_VERSION_1_0) {
        const auto& d10 = pRootSignature->Desc_1_0;
        numParams = d10.NumParameters;
        numSamplers = d10.NumStaticSamplers;
        samplers = d10.pStaticSamplers;
        convertedParams.resize(numParams);
        convertedRanges.resize(numParams);
        for (UINT i = 0; i < numParams; ++i) {
            const auto& sp = d10.pParameters[i];
            auto& dp = convertedParams[i];
            dp.ParameterType = sp.ParameterType;
            dp.ShaderVisibility = sp.ShaderVisibility;
            if (sp.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
                dp.DescriptorTable.NumDescriptorRanges = sp.DescriptorTable.NumDescriptorRanges;
                convertedRanges[i].resize(sp.DescriptorTable.NumDescriptorRanges);
                for (UINT j = 0; j < sp.DescriptorTable.NumDescriptorRanges; ++j) {
                    const auto& sr = sp.DescriptorTable.pDescriptorRanges[j];
                    auto& dr = convertedRanges[i][j];
                    dr.RangeType = sr.RangeType;
                    dr.NumDescriptors = sr.NumDescriptors;
                    dr.BaseShaderRegister = sr.BaseShaderRegister;
                    dr.RegisterSpace = sr.RegisterSpace;
                    dr.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
                    dr.OffsetInDescriptorsFromTableStart = sr.OffsetInDescriptorsFromTableStart;
                }
                dp.DescriptorTable.pDescriptorRanges = convertedRanges[i].data();
            } else if (sp.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                dp.Constants = sp.Constants;
            } else {
                dp.Descriptor.ShaderRegister = sp.Descriptor.ShaderRegister;
                dp.Descriptor.RegisterSpace = sp.Descriptor.RegisterSpace;
                dp.Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE;
            }
        }
        params = convertedParams.data();
    } else {
        const auto& d11 = pRootSignature->Desc_1_1;
        numParams = d11.NumParameters;
        params = d11.pParameters;
        numSamplers = d11.NumStaticSamplers;
        samplers = d11.pStaticSamplers;
    }

    BinaryWriter w;
    w.write(kNacelleRootSigMagic);
    w.write(uint32_t(numParams));
    w.write(uint32_t(numSamplers));
    for (UINT i = 0; i < numParams; ++i) {
        const auto& sp = params[i];
        w.write(sp.ParameterType);
        w.write(sp.ShaderVisibility);
        uint32_t reg = 0, space = 0, numVals = 0, numRanges = 0;
        if (sp.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
            reg = sp.Constants.ShaderRegister;
            space = sp.Constants.RegisterSpace;
            numVals = sp.Constants.Num32BitValues;
        } else if (sp.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
            numRanges = sp.DescriptorTable.NumDescriptorRanges;
        } else {
            reg = sp.Descriptor.ShaderRegister;
            space = sp.Descriptor.RegisterSpace;
        }
        w.write(reg);
        w.write(space);
        w.write(numVals);
        w.write(numRanges);
        uint32_t runningOff = 0;
        for (UINT j = 0; j < numRanges; ++j) {
            const auto& sr = sp.DescriptorTable.pDescriptorRanges[j];
            RootSignatureLayout::Param::Range rng;
            rng.type = sr.RangeType;
            rng.numDescriptors = sr.NumDescriptors;
            rng.baseShaderRegister = sr.BaseShaderRegister;
            rng.registerSpace = sr.RegisterSpace;
            rng.offsetInTable = (sr.OffsetInDescriptorsFromTableStart == UINT32_MAX)
                                    ? runningOff
                                    : sr.OffsetInDescriptorsFromTableStart;
            runningOff = rng.offsetInTable + sr.NumDescriptors;
            w.write(rng);
        }
    }
    for (UINT i = 0; i < numSamplers; ++i) {
        const auto& ss = samplers[i];
        RootSignatureLayout::StaticSampler out{};
        out.shaderRegister = ss.ShaderRegister;
        out.registerSpace = ss.RegisterSpace;
        out.visibility = ss.ShaderVisibility;
        out.desc.Filter = ss.Filter;
        out.desc.AddressU = ss.AddressU;
        out.desc.AddressV = ss.AddressV;
        out.desc.AddressW = ss.AddressW;
        out.desc.MipLODBias = ss.MipLODBias;
        out.desc.MaxAnisotropy = ss.MaxAnisotropy;
        out.desc.ComparisonFunc = ss.ComparisonFunc;
        out.desc.MinLOD = ss.MinLOD;
        out.desc.MaxLOD = ss.MaxLOD;
        if (ss.BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK) {
            out.desc.BorderColor[3] = 1.0f;
        } else if (ss.BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE) {
            out.desc.BorderColor[0] = out.desc.BorderColor[1] = out.desc.BorderColor[2] = out.desc.BorderColor[3] = 1.0f;
        }
        w.write(out);
    }
    *ppBlob = new NacelleBlob(w.buf.data(), w.buf.size());
    return S_OK;
}

HRESULT NacelleSerializeRootSignature(
    const D3D12_ROOT_SIGNATURE_DESC* pRootSignature,
    D3D_ROOT_SIGNATURE_VERSION Version,
    ID3DBlob** ppBlob,
    ID3DBlob** ppErrorBlob) {
    if (ppErrorBlob) *ppErrorBlob = nullptr;
    if (!pRootSignature) return E_INVALIDARG;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC vd{};
    vd.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    vd.Desc_1_0 = *pRootSignature;
    (void)Version;
    return serializeVersionedInternal(&vd, ppBlob);
}

HRESULT NacelleSerializeVersionedRootSignature(
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignature,
    ID3DBlob** ppBlob,
    ID3DBlob** ppErrorBlob) {
    if (ppErrorBlob) *ppErrorBlob = nullptr;
    return serializeVersionedInternal(pRootSignature, ppBlob);
}

class NacelleVersionedRootSignatureDeserializer final
    : public ComObject<ID3D12VersionedRootSignatureDeserializer> {
public:
    explicit NacelleVersionedRootSignatureDeserializer(const RootSignatureLayout& layout) {
        params_.resize(layout.params.size());
        rangeStorage_.resize(layout.params.size());
        for (size_t i = 0; i < layout.params.size(); ++i) {
            const auto& sp = layout.params[i];
            auto& dp = params_[i];
            dp.ParameterType = sp.type;
            dp.ShaderVisibility = sp.visibility;
            if (sp.type == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
                dp.Constants.ShaderRegister = sp.shaderRegister;
                dp.Constants.RegisterSpace = sp.registerSpace;
                dp.Constants.Num32BitValues = sp.num32BitValues;
            } else if (sp.type == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
                auto& ranges = rangeStorage_[i];
                ranges.resize(sp.ranges.size());
                for (size_t j = 0; j < sp.ranges.size(); ++j) {
                    const auto& sr = sp.ranges[j];
                    ranges[j].RangeType = sr.type;
                    ranges[j].NumDescriptors = sr.numDescriptors;
                    ranges[j].BaseShaderRegister = sr.baseShaderRegister;
                    ranges[j].RegisterSpace = sr.registerSpace;
                    ranges[j].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
                    ranges[j].OffsetInDescriptorsFromTableStart = sr.offsetInTable;
                }
                dp.DescriptorTable.NumDescriptorRanges = static_cast<UINT>(ranges.size());
                dp.DescriptorTable.pDescriptorRanges = ranges.empty() ? nullptr : ranges.data();
            } else {
                dp.Descriptor.ShaderRegister = sp.shaderRegister;
                dp.Descriptor.RegisterSpace = sp.registerSpace;
                dp.Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE;
            }
        }
        samplers_.resize(layout.staticSamplers.size());
        for (size_t i = 0; i < layout.staticSamplers.size(); ++i) {
            const auto& ss = layout.staticSamplers[i];
            auto& ds = samplers_[i];
            ds.Filter = ss.desc.Filter;
            ds.AddressU = ss.desc.AddressU;
            ds.AddressV = ss.desc.AddressV;
            ds.AddressW = ss.desc.AddressW;
            ds.MipLODBias = ss.desc.MipLODBias;
            ds.MaxAnisotropy = ss.desc.MaxAnisotropy;
            ds.ComparisonFunc = ss.desc.ComparisonFunc;
            ds.MinLOD = ss.desc.MinLOD;
            ds.MaxLOD = ss.desc.MaxLOD;
            ds.ShaderRegister = ss.shaderRegister;
            ds.RegisterSpace = ss.registerSpace;
            ds.ShaderVisibility = ss.visibility;
        }
        desc_.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        desc_.Desc_1_1.NumParameters = static_cast<UINT>(params_.size());
        desc_.Desc_1_1.pParameters = params_.empty() ? nullptr : params_.data();
        desc_.Desc_1_1.NumStaticSamplers = static_cast<UINT>(samplers_.size());
        desc_.Desc_1_1.pStaticSamplers = samplers_.empty() ? nullptr : samplers_.data();
        desc_.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        if (InlineIsEqualGUID(riid, __uuidof(ID3D12VersionedRootSignatureDeserializer))) {
            *ppvObject = static_cast<ID3D12VersionedRootSignatureDeserializer*>(this);
            AddRef();
            return S_OK;
        }
        return ComObject::QueryInterface(riid, ppvObject);
    }

    HRESULT STDMETHODCALLTYPE GetRootSignatureDescAtVersion(
        D3D_ROOT_SIGNATURE_VERSION convertToVersion,
        const D3D12_VERSIONED_ROOT_SIGNATURE_DESC** ppDesc) override {
        if (!ppDesc) return E_POINTER;
        (void)convertToVersion;
        *ppDesc = &desc_;
        return S_OK;
    }

    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* STDMETHODCALLTYPE
    GetUnconvertedRootSignatureDesc() override {
        return &desc_;
    }

private:
    std::vector<D3D12_ROOT_PARAMETER1> params_;
    std::vector<std::vector<D3D12_DESCRIPTOR_RANGE1>> rangeStorage_;
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers_;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc_{};
};

HRESULT NacelleCreateVersionedRootSignatureDeserializer(
    LPCVOID pSrcData,
    SIZE_T SrcDataSizeInBytes,
    REFIID pRootSignatureDeserializerInterface,
    void** ppRootSignatureDeserializer) {
    if (!ppRootSignatureDeserializer) return E_POINTER;
    *ppRootSignatureDeserializer = nullptr;
    RootSignatureLayout layout;
    HRESULT hr = NacelleRootSignature::parseBlob(pSrcData, SrcDataSizeInBytes, layout);
    if (FAILED(hr)) return hr;
    auto* deser = new NacelleVersionedRootSignatureDeserializer(layout);
    hr = deser->QueryInterface(pRootSignatureDeserializerInterface, ppRootSignatureDeserializer);
    deser->Release();
    return hr;
}

#if !defined(_WIN32) || defined(NACELLE_EXPORT_D3D12_SYMBOLS)
HRESULT WINAPI D3D12SerializeRootSignature(
    const D3D12_ROOT_SIGNATURE_DESC* pRootSignature,
    D3D_ROOT_SIGNATURE_VERSION Version,
    ID3DBlob** ppBlob,
    ID3DBlob** ppErrorBlob) {
    return NacelleSerializeRootSignature(pRootSignature, Version, ppBlob, ppErrorBlob);
}

HRESULT WINAPI D3D12SerializeVersionedRootSignature(
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* pRootSignature,
    ID3DBlob** ppBlob,
    ID3DBlob** ppErrorBlob) {
    return NacelleSerializeVersionedRootSignature(pRootSignature, ppBlob, ppErrorBlob);
}

HRESULT WINAPI D3D12CreateVersionedRootSignatureDeserializer(
    LPCVOID pSrcData,
    SIZE_T SrcDataSizeInBytes,
    REFIID pRootSignatureDeserializerInterface,
    void** ppRootSignatureDeserializer) {
    return NacelleCreateVersionedRootSignatureDeserializer(
        pSrcData, SrcDataSizeInBytes, pRootSignatureDeserializerInterface, ppRootSignatureDeserializer);
}
#endif

} // extern "C"

} // namespace nacelle
