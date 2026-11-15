#include "test_common.h"

#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> assembleDxilBitcode(const std::string& irText) {
    llvm::LLVMContext ctx;
    llvm::SMDiagnostic err;
    auto mod = llvm::parseAssemblyString(irText, err, ctx);
    if (!mod) {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        err.print("linalg_test", os);
        throw std::runtime_error("Failed to parse test DXIL IR: " + msg);
    }
    llvm::SmallVector<char, 4096> buf;
    llvm::raw_svector_ostream os(buf);
    llvm::WriteBitcodeToFile(*mod, os);
    return std::vector<uint8_t>(buf.begin(), buf.end());
}

ComPtr<ID3D12RootSignature> createUavSrvRS(ID3D12Device* dev, UINT numSrv, UINT numUav) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    UINT numRanges = 0;
    if (numSrv > 0) {
        ranges[numRanges].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[numRanges].NumDescriptors = numSrv;
        ranges[numRanges].BaseShaderRegister = 0;
        ranges[numRanges].OffsetInDescriptorsFromTableStart = 0;
        numRanges++;
    }
    if (numUav > 0) {
        ranges[numRanges].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[numRanges].NumDescriptors = numUav;
        ranges[numRanges].BaseShaderRegister = 0;
        ranges[numRanges].OffsetInDescriptorsFromTableStart = numSrv;
        numRanges++;
    }
    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = numRanges;
    rp.DescriptorTable.pDescriptorRanges = ranges;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 1;
    desc.pParameters = &rp;
    ComPtr<ID3DBlob> blob, err;
    testutil::checkHR(NacelleSerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, &err), "SerializeRS");
    ComPtr<ID3D12RootSignature> rs;
    testutil::checkHR(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)), "CreateRS");
    return rs;
}

ComPtr<ID3D12PipelineState> createComputePSOFromIR(ID3D12Device* dev, ID3D12RootSignature* rs, const std::string& ir) {
    auto bc = assembleDxilBitcode(ir);
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = rs;
    desc.CS = {bc.data(), bc.size()};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), "CreateComputePipelineState");
    return pso;
}

void createRawBufferUav(ID3D12Device* dev, ID3D12Resource* res, UINT byteSize, D3D12_CPU_DESCRIPTOR_HANDLE cpu) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = byteSize / 4;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    dev->CreateUnorderedAccessView(res, nullptr, &uav, cpu);
}

void createRawBufferSrv(ID3D12Device* dev, ID3D12Resource* res, UINT byteSize, D3D12_CPU_DESCRIPTOR_HANDLE cpu) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = byteSize / 4;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    dev->CreateShaderResourceView(res, &srv, cpu);
}

bool nearF(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps;
}

bool testCheckFeatureSupport(testutil::TestRuntime& rt) {
    std::cout << "=== Test: SM 6.10 CheckFeatureSupport ===\n";
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_10};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm)), "SM");
    if (sm.HighestShaderModel != D3D_SHADER_MODEL_6_10) {
        std::cerr << "Expected D3D_SHADER_MODEL_6_10, got " << std::hex << sm.HighestShaderModel << std::dec << "\n";
        return false;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 opt9{};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &opt9, sizeof(opt9)), "OPTIONS9");
    if (opt9.WaveMMATier != D3D12_WAVE_MMA_TIER_1_0) {
        std::cerr << "Expected D3D12_WAVE_MMA_TIER_1_0\n";
        return false;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS21 opt21{};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21, &opt21, sizeof(opt21)), "OPTIONS21");
    D3D12_FEATURE_DATA_D3D12_OPTIONS22 opt22{};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS22, &opt22, sizeof(opt22)), "OPTIONS22");
    std::cout << "PASS\n";
    return true;
}

bool testDescriptorIO(testutil::TestRuntime& rt) {
    std::cout << "=== Test: DxilConf_SM610_LinAlg_DescriptorIO ===\n";
    const std::string ir = R"IR(
%dx.types.Handle = type { i8* }
%dx.types.ResRet.i32 = type { i32, i32, i32, i32, i32 }
%dx.types.LinAlgMatrixC9M4N4U0S1 = type { i8* }

declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1)
declare %dx.types.LinAlgMatrixC9M4N4U0S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M4N4U0S1(i32, %dx.types.Handle, i32, i32, i32, i32)
declare void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U0S1(i32, %dx.types.LinAlgMatrixC9M4N4U0S1, %dx.types.Handle, i32, i32, i32, i32)
declare void @dx.op.linAlgMatrixAccumulateToDescriptor.C9M4N4U0S1(i32, %dx.types.LinAlgMatrixC9M4N4U0S1, %dx.types.Handle, i32, i32, i32, i32)

define void @main() {
entry:
  %srv = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false)
  %uav = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)
  ; Load 4x4 row-major matrix from srv offset 0, stride 16
  %mat = call %dx.types.LinAlgMatrixC9M4N4U0S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M4N4U0S1(i32 325, %dx.types.Handle %srv, i32 0, i32 16, i32 0, i32 4)
  ; Store column-major (layout=1) to uav offset 0, stride 16
  call void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U0S1(i32 331, %dx.types.LinAlgMatrixC9M4N4U0S1 %mat, %dx.types.Handle %uav, i32 0, i32 16, i32 1, i32 4)
  ; Accumulate row-major (layout=0) to uav offset 64, stride 16
  call void @dx.op.linAlgMatrixAccumulateToDescriptor.C9M4N4U0S1(i32 338, %dx.types.LinAlgMatrixC9M4N4U0S1 %mat, %dx.types.Handle %uav, i32 64, i32 16, i32 0, i32 4)
  ret void
}

!dx.shaderModel = !{!0}
!dx.resources = !{!1}
!dx.entryPoints = !{!6}
!0 = !{!"cs", i32 6, i32 10}
!1 = !{!2, !4, null, null}
!2 = !{!3}
!3 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i32 0, null}
!4 = !{!5}
!5 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!6 = !{void ()* @main, !"main", null, !1, !7}
!7 = !{i32 4, !8}
!8 = !{i32 4, i32 1, i32 1}
)IR";

    auto rs = createUavSrvRS(rt.device.Get(), 1, 1);
    auto pso = createComputePSOFromIR(rt.device.Get(), rs.Get(), ir);
    auto bufIn = rt.createBuffer(64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE);
    auto bufOut = rt.createBuffer(128, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    float inMat[16];
    for (int i = 0; i < 16; ++i) inMat[i] = float(i + 1);
    rt.uploadBuffer(bufIn.Get(), inMat, sizeof(inMat));

    float outInit[32];
    for (int i = 0; i < 16; ++i) outInit[i] = 0.0f;
    for (int i = 16; i < 32; ++i) outInit[i] = 100.0f;
    rt.uploadBuffer(bufOut.Get(), outInit, sizeof(outInit));

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
    createRawBufferSrv(rt.device.Get(), bufIn.Get(), 64, cpu);
    createRawBufferUav(rt.device.Get(), bufOut.Get(), 128, {cpu.ptr + inc});

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto raw = rt.readbackBuffer(bufOut.Get(), 128);
    float out[32];
    std::memcpy(out, raw.data(), sizeof(out));
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (!nearF(out[c * 4 + r], inMat[r * 4 + c])) {
                std::cerr << "Transpose store mismatch at (" << r << "," << c << ")\n";
                return false;
            }
            if (!nearF(out[16 + r * 4 + c], 100.0f + inMat[r * 4 + c])) {
                std::cerr << "Accumulate mismatch at (" << r << "," << c << ")\n";
                return false;
            }
        }
    }
    std::cout << "PASS\n";
    return true;
}

bool testGroupSharedIO(testutil::TestRuntime& rt) {
    std::cout << "=== Test: DxilConf_SM610_LinAlg_GroupSharedIO ===\n";
    const std::string ir = R"IR(
%dx.types.Handle = type { i8* }
%dx.types.LinAlgMatrixC9M4N4U0S1 = type { i8* }
@gs = addrspace(3) global [16 x float] zeroinitializer, align 4

declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1)
declare %dx.types.LinAlgMatrixC9M4N4U0S1 @dx.op.linAlgFillMatrix.C9M4N4U0S1(i32, i1, float)
declare void @dx.op.linAlgMatrixStoreToMemory.C9M4N4U0S1(i32, %dx.types.LinAlgMatrixC9M4N4U0S1, float addrspace(3)*, i32, i32, i32)
declare void @dx.op.linAlgMatrixAccumulateToMemory.C9M4N4U0S1(i32, %dx.types.LinAlgMatrixC9M4N4U0S1, float addrspace(3)*, i32, i32, i32)
declare %dx.types.LinAlgMatrixC9M4N4U0S1 @dx.op.linAlgMatrixLoadFromMemory.C9M4N4U0S1(i32, float addrspace(3)*, i32, i32, i32)
declare void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U0S1(i32, %dx.types.LinAlgMatrixC9M4N4U0S1, %dx.types.Handle, i32, i32, i32, i32)

define void @main() {
entry:
  %uav = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)
  %m1 = call %dx.types.LinAlgMatrixC9M4N4U0S1 @dx.op.linAlgFillMatrix.C9M4N4U0S1(i32 323, i1 false, float 2.5)
  %m2 = call %dx.types.LinAlgMatrixC9M4N4U0S1 @dx.op.linAlgFillMatrix.C9M4N4U0S1(i32 323, i1 false, float 1.5)
  %gsPtr = getelementptr inbounds [16 x float], [16 x float] addrspace(3)* @gs, i32 0, i32 0
  call void @dx.op.linAlgMatrixStoreToMemory.C9M4N4U0S1(i32 332, %dx.types.LinAlgMatrixC9M4N4U0S1 %m1, float addrspace(3)* %gsPtr, i32 0, i32 4, i32 0)
  call void @dx.op.linAlgMatrixAccumulateToMemory.C9M4N4U0S1(i32 339, %dx.types.LinAlgMatrixC9M4N4U0S1 %m2, float addrspace(3)* %gsPtr, i32 0, i32 4, i32 0)
  %loaded = call %dx.types.LinAlgMatrixC9M4N4U0S1 @dx.op.linAlgMatrixLoadFromMemory.C9M4N4U0S1(i32 326, float addrspace(3)* %gsPtr, i32 0, i32 4, i32 0)
  call void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U0S1(i32 331, %dx.types.LinAlgMatrixC9M4N4U0S1 %loaded, %dx.types.Handle %uav, i32 0, i32 16, i32 0, i32 4)
  ret void
}

!dx.shaderModel = !{!0}
!dx.resources = !{!1}
!dx.entryPoints = !{!4}
!0 = !{!"cs", i32 6, i32 10}
!1 = !{null, !2, null, null}
!2 = !{!3}
!3 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!4 = !{void ()* @main, !"main", null, !1, !5}
!5 = !{i32 4, !6}
!6 = !{i32 4, i32 1, i32 1}
)IR";

    auto rs = createUavSrvRS(rt.device.Get(), 0, 1);
    auto pso = createComputePSOFromIR(rt.device.Get(), rs.Get(), ir);
    auto bufOut = rt.createBuffer(64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    createRawBufferUav(rt.device.Get(), bufOut.Get(), 64, heap->GetCPUDescriptorHandleForHeapStart());

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto raw = rt.readbackBuffer(bufOut.Get(), 64);
    float out[16];
    std::memcpy(out, raw.data(), sizeof(out));
    for (int i = 0; i < 16; ++i) {
        if (!nearF(out[i], 4.0f)) {
            std::cerr << "GroupSharedIO mismatch at " << i << ": " << out[i] << "\n";
            return false;
        }
    }
    std::cout << "PASS\n";
    return true;
}

bool testElementAccess(testutil::TestRuntime& rt) {
    std::cout << "=== Test: DxilConf_SM610_LinAlg_ElementAccess ===\n";
    const std::string ir = R"IR(
%dx.types.Handle = type { i8* }
%dx.types.LinAlgMatrixC9M4N4U2S1 = type { i8* }

declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1)
declare %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgFillMatrix.C9M4N4U2S1(i32, i1, float)
declare i32 @dx.op.linAlgMatrixLength.C9M4N4U2S1(i32, %dx.types.LinAlgMatrixC9M4N4U2S1)
declare <2 x i32> @dx.op.linAlgMatrixGetCoordinate.C9M4N4U2S1(i32, %dx.types.LinAlgMatrixC9M4N4U2S1, i32)
declare float @dx.op.linAlgMatrixGetElement.C9M4N4U2S1(i32, %dx.types.LinAlgMatrixC9M4N4U2S1, i32)
declare %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgMatrixSetElement.C9M4N4U2S1(i32, %dx.types.LinAlgMatrixC9M4N4U2S1, i32, float)
declare void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U2S1(i32, %dx.types.LinAlgMatrixC9M4N4U2S1, %dx.types.Handle, i32, i32, i32, i32)

define void @main() {
entry:
  %uav = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)
  %m0 = call %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgFillMatrix.C9M4N4U2S1(i32 323, i1 false, float 1.0)
  %len = call i32 @dx.op.linAlgMatrixLength.C9M4N4U2S1(i32 327, %dx.types.LinAlgMatrixC9M4N4U2S1 %m0)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %cur = phi %dx.types.LinAlgMatrixC9M4N4U2S1 [ %m0, %entry ], [ %updated, %loop ]
  %coord = call <2 x i32> @dx.op.linAlgMatrixGetCoordinate.C9M4N4U2S1(i32 328, %dx.types.LinAlgMatrixC9M4N4U2S1 %cur, i32 %i)
  %col = extractelement <2 x i32> %coord, i32 0
  %row = extractelement <2 x i32> %coord, i32 1
  %old = call float @dx.op.linAlgMatrixGetElement.C9M4N4U2S1(i32 329, %dx.types.LinAlgMatrixC9M4N4U2S1 %cur, i32 %i)
  %rc = mul i32 %row, 4
  %lin = add i32 %rc, %col
  %linF = sitofp i32 %lin to float
  %val = fadd float %old, %linF
  %updated = call %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgMatrixSetElement.C9M4N4U2S1(i32 330, %dx.types.LinAlgMatrixC9M4N4U2S1 %cur, i32 %i, float %val)
  %next = add i32 %i, 1
  %cond = icmp ult i32 %next, %len
  br i1 %cond, label %loop, label %exit

exit:
  call void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U2S1(i32 331, %dx.types.LinAlgMatrixC9M4N4U2S1 %updated, %dx.types.Handle %uav, i32 0, i32 16, i32 0, i32 4)
  ret void
}

!dx.shaderModel = !{!0}
!dx.resources = !{!1}
!dx.entryPoints = !{!4}
!0 = !{!"cs", i32 6, i32 10}
!1 = !{null, !2, null, null}
!2 = !{!3}
!3 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!4 = !{void ()* @main, !"main", null, !1, !5}
!5 = !{i32 4, !6}
!6 = !{i32 4, i32 1, i32 1}
)IR";

    auto rs = createUavSrvRS(rt.device.Get(), 0, 1);
    auto pso = createComputePSOFromIR(rt.device.Get(), rs.Get(), ir);
    auto bufOut = rt.createBuffer(64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    createRawBufferUav(rt.device.Get(), bufOut.Get(), 64, heap->GetCPUDescriptorHandleForHeapStart());

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto raw = rt.readbackBuffer(bufOut.Get(), 64);
    float out[16];
    std::memcpy(out, raw.data(), sizeof(out));
    for (int i = 0; i < 16; ++i) {
        if (!nearF(out[i], float(i + 1))) {
            std::cerr << "ElementAccess mismatch at " << i << ": " << out[i] << "\n";
            return false;
        }
    }
    std::cout << "PASS\n";
    return true;
}

bool testConversion(testutil::TestRuntime& rt) {
    std::cout << "=== Test: DxilConf_SM610_LinAlg_Conversion ===\n";
    const std::string ir = R"IR(
%dx.types.Handle = type { i8* }
%dx.types.LinAlgMatrixC9M2N2U0S0 = type { i8* }
%dx.types.LinAlgMatrixC21M2N2U0S0 = type { i8* }

declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1)
declare <4 x i8> @dx.op.linAlgConvert.v4i8.v4f32(i32, <4 x float>, i32, i32)
declare <4 x float> @dx.op.linAlgConvert.v4f32.v4i8(i32, <4 x i8>, i32, i32)
declare <4 x i16> @dx.op.linAlgConvert.v4i16.v4f32(i32, <4 x float>, i32, i32)
declare <4 x float> @dx.op.linAlgConvert.v4f32.v4i16(i32, <4 x i16>, i32, i32)
declare %dx.types.LinAlgMatrixC9M2N2U0S0 @dx.op.linAlgMatrixLoadFromDescriptor.C9M2N2U0S0(i32, %dx.types.Handle, i32, i32, i32, i32)
declare %dx.types.LinAlgMatrixC21M2N2U0S0 @dx.op.linAlgCopyConvertMatrix.C21M2N2U0S0.C9M2N2U0S0(i32, %dx.types.LinAlgMatrixC9M2N2U0S0, i1)
declare %dx.types.LinAlgMatrixC9M2N2U0S0 @dx.op.linAlgCopyConvertMatrix.C9M2N2U0S0.C21M2N2U0S0(i32, %dx.types.LinAlgMatrixC21M2N2U0S0, i1)
declare void @dx.op.linAlgMatrixStoreToDescriptor.C9M2N2U0S0(i32, %dx.types.LinAlgMatrixC9M2N2U0S0, %dx.types.Handle, i32, i32, i32, i32)
declare void @dx.op.rawBufferStore.f32(i32, %dx.types.Handle, i32, i32, float, float, float, float, i8, i32)
declare void @dx.op.rawBufferStore.i32(i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8, i32)

define void @main() {
entry:
  %srv = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false)
  %uav = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)
  ; F32 -> F8_E4M3FN (21) -> F32 (9)
  %f8e4 = call <4 x i8> @dx.op.linAlgConvert.v4i8.v4f32(i32 341, <4 x float> <float 1.5, float -2.0, float 0.25, float 448.0>, i32 9, i32 21)
  %rtE4 = call <4 x float> @dx.op.linAlgConvert.v4f32.v4i8(i32 341, <4 x i8> %f8e4, i32 21, i32 9)
  %e0 = extractelement <4 x float> %rtE4, i32 0
  %e1 = extractelement <4 x float> %rtE4, i32 1
  %e2 = extractelement <4 x float> %rtE4, i32 2
  %e3 = extractelement <4 x float> %rtE4, i32 3
  call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %uav, i32 0, i32 undef, float %e0, float %e1, float %e2, float %e3, i8 15, i32 4)

  ; F32 -> F8_E5M2 (22) -> F32 (9)
  %f8e5 = call <4 x i8> @dx.op.linAlgConvert.v4i8.v4f32(i32 341, <4 x float> <float 1.5, float -4.0, float 0.5, float 16.0>, i32 9, i32 22)
  %rtE5 = call <4 x float> @dx.op.linAlgConvert.v4f32.v4i8(i32 341, <4 x i8> %f8e5, i32 22, i32 9)
  %d0 = extractelement <4 x float> %rtE5, i32 0
  %d1 = extractelement <4 x float> %rtE5, i32 1
  %d2 = extractelement <4 x float> %rtE5, i32 2
  %d3 = extractelement <4 x float> %rtE5, i32 3
  call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %uav, i32 16, i32 undef, float %d0, float %d1, float %d2, float %d3, i8 15, i32 4)

  ; F32 -> BFloat16 (23) -> F32 (9)
  %bf16 = call <4 x i16> @dx.op.linAlgConvert.v4i16.v4f32(i32 341, <4 x float> <float 3.5, float -1.25, float 0.0, float 64.0>, i32 9, i32 23)
  %rtBf = call <4 x float> @dx.op.linAlgConvert.v4f32.v4i16(i32 341, <4 x i16> %bf16, i32 23, i32 9)
  %b0 = extractelement <4 x float> %rtBf, i32 0
  %b1 = extractelement <4 x float> %rtBf, i32 1
  %b2 = extractelement <4 x float> %rtBf, i32 2
  %b3 = extractelement <4 x float> %rtBf, i32 3
  call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %uav, i32 32, i32 undef, float %b0, float %b1, float %b2, float %b3, i8 15, i32 4)

  ; Saturating F32 -> I8 (19)
  %i8v = call <4 x i8> @dx.op.linAlgConvert.v4i8.v4f32(i32 341, <4 x float> <float 500.0, float -500.0, float 42.0, float -12.0>, i32 9, i32 19)
  %i0 = sext i8 extractelement (<4 x i8> %i8v, i32 0) to i32
  %s0 = extractelement <4 x i8> %i8v, i32 0
  %s1 = extractelement <4 x i8> %i8v, i32 1
  %s2 = extractelement <4 x i8> %i8v, i32 2
  %s3 = extractelement <4 x i8> %i8v, i32 3
  %x0 = sext i8 %s0 to i32
  %x1 = sext i8 %s1 to i32
  %x2 = sext i8 %s2 to i32
  %x3 = sext i8 %s3 to i32
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %uav, i32 48, i32 undef, i32 %x0, i32 %x1, i32 %x2, i32 %x3, i8 15, i32 4)

  ; CopyConvertMatrix with transpose
  %mIn = call %dx.types.LinAlgMatrixC9M2N2U0S0 @dx.op.linAlgMatrixLoadFromDescriptor.C9M2N2U0S0(i32 325, %dx.types.Handle %srv, i32 0, i32 8, i32 0, i32 4)
  %mF8 = call %dx.types.LinAlgMatrixC21M2N2U0S0 @dx.op.linAlgCopyConvertMatrix.C21M2N2U0S0.C9M2N2U0S0(i32 324, %dx.types.LinAlgMatrixC9M2N2U0S0 %mIn, i1 true)
  %mOut = call %dx.types.LinAlgMatrixC9M2N2U0S0 @dx.op.linAlgCopyConvertMatrix.C9M2N2U0S0.C21M2N2U0S0(i32 324, %dx.types.LinAlgMatrixC21M2N2U0S0 %mF8, i1 false)
  call void @dx.op.linAlgMatrixStoreToDescriptor.C9M2N2U0S0(i32 331, %dx.types.LinAlgMatrixC9M2N2U0S0 %mOut, %dx.types.Handle %uav, i32 64, i32 8, i32 0, i32 4)
  ret void
}

!dx.shaderModel = !{!0}
!dx.resources = !{!1}
!dx.entryPoints = !{!6}
!0 = !{!"cs", i32 6, i32 10}
!1 = !{!2, !4, null, null}
!2 = !{!3}
!3 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i32 0, null}
!4 = !{!5}
!5 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!6 = !{void ()* @main, !"main", null, !1, !7}
!7 = !{i32 4, !8}
!8 = !{i32 1, i32 1, i32 1}
)IR";
    // Fix the stray constant expression line `%i0 = sext i8 extractelement...`
    std::string cleanIr = ir;
    size_t pos = cleanIr.find("  %i0 = sext");
    if (pos != std::string::npos) {
        size_t nl = cleanIr.find('\n', pos);
        cleanIr.erase(pos, nl - pos + 1);
    }

    auto rs = createUavSrvRS(rt.device.Get(), 1, 1);
    auto pso = createComputePSOFromIR(rt.device.Get(), rs.Get(), cleanIr);
    auto bufIn = rt.createBuffer(16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE);
    auto bufOut = rt.createBuffer(96, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    float inMat[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    rt.uploadBuffer(bufIn.Get(), inMat, sizeof(inMat));

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
    createRawBufferSrv(rt.device.Get(), bufIn.Get(), 16, cpu);
    createRawBufferUav(rt.device.Get(), bufOut.Get(), 96, {cpu.ptr + inc});

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto raw = rt.readbackBuffer(bufOut.Get(), 96);
    float fOut[24];
    int32_t iOut[24];
    std::memcpy(fOut, raw.data(), sizeof(fOut));
    std::memcpy(iOut, raw.data(), sizeof(iOut));

    if (!nearF(fOut[0], 1.5f) || !nearF(fOut[1], -2.0f) || !nearF(fOut[2], 0.25f) || !nearF(fOut[3], 448.0f)) {
        std::cerr << "F8_E4M3FN mismatch\n";
        return false;
    }
    if (!nearF(fOut[4], 1.5f) || !nearF(fOut[5], -4.0f) || !nearF(fOut[6], 0.5f) || !nearF(fOut[7], 16.0f)) {
        std::cerr << "F8_E5M2 mismatch\n";
        return false;
    }
    if (!nearF(fOut[8], 3.5f) || !nearF(fOut[9], -1.25f) || !nearF(fOut[10], 0.0f) || !nearF(fOut[11], 64.0f)) {
        std::cerr << "BFloat16 mismatch\n";
        return false;
    }
    if (iOut[12] != 127 || iOut[13] != -128 || iOut[14] != 42 || iOut[15] != -12) {
        std::cerr << "Saturating I8 mismatch\n";
        return false;
    }
    if (!nearF(fOut[16], 1.0f) || !nearF(fOut[17], 3.0f) || !nearF(fOut[18], 2.0f) || !nearF(fOut[19], 4.0f)) {
        std::cerr << "CopyConvertMatrix transpose mismatch\n";
        return false;
    }
    std::cout << "PASS\n";
    return true;
}

bool testMatrixArithmetic(testutil::TestRuntime& rt) {
    std::cout << "=== Test: DxilConf_SM610_LinAlg_MatrixArithmetic ===\n";
    const std::string ir = R"IR(
%dx.types.Handle = type { i8* }
%dx.types.LinAlgMatrixC9M4N4U0S1 = type { i8* }
%dx.types.LinAlgMatrixC9M4N4U1S1 = type { i8* }
%dx.types.LinAlgMatrixC9M4N4U2S1 = type { i8* }

declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1)
declare i32 @dx.op.linAlgMatrixQueryAccumulatorLayout(i32)
declare %dx.types.LinAlgMatrixC9M4N4U0S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M4N4U0S1(i32, %dx.types.Handle, i32, i32, i32, i32)
declare %dx.types.LinAlgMatrixC9M4N4U1S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M4N4U1S1(i32, %dx.types.Handle, i32, i32, i32, i32)
declare %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgFillMatrix.C9M4N4U2S1(i32, i1, float)
declare %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgMatrixMultiply.C9M4N4U2S1.C9M4N4U0S1.C9M4N4U1S1(i32, %dx.types.LinAlgMatrixC9M4N4U0S1, %dx.types.LinAlgMatrixC9M4N4U1S1)
declare %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgMatrixMultiplyAccumulate.C9M4N4U2S1.C9M4N4U0S1.C9M4N4U1S1.C9M4N4U2S1(i32, %dx.types.LinAlgMatrixC9M4N4U0S1, %dx.types.LinAlgMatrixC9M4N4U1S1, %dx.types.LinAlgMatrixC9M4N4U2S1)
declare %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgMatrixAccumulate.C9M4N4U2S1.C9M4N4U2S1.C9M4N4U0S1(i32, %dx.types.LinAlgMatrixC9M4N4U2S1, %dx.types.LinAlgMatrixC9M4N4U0S1)
declare void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U2S1(i32, %dx.types.LinAlgMatrixC9M4N4U2S1, %dx.types.Handle, i32, i32, i32, i32)

define void @main() {
entry:
  %srv = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false)
  %uav = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)
  %ql = call i32 @dx.op.linAlgMatrixQueryAccumulatorLayout(i32 333)
  %matA = call %dx.types.LinAlgMatrixC9M4N4U0S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M4N4U0S1(i32 325, %dx.types.Handle %srv, i32 0, i32 16, i32 0, i32 4)
  %matB = call %dx.types.LinAlgMatrixC9M4N4U1S1 @dx.op.linAlgMatrixLoadFromDescriptor.C9M4N4U1S1(i32 325, %dx.types.Handle %srv, i32 64, i32 16, i32 0, i32 4)
  %matC = call %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgFillMatrix.C9M4N4U2S1(i32 323, i1 false, float 10.0)
  %ab = call %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgMatrixMultiply.C9M4N4U2S1.C9M4N4U0S1.C9M4N4U1S1(i32 334, %dx.types.LinAlgMatrixC9M4N4U0S1 %matA, %dx.types.LinAlgMatrixC9M4N4U1S1 %matB)
  %abc = call %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgMatrixMultiplyAccumulate.C9M4N4U2S1.C9M4N4U0S1.C9M4N4U1S1.C9M4N4U2S1(i32 322, %dx.types.LinAlgMatrixC9M4N4U0S1 %matA, %dx.types.LinAlgMatrixC9M4N4U1S1 %matB, %dx.types.LinAlgMatrixC9M4N4U2S1 %matC)
  %acc = call %dx.types.LinAlgMatrixC9M4N4U2S1 @dx.op.linAlgMatrixAccumulate.C9M4N4U2S1.C9M4N4U2S1.C9M4N4U0S1(i32 335, %dx.types.LinAlgMatrixC9M4N4U2S1 %ab, %dx.types.LinAlgMatrixC9M4N4U0S1 %matA)
  call void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U2S1(i32 331, %dx.types.LinAlgMatrixC9M4N4U2S1 %abc, %dx.types.Handle %uav, i32 0, i32 16, i32 0, i32 4)
  call void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U2S1(i32 331, %dx.types.LinAlgMatrixC9M4N4U2S1 %acc, %dx.types.Handle %uav, i32 64, i32 16, i32 0, i32 4)
  ret void
}

!dx.shaderModel = !{!0}
!dx.resources = !{!1}
!dx.entryPoints = !{!6}
!0 = !{!"cs", i32 6, i32 10}
!1 = !{!2, !4, null, null}
!2 = !{!3}
!3 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i32 0, null}
!4 = !{!5}
!5 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!6 = !{void ()* @main, !"main", null, !1, !7}
!7 = !{i32 4, !8}
!8 = !{i32 4, i32 1, i32 1}
)IR";

    auto rs = createUavSrvRS(rt.device.Get(), 1, 1);
    auto pso = createComputePSOFromIR(rt.device.Get(), rs.Get(), ir);
    auto bufIn = rt.createBuffer(128, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE);
    auto bufOut = rt.createBuffer(128, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    float inBuf[32];
    for (int i = 0; i < 16; ++i) {
        inBuf[i] = float(i + 1);
        inBuf[16 + i] = float((i % 4 == i / 4) ? 2.0f : 0.0f); // 2 * I
    }
    rt.uploadBuffer(bufIn.Get(), inBuf, sizeof(inBuf));

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
    createRawBufferSrv(rt.device.Get(), bufIn.Get(), 128, cpu);
    createRawBufferUav(rt.device.Get(), bufOut.Get(), 128, {cpu.ptr + inc});

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto raw = rt.readbackBuffer(bufOut.Get(), 128);
    float out[32];
    std::memcpy(out, raw.data(), sizeof(out));
    for (int i = 0; i < 16; ++i) {
        if (!nearF(out[i], 2.0f * inBuf[i] + 10.0f)) {
            std::cerr << "MatMatMulAccum mismatch at " << i << "\n";
            return false;
        }
        if (!nearF(out[16 + i], 3.0f * inBuf[i])) {
            std::cerr << "MatAccum mismatch at " << i << "\n";
            return false;
        }
    }
    std::cout << "PASS\n";
    return true;
}

bool testMatVecAndOuterProduct(testutil::TestRuntime& rt) {
    std::cout << "=== Test: DxilConf_SM610_LinAlg_MatVec & OuterVectorAccumulation ===\n";
    const std::string ir = R"IR(
%dx.types.Handle = type { i8* }
%dx.types.LinAlgMatrixC9M4N4U0S0 = type { i8* }
%dx.types.LinAlgMatrixC9M4N4U2S0 = type { i8* }

declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1)
declare %dx.types.LinAlgMatrixC9M4N4U0S0 @dx.op.linAlgMatrixLoadFromDescriptor.C9M4N4U0S0(i32, %dx.types.Handle, i32, i32, i32, i32)
declare <4 x float> @dx.op.linAlgMatVecMul.v4f32.C9M4N4U0S0.v4f32(i32, %dx.types.LinAlgMatrixC9M4N4U0S0, i1, <4 x float>, i32)
declare <4 x float> @dx.op.linAlgMatVecMulAdd.v4f32.C9M4N4U0S0.v4f32.v4f32(i32, %dx.types.LinAlgMatrixC9M4N4U0S0, i1, <4 x float>, i32, <4 x float>)
declare %dx.types.LinAlgMatrixC9M4N4U2S0 @dx.op.linAlgMatrixOuterProduct.C9M4N4U2S0.v4f32.v4f32(i32, i1, <4 x float>, <4 x float>)
declare void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U2S0(i32, %dx.types.LinAlgMatrixC9M4N4U2S0, %dx.types.Handle, i32, i32, i32, i32)
declare void @dx.op.linAlgVectorAccumulateToDescriptor.v4f32(i32, %dx.types.Handle, i32, i32, <4 x float>)
declare void @dx.op.rawBufferStore.f32(i32, %dx.types.Handle, i32, i32, float, float, float, float, i8, i32)

define void @main() {
entry:
  %srv = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false)
  %uav = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)
  %mat = call %dx.types.LinAlgMatrixC9M4N4U0S0 @dx.op.linAlgMatrixLoadFromDescriptor.C9M4N4U0S0(i32 325, %dx.types.Handle %srv, i32 0, i32 16, i32 0, i32 4)
  %mv = call <4 x float> @dx.op.linAlgMatVecMul.v4f32.C9M4N4U0S0.v4f32(i32 336, %dx.types.LinAlgMatrixC9M4N4U0S0 %mat, i1 false, <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, i32 9)
  %mva = call <4 x float> @dx.op.linAlgMatVecMulAdd.v4f32.C9M4N4U0S0.v4f32.v4f32(i32 337, %dx.types.LinAlgMatrixC9M4N4U0S0 %mat, i1 false, <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, i32 9, <4 x float> <float 10.0, float 20.0, float 30.0, float 40.0>)
  %m0 = extractelement <4 x float> %mv, i32 0
  %m1 = extractelement <4 x float> %mv, i32 1
  %m2 = extractelement <4 x float> %mv, i32 2
  %m3 = extractelement <4 x float> %mv, i32 3
  call void @dx.op.rawBufferStore.f32(i32 140, %dx.types.Handle %uav, i32 0, i32 undef, float %m0, float %m1, float %m2, float %m3, i8 15, i32 4)
  call void @dx.op.linAlgVectorAccumulateToDescriptor.v4f32(i32 342, %dx.types.Handle %uav, i32 16, i32 4, <4 x float> %mva)
  %opMat = call %dx.types.LinAlgMatrixC9M4N4U2S0 @dx.op.linAlgMatrixOuterProduct.C9M4N4U2S0.v4f32.v4f32(i32 340, i1 false, <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, <4 x float> <float 2.0, float 3.0, float 4.0, float 5.0>)
  call void @dx.op.linAlgMatrixStoreToDescriptor.C9M4N4U2S0(i32 331, %dx.types.LinAlgMatrixC9M4N4U2S0 %opMat, %dx.types.Handle %uav, i32 32, i32 16, i32 0, i32 4)
  ret void
}

!dx.shaderModel = !{!0}
!dx.resources = !{!1}
!dx.entryPoints = !{!6}
!0 = !{!"cs", i32 6, i32 10}
!1 = !{!2, !4, null, null}
!2 = !{!3}
!3 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i32 0, null}
!4 = !{!5}
!5 = !{i32 0, i32 0, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!6 = !{void ()* @main, !"main", null, !1, !7}
!7 = !{i32 4, !8}
!8 = !{i32 1, i32 1, i32 1}
)IR";

    auto rs = createUavSrvRS(rt.device.Get(), 1, 1);
    auto pso = createComputePSOFromIR(rt.device.Get(), rs.Get(), ir);
    auto bufIn = rt.createBuffer(64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE);
    auto bufOut = rt.createBuffer(96, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    float inMat[16];
    for (int i = 0; i < 16; ++i) inMat[i] = float(i + 1);
    rt.uploadBuffer(bufIn.Get(), inMat, sizeof(inMat));

    float outInit[24]{};
    for (int i = 4; i < 8; ++i) outInit[i] = 1.0f; // base for VectorAccumulateToDescriptor
    rt.uploadBuffer(bufOut.Get(), outInit, sizeof(outInit));

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
    createRawBufferSrv(rt.device.Get(), bufIn.Get(), 64, cpu);
    createRawBufferUav(rt.device.Get(), bufOut.Get(), 96, {cpu.ptr + inc});

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto raw = rt.readbackBuffer(bufOut.Get(), 96);
    float out[24];
    std::memcpy(out, raw.data(), sizeof(out));

    // Row 0: 1*1+2*2+3*3+4*4 = 30; Row 1: 5*1+6*2+7*3+8*4 = 70; Row 2: 110; Row 3: 150
    const float expMv[4] = {30.0f, 70.0f, 110.0f, 150.0f};
    for (int r = 0; r < 4; ++r) {
        if (!nearF(out[r], expMv[r])) {
            std::cerr << "MatVecMul mismatch at " << r << "\n";
            return false;
        }
        if (!nearF(out[4 + r], 1.0f + expMv[r] + float((r + 1) * 10))) {
            std::cerr << "MatVecMulAdd + VectorAccumulate mismatch at " << r << "\n";
            return false;
        }
        for (int c = 0; c < 4; ++c) {
            if (!nearF(out[8 + r * 4 + c], float((r + 1) * (c + 2)))) {
                std::cerr << "OuterProduct mismatch at (" << r << "," << c << ")\n";
                return false;
            }
        }
    }
    std::cout << "PASS\n";
    return true;
}

} // namespace

int main() {
    std::cout << std::unitbuf;
    testutil::TestRuntime rt;
    rt.init();
    bool ok = true;
    ok &= testCheckFeatureSupport(rt);
    ok &= testDescriptorIO(rt);
    ok &= testGroupSharedIO(rt);
    ok &= testElementAccess(rt);
    ok &= testConversion(rt);
    ok &= testMatrixArithmetic(rt);
    ok &= testMatVecAndOuterProduct(rt);
    return ok ? 0 : 1;
}
