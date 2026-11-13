find_program(DXC_EXECUTABLE dxc HINTS /usr/local/bin /opt/homebrew/bin REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(FILECHECK_EXECUTABLE FileCheck HINTS "${LLVM_TOOLS_BINARY_DIR}" /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin REQUIRED)
find_program(SPLIT_FILE_EXECUTABLE split-file HINTS "${LLVM_TOOLS_BINARY_DIR}" /opt/homebrew/opt/llvm/bin /usr/local/opt/llvm/bin REQUIRED)

set(NACELLE_SHADER_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(NACELLE_SHADER_OUTPUTS "")

function(add_hlsl_dxil hlsl_name profile entry out_name)
    set(src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${hlsl_name}")
    set(dst "${NACELLE_SHADER_BIN_DIR}/${out_name}")
    add_custom_command(
        OUTPUT "${dst}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${NACELLE_SHADER_BIN_DIR}"
        COMMAND "${DXC_EXECUTABLE}" -T ${profile} -E ${entry} -O3 -Fo "${dst}" "${src}"
        DEPENDS "${src}"
        COMMENT "Compiling HLSL -> DXIL: ${hlsl_name} (${profile}, ${entry}) -> ${out_name}"
    )
    list(APPEND NACELLE_SHADER_OUTPUTS "${dst}")
    set(NACELLE_SHADER_OUTPUTS "${NACELLE_SHADER_OUTPUTS}" PARENT_SCOPE)
endfunction()

function(add_hlsl_spirv hlsl_name profile entry out_name)
    set(src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${hlsl_name}")
    set(dst "${NACELLE_SHADER_BIN_DIR}/${out_name}")
    add_custom_command(
        OUTPUT "${dst}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${NACELLE_SHADER_BIN_DIR}"
        COMMAND "${DXC_EXECUTABLE}" -spirv -fspv-target-env=vulkan1.1
                -fvk-t-shift 0 0 -fvk-u-shift 8 0 -fvk-b-shift 16 0
                -T ${profile} -E ${entry} -O3 -Fo "${dst}" "${src}"
        DEPENDS "${src}"
        COMMENT "Compiling HLSL -> SPIR-V: ${hlsl_name} (${profile}, ${entry}) -> ${out_name}"
    )
    list(APPEND NACELLE_SHADER_OUTPUTS "${dst}")
    set(NACELLE_SHADER_OUTPUTS "${NACELLE_SHADER_OUTPUTS}" PARENT_SCOPE)
endfunction()

# Compute shaders (DXIL)
add_hlsl_dxil(vec_add.hlsl cs_6_0 CSMain vec_add.dxil)
add_hlsl_dxil(matmul.hlsl cs_6_0 CSMain matmul.dxil)
add_hlsl_dxil(bindless_sm66.hlsl cs_6_6 CSMain bindless_sm66.dxil)
add_hlsl_dxil(divergent_cf_atomics.hlsl cs_6_0 CSMain divergent_cf_atomics.dxil)
add_hlsl_dxil(wave_intrinsics.hlsl cs_6_0 CSMain wave_intrinsics.dxil)
add_hlsl_dxil(prefix_sum_groupshared.hlsl cs_6_0 CSMain prefix_sum_groupshared.dxil)
add_hlsl_dxil(interlocked_barrier.hlsl cs_6_6 CSMain interlocked_barrier.dxil)
add_hlsl_dxil(matrix_swizzle.hlsl cs_6_0 CSMain matrix_swizzle.dxil)
add_hlsl_dxil(math_denorms_bits.hlsl cs_6_5 CSMain math_denorms_bits.dxil)
add_hlsl_dxil(byte_address_get_dims.hlsl cs_6_5 CSMain byte_address_get_dims.dxil)
add_hlsl_dxil(uav_counter_dec.hlsl cs_6_0 CSMain uav_counter_dec.dxil)
add_hlsl_dxil(bit_fpclass_intrinsics.hlsl cs_6_5 CSMain bit_fpclass_intrinsics.dxil)
add_hlsl_dxil(quad_derivatives.hlsl vs_6_0 VSMain quad_derivatives_vs.dxil)
add_hlsl_dxil(quad_derivatives.hlsl ps_6_0 PSMain quad_derivatives_ps.dxil)
add_hlsl_dxil(texture_array_ops.hlsl cs_6_0 CSMain texture_array_ops.dxil)
add_hlsl_dxil(depth_stencil_readback.hlsl vs_6_0 VSMain depth_stencil_readback_vs.dxil)
add_hlsl_dxil(depth_stencil_readback.hlsl ps_6_0 PSMain depth_stencil_readback_ps.dxil)

# Cross-backend benchmark shader (DXIL + SPIR-V)
add_hlsl_dxil(bench_raymarch.hlsl cs_6_0 main bench_raymarch.dxil)
add_hlsl_spirv(bench_raymarch.hlsl cs_6_0 main bench_raymarch.spv)

# Graphics shaders (DXIL)
add_hlsl_dxil(raster_basic.hlsl vs_6_0 VSMain raster_basic_vs.dxil)
add_hlsl_dxil(raster_basic.hlsl ps_6_0 PSMain raster_basic_ps.dxil)
add_hlsl_dxil(shadow_depth.hlsl vs_6_0 VSMain shadow_depth_vs.dxil)
add_hlsl_dxil(pbr_mrt.hlsl vs_6_0 VSMain pbr_mrt_vs.dxil)
add_hlsl_dxil(pbr_mrt.hlsl ps_6_0 PSMain pbr_mrt_ps.dxil)
add_hlsl_dxil(fullscreen_quad.hlsl vs_6_0 VSMain fullscreen_quad_vs.dxil)
add_hlsl_dxil(graphics_features.hlsl vs_6_4 FullscreenVS gfx_fullscreen_vs.dxil)
add_hlsl_dxil(graphics_features.hlsl vs_6_4 BaryVS gfx_bary_vs.dxil)
add_hlsl_dxil(graphics_features.hlsl ps_6_4 BaryPerspPS gfx_bary_persp_ps.dxil)
add_hlsl_dxil(graphics_features.hlsl ps_6_4 BaryNoPerspPS gfx_bary_nopersp_ps.dxil)
add_hlsl_dxil(graphics_features.hlsl ps_6_4 StencilRefPS gfx_stencil_ref_ps.dxil)
add_hlsl_dxil(graphics_features.hlsl ps_6_4 ShadingRatePS gfx_shading_rate_ps.dxil)
add_hlsl_dxil(graphics_features.hlsl vs_6_4 MultiVPVS gfx_multivp_vs.dxil)
add_hlsl_dxil(graphics_features.hlsl ps_6_4 MultiVPPS gfx_multivp_ps.dxil)
add_hlsl_dxil(graphics_features.hlsl ps_6_4 LodClampPS gfx_lod_clamp_ps.dxil)

# Converted Shadertoy pixel shaders (DXIL)
add_hlsl_dxil(light_cycle.hlsl ps_6_0 PSMain light_cycle.dxil)
add_hlsl_dxil(aurora_paint.hlsl ps_6_0 PSMain aurora_paint.dxil)
add_hlsl_dxil(sunflower_fields.hlsl ps_6_0 PSMain sunflower_fields.dxil)
add_hlsl_dxil(rainbow_travel.hlsl ps_6_0 PSMain rainbow_travel.dxil)
add_hlsl_dxil(mt_fuji_watercolor.hlsl ps_6_0 PSMain mt_fuji_watercolor.dxil)
add_hlsl_dxil(van_gogh_sunset.hlsl ps_6_0 PSMain van_gogh_sunset.dxil)
add_hlsl_dxil(rainbow_particle_hills.hlsl ps_6_0 PSMain rainbow_particle_hills.dxil)
add_hlsl_dxil(goldeen.hlsl ps_6_0 PSMain goldeen.dxil)
add_hlsl_dxil(disco_maze.hlsl ps_6_0 PSMain disco_maze.dxil)

# Converted Shadertoy pixel shaders (DXIL)
add_hlsl_dxil(light_cycle.hlsl ps_6_0 PSMain light_cycle.dxil)
add_hlsl_dxil(aurora_paint.hlsl ps_6_0 PSMain aurora_paint.dxil)
add_hlsl_dxil(sunflower_fields.hlsl ps_6_0 PSMain sunflower_fields.dxil)
add_hlsl_dxil(rainbow_travel.hlsl ps_6_0 PSMain rainbow_travel.dxil)
add_hlsl_dxil(mt_fuji_watercolor.hlsl ps_6_0 PSMain mt_fuji_watercolor.dxil)
add_hlsl_dxil(van_gogh_sunset.hlsl ps_6_0 PSMain van_gogh_sunset.dxil)
add_hlsl_dxil(rainbow_particle_hills.hlsl ps_6_0 PSMain rainbow_particle_hills.dxil)
add_hlsl_dxil(goldeen.hlsl ps_6_0 PSMain goldeen.dxil)
add_hlsl_dxil(disco_maze.hlsl ps_6_0 PSMain disco_maze.dxil)

add_custom_target(nacelle_test_shaders DEPENDS ${NACELLE_SHADER_OUTPUTS})

function(add_nacelle_test name src)
    add_executable(${name} ${src})
    add_dependencies(${name} nacelle_test_shaders)
    target_link_libraries(${name} PRIVATE nacelle)
    target_include_directories(${name} PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/third_party/DirectX-Headers/include"
        "${CMAKE_SOURCE_DIR}/third_party/DirectXMath/Inc"
        "${CMAKE_SOURCE_DIR}/third_party/thread-pool/include"
    )
    if(NOT WIN32)
        target_include_directories(${name} PRIVATE "${CMAKE_SOURCE_DIR}/src/common/posix_sal")
    endif()
    target_compile_definitions(${name} PRIVATE
        NACELLE_SHADER_BIN_DIR="${NACELLE_SHADER_BIN_DIR}"
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_nacelle_test(d3d12_api_test d3d12_api_test.cpp)
add_nacelle_test(compute_dispatch_test compute_dispatch_test.cpp)
add_nacelle_test(simd_wave_test simd_wave_test.cpp)
add_nacelle_test(rasterizer_pipeline_test rasterizer_pipeline_test.cpp)
add_nacelle_test(graphics_features_test graphics_features_test.cpp)
add_nacelle_test(pbr_render_test pbr_render_test.cpp)
add_nacelle_test(pso_cache_bcn_test pso_cache_bcn_test.cpp)
add_nacelle_test(interlocked_barrier_test interlocked_barrier_test.cpp)
add_nacelle_test(hlsl_intrinsics_matrix_test hlsl_intrinsics_matrix_test.cpp)
add_nacelle_test(texture_sampler_test texture_sampler_test.cpp)

# lit / FileCheck compiler IR tests
file(GLOB NACELLE_LIT_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/lit/*.test")
foreach(lit_test ${NACELLE_LIT_TESTS})
    get_filename_component(lit_name "${lit_test}" NAME_WE)
    add_test(
        NAME "lit_${lit_name}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/lit/run_lit_test.py"
                --nacelle-compile "$<TARGET_FILE:nacelle-compile>"
                --dxc "${DXC_EXECUTABLE}"
                --filecheck "${FILECHECK_EXECUTABLE}"
                --split-file "${SPLIT_FILE_EXECUTABLE}"
                "${lit_test}"
    )
endforeach()

# Quick benchmark smoke test
add_test(
    NAME nacelle_bench_smoke
    COMMAND $<TARGET_FILE:nacelle-bench> --quick --simd-width=4,8
)
