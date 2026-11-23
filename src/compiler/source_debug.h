#pragma once
#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include <nacelle/source_debug.h>
#include <string>
#include <string_view>
#include <vector>
namespace llvm { class Module; }
namespace nacelle { struct PixelBatchArgs; }
namespace nacelle::source_debug {
struct Site {
    NacelleSourceDebugSite location;
    std::vector<NacelleSourceDebugLocal> locals;
    uint64_t compilation = 0, entryVersion = 0;
    uint32_t tier = 0, mapping = NACELLE_SOURCE_MAPPING_EXACT;
    std::vector<uint32_t> availability;
    std::vector<NacelleOptimizedSourceDebugOrigin> origins;
    NacelleNativeDebugCode code;
};
NacelleOptimizedSourceDebugOrigin primaryOrigin(const NacelleSourceDebugSite& site);
bool enabled();
bool optimized();
bool freeze();
bool addSource(const std::string& id, const std::string& text, std::string& error);
uint32_t addSite(Site site, std::string& error);
bool readSite(uint32_t id, Site& site);
bool bindCompilation(uint64_t compilation, uint64_t address, uint64_t entryVersion, std::string& error);
void reportCompilationFailure(const std::string& error);
void publishShader(const std::string& identity);
bool instrument(llvm::Module& module, const std::string& identity, uint32_t stage,
    const uint32_t* threadsPerGroup, std::string& error);
bool sameProgram(const llvm::Module& executable, const llvm::Module& debug, std::string& error);
bool prepareOptimizedModule(llvm::Module& module, std::string& error);
bool lowerOptimizedObservations(llvm::Module& module, uint32_t tier, std::string& error);
uint64_t optimizedCompilation(const llvm::Module& module);
class Work {
public:
    Work();
    ~Work();
    Work(const Work&) = delete;
    Work& operator=(const Work&) = delete;
private:
    bool admitted = false;
    bool tracked = false;

};
class Operation {
public:
    explicit Operation(std::string_view shader);
    ~Operation();
    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;
private:
    bool admitted = false;
};
}
extern "C" void nacelle_source_debug_observe(uint32_t site, uint32_t width,
    uint32_t active, uint32_t real,     const uint32_t* xyz, const uint64_t* values, const uint64_t* invocations = nullptr);
extern "C" bool nacelle_source_debug_read_pixel_context(const nacelle::PixelBatchArgs* args,
    uint64_t batch, uint32_t width, uint32_t* xyz, uint32_t* masks, uint64_t* invocations) noexcept;
#endif
