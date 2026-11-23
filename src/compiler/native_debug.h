#pragma once

#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include <memory>
#include <string>

namespace llvm {
class JITEventListener;
class Module;
namespace orc {
class LLJIT;
class LLJITBuilder;
}
}

namespace nacelle::native_debug {
bool enabled();
bool freeze();
void prepareModule(llvm::Module& module);
bool configureBuilder(llvm::orc::LLJITBuilder& builder,
                      std::shared_ptr<llvm::JITEventListener>& listener, std::string& error);
void installObjectAdmission(llvm::orc::LLJIT& jit);
std::string objectIdentity(uint64_t address);
bool verifyEntry(uint64_t address, std::string& error);
}
#endif
