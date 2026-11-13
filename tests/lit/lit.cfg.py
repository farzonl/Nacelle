import os
import lit.formats

config.name = "Nacelle"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = getattr(config, "nacelle_obj_root", config.test_source_root)

config.substitutions.append(("%nacelle_compile", getattr(config, "nacelle_compile", "nacelle-compile")))
config.substitutions.append(("%dxc", getattr(config, "dxc_exe", "dxc")))
config.substitutions.append(("%FileCheck", getattr(config, "filecheck_exe", "FileCheck")))
config.substitutions.append(("%split_file", getattr(config, "split_file_exe", "split-file")))
