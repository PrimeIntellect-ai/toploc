from setuptools import setup
import os
import platform
from torch.utils.cpp_extension import BuildExtension, CppExtension

CSRC_DIR = os.path.join("toploc", "C", "csrc")

# Define mac-specific compiler and linker flags
extra_compile_args = ["-O3"]
if os.environ.get("DEBUG"):
    extra_compile_args.append("-DDEBUG")
extra_link_args: list[str] = []

# Add macOS specific flags
if platform.system() == "Darwin":
    # Enable support for both Intel and Apple Silicon
    extra_compile_args.extend(["-arch", "x86_64", "-arch", "arm64"])
    extra_link_args.extend(["-arch", "x86_64", "-arch", "arm64"])

    # Add minimum deployment target for macOS
    extra_compile_args.append("-mmacosx-version-min=10.13")
    extra_link_args.append("-mmacosx-version-min=10.13")

extensions = [
    CppExtension(
        name="toploc.C.csrc.ndd",
        sources=[os.path.join(CSRC_DIR, "ndd.cpp")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    CppExtension(
        name="toploc.C.csrc.utils",
        sources=[os.path.join(CSRC_DIR, "utils.cpp")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    CppExtension(
        name="toploc.C.csrc.poly",
        sources=[os.path.join(CSRC_DIR, "poly.cpp")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="toploc",
    ext_modules=extensions,
    packages=["toploc", "toploc.C.csrc"],
    package_data={
        "toploc.C.csrc": ["*.pyi"],  # Include .pyi files
    },
    cmdclass={"build_ext": BuildExtension},
)
