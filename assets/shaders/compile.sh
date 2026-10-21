#!/usr/bin/env bash
# Regenerate the test SPIR-V from Slang source.
#
# The .spv files are checked in so that `cargo test` is hermetic: no toolchain is
# needed to run the suite, and a shader compiler bug cannot turn into a backend
# bug. This script is how they are rebuilt.
#
# Use the same Slang the C++ tree vendored (2026.11). CI downloads that release
# and points `SLANGC` at it; a local C++ checkout has it at
# `UHE/vendor/bin/linux/slang/bin/slangc`. One version, so there is no second
# compiler in the project to keep patched.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
slangc="${SLANGC:-../UNHUMAN/UHE/vendor/bin/linux/slang/bin/slangc}"

if [[ ! -x "$slangc" ]]; then
  echo "slangc not found at $slangc" >&2
  echo "set SLANGC=/path/to/slangc" >&2
  exit 1
fi

# One SPIR-V module per entry point. Vulkan has no linked shader stage, so a
# vertex/fragment pair is two modules, and the backend loads them as a pair.
#
# sm_6_0 is the floor the engine targets; nothing in the test needs more, and
# compiling higher would hide a missing feature rather than report it.
# `source` is an argument rather than hardcoded, because there is more than one
# shader now and a compiler that can only build one of them is a compiler that
# makes the second shader look like a backend bug.
compile() {
  local entry="$1" stage="$2" out="$3" source="$4"
  "$slangc" \
    -target spirv \
    -profile sm_6_0 \
    -emit-spirv-directly \
    -entry "$entry" \
    -stage "$stage" \
    "$here/$source" \
    -o "$here/$out"
  echo "  $out"
}

echo "compiling with $slangc"
compile vs_main vertex Triangle.vert.spv Triangle.slang
compile fs_main fragment Triangle.frag.spv Triangle.slang

# The sampling shader. Two bindings -- a sampled image and a sampler -- because
# wgpu has no combined descriptor, so `CombinedImageSampler` would compile here
# and be unbuildable on the second backend.
compile vs_main vertex Textured.vert.spv Textured.slang
compile fs_main fragment Textured.frag.spv Textured.slang

# The compute shader. One stage module, no vertex/fragment pair.
compile cs_main compute StoreIndex.spv StoreIndex.slang

# A zero-byte or truncated module fails at pipeline creation with a message that
# points at the backend rather than the shader, so check the header here.
for spv in "$here"/*.spv; do
  magic=$(od -A n -t x4 -N 4 "$spv" | tr -d ' ')
  if [[ "$magic" != "07230203" ]]; then
    echo "$(basename "$spv"): bad SPIR-V magic 0x$magic, expected 0x07230203" >&2
    exit 1
  fi
done
echo "ok"