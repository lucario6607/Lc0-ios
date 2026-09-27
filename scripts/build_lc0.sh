#!/usr/bin/env bash
# Cross-compiles lc0 for iOS (arm64) as static libraries the app links against.
#
# Runs on macOS with Xcode. Produces:
#   app/Vendor/lib/liblc0_all.a      lc0 + abseil + everything meson built
#   app/Vendor/lib/libonnxruntime.a  ONNX Runtime (only when ONNX is enabled)
#   app/Vendor/lc0.xcconfig          linker flags matching what was built
#
# Environment knobs:
#   LC0_REF      lc0 branch/tag to build           (default: release/0.33)
#   ORT_VERSION  ONNX Runtime iOS version, or ""   (default: 1.30.0)
#                to build without onnx-coreml/onnx-cpu
#   IOS_MIN      iOS deployment target             (default: 16.0)
set -euo pipefail

LC0_REF="${LC0_REF:-release/0.33}"
ORT_VERSION="${ORT_VERSION-1.30.0}"
IOS_MIN="${IOS_MIN:-16.0}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$ROOT/build"
VENDOR="$ROOT/app/Vendor"
mkdir -p "$WORK" "$VENDOR/lib"

# --- lc0 source -------------------------------------------------------------
# (The CI cache may pre-create build/lc0/subprojects/packagecache, so test for .git.)
if [ ! -d "$WORK/lc0/.git" ]; then
  git init -q "$WORK/lc0"
  git -C "$WORK/lc0" fetch -q --depth 1 https://github.com/LeelaChessZero/lc0.git "$LC0_REF"
  git -C "$WORK/lc0" checkout -q FETCH_HEAD
  git -C "$WORK/lc0" apply "$ROOT/patches/lc0-ios.patch"
fi
LC0_REV="$(git -C "$WORK/lc0" rev-parse --short HEAD)"
echo "lc0 $LC0_REF @ $LC0_REV"

# --- ONNX Runtime (static iOS framework from the onnxruntime-c pod) ---------
ORT_DIR="$WORK/onnxruntime-$ORT_VERSION"
if [ -n "$ORT_VERSION" ] && [ ! -f "$ORT_DIR/lib/libonnxruntime.a" ]; then
  rm -rf "$ORT_DIR" && mkdir -p "$ORT_DIR/lib"
  curl -fsSL -o "$WORK/ort.zip" \
    "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-$ORT_VERSION.zip"
  unzip -q "$WORK/ort.zip" -d "$ORT_DIR/pod"
  cp -R "$ORT_DIR/pod/Headers" "$ORT_DIR/include"
  lipo -thin arm64 \
    "$ORT_DIR/pod/onnxruntime.xcframework/ios-arm64/onnxruntime.framework/onnxruntime" \
    -output "$ORT_DIR/lib/libonnxruntime.a"
fi

# --- meson cross file -------------------------------------------------------
SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
CC="$(xcrun --sdk iphoneos -f clang)"
CXX="$(xcrun --sdk iphoneos -f clang++)"
AR="$(xcrun --sdk iphoneos -f ar)"
STRIP="$(xcrun --sdk iphoneos -f strip)"
FLAGS="'-arch', 'arm64', '-isysroot', '$SDK', '-miphoneos-version-min=$IOS_MIN'"

cat > "$WORK/ios-arm64.ini" <<EOF
[host_machine]
system = 'darwin'
subsystem = 'ios'
kernel = 'xnu'
cpu_family = 'aarch64'
cpu = 'arm64'
endian = 'little'

[properties]
needs_exe_wrapper = true

[binaries]
c = '$CC'
cpp = '$CXX'
objc = '$CC'
objcpp = '$CXX'
ar = '$AR'
strip = '$STRIP'

[built-in options]
c_args = [$FLAGS]
cpp_args = [$FLAGS]
objc_args = [$FLAGS]
objcpp_args = [$FLAGS]
c_link_args = [$FLAGS]
cpp_link_args = [$FLAGS]
objc_link_args = [$FLAGS]
objcpp_link_args = [$FLAGS]
EOF

# --- configure + build ------------------------------------------------------
build_lc0() {  # $1 = true/false: include ONNX backends
  local with_onnx="$1"
  rm -rf "$WORK/meson"
  meson setup "$WORK/meson" "$WORK/lc0" \
    --cross-file "$WORK/ios-arm64.ini" \
    --buildtype=release \
    -Ddefault_library=static \
    -Db_lto=false \
    -Db_ndebug=true \
    -Dios_library=true \
    -Dnative_arch=false \
    -Dneon=false \
    -Dgtest=false \
    -Dispc=false \
    -Dopenblas=false \
    -Dopencl=false \
    -Dplain_cuda=false \
    -Dcudnn=false \
    -Dnvcc=false \
    -Dmetal=enabled \
    -Daccelerate=true \
    -Donnx="$with_onnx" \
    -Donnx_include="$ORT_DIR/include" \
    -Donnx_libdir="$ORT_DIR/lib" \
  && ninja -C "$WORK/meson"
}

ONNX_BUILT=false
if [ -n "$ORT_VERSION" ] && build_lc0 true; then
  ONNX_BUILT=true
else
  if [ -n "$ORT_VERSION" ]; then
    echo "::warning::lc0 build with ONNX Runtime failed; retrying without onnx backends"
  fi
  build_lc0 false
fi

# Show which backends made it in, for the CI log.
grep -E "Accelerate|appleframeworks|onnxruntime|Run-time dependency" \
  "$WORK/meson/meson-logs/meson-log.txt" | grep -iE "found" | sort -u || true

# --- bundle -----------------------------------------------------------------
# lc0 is a static library, but its dependencies (abseil, maybe zlib) are
# separate archives. Merge everything into one so Xcode needs a single -l.
find "$WORK/meson" -name '*.a' -print > "$WORK/archives.txt"
cat "$WORK/archives.txt"
xcrun libtool -static -no_warning_for_no_symbols -o "$VENDOR/lib/liblc0_all.a" \
  $(cat "$WORK/archives.txt")

# -force_load: backends and search algorithms register themselves from static
# initializers that nothing references, so a plain -l would drop them all.
LDFLAGS="-Wl,-force_load,\$(PROJECT_DIR)/Vendor/lib/liblc0_all.a -lz -lc++ -framework Foundation -framework Metal"
LDFLAGS="$LDFLAGS -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph"
LDFLAGS="$LDFLAGS -framework Accelerate"
if [ "$ONNX_BUILT" = true ]; then
  cp "$ORT_DIR/lib/libonnxruntime.a" "$VENDOR/lib/"
  LDFLAGS="$LDFLAGS -lonnxruntime -framework CoreML -framework Network"
else
  rm -f "$VENDOR/lib/libonnxruntime.a"
fi

cat > "$VENDOR/lc0.xcconfig" <<EOF
// Generated by scripts/build_lc0.sh -- do not edit.
LIBRARY_SEARCH_PATHS = \$(inherited) \$(PROJECT_DIR)/Vendor/lib
OTHER_LDFLAGS = \$(inherited) $LDFLAGS
LC0_VERSION_INFO = $LC0_REF ($LC0_REV)
LC0_ONNX = $ONNX_BUILT
EOF

echo "::notice::Built lc0 $LC0_REF ($LC0_REV), onnx=$ONNX_BUILT, liblc0_all.a $(stat -f%z "$VENDOR/lib/liblc0_all.a") bytes from $(wc -l < "$WORK/archives.txt") archives"
ls -lh "$VENDOR/lib"
