# LeelaBench — lc0 0.33 on iPhone / iPad

A small iOS app that embeds [lc0](https://github.com/LeelaChessZero/lc0) (`release/0.33`) and
benchmarks networks on the device's backends:

| backend | runs on |
|---|---|
| `metal` | GPU, via MPSGraph |
| `blas` | CPU, via Apple Accelerate |
| `eigen` | CPU |
| `onnx-coreml` | Core ML: CPU+GPU, CPU+Neural Engine, or All |
| `onnx-cpu` | ONNX Runtime CPU |
| `random` | no NN — measures search overhead only |

Modes: **Backend** (`lc0 backendbench`, nps per batch size, charted), **Search**
(`lc0 benchmark`), **Quick** (`lc0 bench`) and **Net info** (`lc0 describenet`). Results are kept,
can be compared on one chart, and exported as CSV.

## How it works

iOS apps can't launch executables, so lc0 is linked into the app and its `main()` is called
in-process. [`patches/lc0-ios.patch`](patches/lc0-ios.patch) makes that possible:

* `-Dios_library=true` builds a static library with `lc0_main()` instead of the `lc0` executable,
  and returns instead of calling `abort()` on errors.
* Metal: uses `MTLCreateSystemDefaultDevice()` (`MTLCopyAllDevices()` is macOS-only) and builds
  a fresh MPSGraph each run so switching nets works.

[`scripts/build_lc0.sh`](scripts/build_lc0.sh) cross-compiles lc0 with meson for `arm64` iOS,
pulls the static ONNX Runtime iOS framework, and writes `app/Vendor/`. The Xcode project is
generated from [`app/project.yml`](app/project.yml) with XcodeGen.

## Building (no Mac needed)

1. Create a GitHub repo and push this folder to it. A **public** repo gets free macOS runner minutes.
2. The **Build iOS app** workflow runs on every push (or run it by hand from the Actions tab, where
   you can choose the lc0 ref and ONNX Runtime version).
3. Download the `LeelaBench-ipa` artifact from the run and unzip it to get `LeelaBench.ipa`.
   Pushing a tag like `v1.0` also attaches the `.ipa` to a GitHub Release.

On a Mac you can build locally instead: `./scripts/build_lc0.sh && cd app && xcodegen generate`,
then open `LeelaBench.xcodeproj`.

## Installing for free (from Windows)

The `.ipa` is unsigned. Sign and install it with your own Apple ID:

* **[Sideloadly](https://sideloadly.io)** (easiest on Windows): install iTunes and iCloud from
  Apple's website (not the Microsoft Store), plug in the device, drop in the `.ipa`, sign in
  with your Apple ID.
* On the device, turn on **Settings → Privacy & Security → Developer Mode** (iOS 16+) and trust
  your Apple ID under **Settings → General → VPN & Device Management**.
* Free Apple IDs: the app expires after **7 days**, so re-sign it with Sideloadly (your results
  and nets are kept). You can have at most 3 sideloaded apps at a time.
  [AltStore](https://altstore.io)/[SideStore](https://sidestore.io) can refresh it automatically.
* With a paid developer account ($99/yr) the signature lasts a year.
* If your device supports [TrollStore](https://github.com/opa334/TrollStore) (older iOS
  versions only), the unsigned `.ipa` installs permanently.

## Using it

1. **Networks** tab: download a net by URL (see [lczero.org best nets](https://lczero.org/play/networks/bestnets/)),
   import a `.pb.gz` from Files, or copy it into the LeelaBench folder with the Files app.
2. **Benchmark** tab: pick net, backend and mode, then Run. Keep the app in the foreground; iOS
   doesn't allow GPU work in the background, and a run can't be cancelled.
3. **Results** tab: tap **Select**, pick runs, then **Compare**.

Tips:
* Big nets (BT4 class) need a lot of memory; on phones with little RAM, iOS may kill the app.
  Small/medium nets are safer on iPhone.
* Thermal state is shown on the Benchmark tab. Phones throttle quickly, so let the device cool
  between runs if you want comparable numbers.
* The first `onnx-coreml` run for a net is slow because Core ML compiles the model.
