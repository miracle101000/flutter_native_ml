## 1.2.2

### Fixed
* **Android: AGP 9 built-in Kotlin support.** With `android.builtInKotlin=true`
  (the default since Android Gradle Plugin 9.0) the plugin previously tried to
  apply the Kotlin Gradle plugin on top of AGP's own Kotlin support and the
  build failed with `Cannot add extension with name 'kotlin'`. The plugin now
  detects built-in Kotlin (and the AGP 9 new DSL) and only applies the Kotlin
  Gradle plugin itself on older toolchains that need it. Enabling built-in
  Kotlin in an app requires Flutter 3.47 or newer with `android.newDsl=false`
  (Flutter's own Gradle tooling does not support the AGP 9 new DSL yet); apps
  that keep `android.builtInKotlin=false` are unaffected.
* Android unit tests: the test dependency is now `kotlin-test-junit5`
  explicitly, because AGP's built-in Kotlin does not auto-select the JUnit 5
  variant of the version-less `kotlin-test` shorthand.

## 1.2.1

* README: added a table of contents and an author section. No code changes.

## 1.2.0

### Added
* **Zero-copy camera input.** `model.startCamera()` opens the device camera
  natively (CameraX on Android, AVFoundation on iOS), resizes every frame into
  the model's input on the model's own thread and streams only the results to
  Dart. Includes a live preview texture (`NativeCameraPreview`), lens and
  resolution selection, `cover` / `fill` / `contain` resizing, mean / std
  normalisation presets, frame-rate throttling, pause / resume and automatic
  stale-frame dropping.
* `FlutterNativeML.checkCameraPermission()` / `requestCameraPermission()`.
* `InferenceResult.frame` with the source frame size, rotation and timestamp.
* `DeviceCapabilities.raw['cameraAvailable']`.

### Changed
* Android `minSdk` is now 23 (required by CameraX 1.5); the plugin declares the
  `CAMERA` permission, which apps that never use the camera can strip with
  `tools:node="remove"`.
* iOS apps using the camera must declare `NSCameraUsageDescription`; the plugin
  reports `MISSING_USAGE_DESCRIPTION` instead of crashing when it is absent.

## 1.1.0

### Fixed
* **Android did not compile.** The Kotlin plugin imported `com.google.ai.edge.litert.*`
  classes that no Maven artifact provides. The plugin now depends on LiteRT
  (`com.google.ai.edge.litert:litert` / `litert-gpu`) and uses its
  `org.tensorflow.lite` Interpreter API.
* Android returned the tensor type under `dtype` while Dart expected `dataType`,
  which crashed `getSignature()`.
* Android ran model loading and inference on the UI thread.
* Inference with the GPU delegate now always happens on the thread that created
  the delegate (a LiteRT requirement); every model has its own worker thread.
* iOS mutated shared state from a concurrent queue and returned some results
  off the main thread.
* iOS ignored the model's declared input shape and always built 1-D arrays.
* iOS/Android accepted only `List<double>` inputs; integer lists, typed lists and
  mixed `num` lists now work.
* `startStream` / `stopStream` were not implemented on Android and only emitted
  timestamps on iOS.
* `dart run flutter_native_ml:ml_builder` did not work (the CLI lived in a
  separate package). It now ships as `bin/ml_builder.dart`.
* Manifest `package` attribute removed (deprecated with AGP 8).

### Added
* `ComputeUnit.cpuAndGpu` and `ComputeUnit.cpuAndNeuralEngine` (NNAPI on Android),
  with automatic CPU fallback. The compute unit actually used is reported in
  `NativeMLModel.acceleratorUsed` and `InferenceResult.acceleratorUsed`.
* `loadModel(filePath: ...)` for models stored on the device, `numThreads` and
  `allowFp16` options.
* iOS: `.mlmodel` / `.mlpackage` assets are compiled on device and cached, so
  a single-file asset is enough (compiled `.mlmodelc` bundles are directories
  that Flutter does not bundle as one asset).
* iOS: image inputs/outputs (`ImageInput`, raw or PNG/JPEG), string, int64,
  double, dictionary and sequence features; float16 arrays.
* Android: all LiteRT tensor types (float32, int8/uint8/int16/int32/int64,
  bool, string), quantization parameters in the signature, SignatureDef
  aliases, dynamic shapes via `TensorData(data, shape: [...])`, native
  inference timing.
* Streaming API: `model.startStream()`, `model.pushStreamInput()`,
  `model.stopStream()` with a bounded native queue that drops stale frames.
* `FlutterNativeML.getDeviceCapabilities()`, `getPlatformVersion()`,
  `disposeAll()`.
* `NativeMLException` with stable error codes instead of raw `PlatformException`s.
* Typed-list outputs (`Float32List`, `Int32List`, ...) plus
  `InferenceResult.doubles()`, `ints()`, `argmax()` and `outputShapes`.
* Swift Package Manager support alongside CocoaPods; privacy manifest bundled.
* Kotlin DSL build script compatible with Flutter's built-in Kotlin support.

### Changed
* Minimum iOS version is 13.0; Android plugin targets Java 17 (as Flutter requires).
* `ModelSignature`/`TensorInfo` gained `type`, `elementCount`, `hasDynamicShape`,
  `isOptional`, quantization fields and `metadata`.
* The static `FlutterNativeML.startStream`/`stopStream` helpers are deprecated in
  favour of the `NativeMLModel` methods.

## 1.0.1

* Fixed bugs on android.

## 1.0.0

* Initial release.
