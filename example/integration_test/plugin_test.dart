// End-to-end test of flutter_native_ml against a real device or emulator.
//
// Place a model at `assets/models/model.tflite` (Android) or
// `assets/models/model.mlmodel` (iOS) and run:
//
//   flutter test integration_test -d <device>
//
// Without a model asset, the parts that need one are skipped.
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_native_ml/flutter_native_ml.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

Future<bool> _assetExists(String path) async {
  try {
    await rootBundle.load(path);
    return true;
  } catch (_) {
    return false;
  }
}

/// Builds a plausible input for every required model input.
Map<String, Object> _sampleInput(ModelSignature signature) {
  final input = <String, Object>{};
  for (final tensor in signature.inputs) {
    if (tensor.isOptional) continue;
    final count = tensor.elementCount;
    switch (tensor.type) {
      case TensorDataType.uint8:
      case TensorDataType.int8:
        input[tensor.name] = Uint8List.fromList(List<int>.filled(count, 128));
      case TensorDataType.int16:
      case TensorDataType.int32:
      case TensorDataType.int64:
        input[tensor.name] = Int32List(count);
      case TensorDataType.bool:
        input[tensor.name] = List<bool>.filled(count, false);
      case TensorDataType.string:
        input[tensor.name] = tensor.shape.isEmpty ? 'hello' : List<String>.filled(count, 'hello');
      case TensorDataType.image:
        final width = (tensor.extra['imageWidth'] as num?)?.toInt() ?? 224;
        final height = (tensor.extra['imageHeight'] as num?)?.toInt() ?? 224;
        input[tensor.name] = ImageInput(
          Uint8List.fromList(List<int>.filled(width * height * 4, 127)),
          width: width,
          height: height,
          format: ImagePixelFormat.rgba,
        );
      case TensorDataType.dictionary:
        input[tensor.name] = tensor.extra['keyType'] == 'int64' ? {0: 1.0} : {'a': 1.0};
      case TensorDataType.sequence:
        input[tensor.name] = tensor.extra['elementType'] == 'int64' ? [1] : ['a'];
      case TensorDataType.float64:
        input[tensor.name] = tensor.shape.isEmpty ? 0.5 : Float64List.fromList(List<double>.filled(count, 0.5));
      default:
        input[tensor.name] = Float32List.fromList(List<double>.filled(count, 0.5));
    }
  }
  return input;
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  final extension = Platform.isIOS ? 'mlmodel' : 'tflite';
  final candidateAssets = ['assets/models/model.$extension', 'assets/models/model2.$extension'];

  testWidgets('platform version and device capabilities', (tester) async {
    final version = await FlutterNativeML.getPlatformVersion();
    expect(version, startsWith(Platform.isIOS ? 'iOS' : 'Android'));

    final capabilities = await FlutterNativeML.getDeviceCapabilities();
    debugPrint('capabilities: ${capabilities.raw}');
    expect(capabilities.platform, Platform.isIOS ? 'ios' : 'android');
    expect(capabilities.cpuCount, greaterThan(0));
    expect(capabilities.supportedComputeUnits, contains(ComputeUnit.all));
    expect(capabilities.supportedComputeUnits, contains(ComputeUnit.cpuOnly));
  });

  testWidgets('loading a missing asset reports MODEL_NOT_FOUND', (tester) async {
    await expectLater(
      FlutterNativeML.loadModel(assetPath: 'assets/models/does_not_exist.bin'),
      throwsA(isA<NativeMLException>().having((e) => e.code, 'code', 'MODEL_NOT_FOUND')),
    );
  });

  testWidgets('load, inspect, run, stream and dispose real models', (tester) async {
    var tested = 0;
    for (final assetPath in candidateAssets) {
      if (!await _assetExists(assetPath)) continue;
      tested++;
      debugPrint('=== testing $assetPath ===');
      await _exerciseModel(assetPath);
    }
    if (tested == 0) {
      debugPrint('SKIPPED: no model asset found (looked for $candidateAssets)');
    }
  });

  testWidgets('camera input feeds frames to the model natively', (tester) async {
    final assetPath = candidateAssets.first;
    if (!await _assetExists(assetPath)) {
      debugPrint('SKIPPED: no model asset at $assetPath');
      return;
    }
    final capabilities = await FlutterNativeML.getDeviceCapabilities();
    final model = await FlutterNativeML.loadModel(assetPath: assetPath);
    final status = await FlutterNativeML.checkCameraPermission();
    debugPrint('camera permission: $status, cameraAvailable: ${capabilities.raw['cameraAvailable']}');

    if (Platform.isIOS && capabilities.isEmulator) {
      // The iOS simulator has no camera: the plugin must report it cleanly.
      await expectLater(
        model.startCamera(),
        throwsA(isA<NativeMLException>().having((e) => e.code, 'code', anyOf('CAMERA_UNAVAILABLE', 'PERMISSION_DENIED'))),
      );
      await model.dispose();
      return;
    }
    if (status != CameraPermissionStatus.granted) {
      debugPrint('SKIPPED: camera permission not granted (grant it with adb before running)');
      await model.dispose();
      return;
    }

    final session = await model.startCamera(resolution: CameraResolution.low, maxFps: 10);
    debugPrint('camera session: $session');
    expect(session.inputWidth, greaterThan(0));
    expect(session.inputHeight, greaterThan(0));
    expect(session.textureId, isNotNull);
    expect(session.previewWidth, greaterThan(0));
    expect(model.cameraSessions, contains(session));

    final received = <InferenceResult>[];
    final errors = <Object>[];
    final subscription = session.results.listen(received.add, onError: errors.add);
    final deadline = DateTime.now().add(const Duration(seconds: 30));
    while (received.length < 3 && DateTime.now().isBefore(deadline)) {
      await Future<void>.delayed(const Duration(milliseconds: 100));
    }
    debugPrint('camera: ${received.length} results, first: ${received.isEmpty ? null : received.first}, frame: ${received.isEmpty ? null : received.first.frame}');
    expect(errors, isEmpty);
    expect(received.length, greaterThanOrEqualTo(3));
    expect(received.first.frame, isNotNull);
    expect(received.first.frame!.width, greaterThan(0));
    expect(received.first.output, isNotEmpty);
    final frameIds = received.map((r) => r.frameId!).toList();
    expect(frameIds, orderedEquals(frameIds.toList()..sort()));

    await session.pause();
    final countAtPause = received.length;
    await Future<void>.delayed(const Duration(milliseconds: 700));
    expect(received.length, countAtPause, reason: 'no results while paused');
    await session.resume();
    final resumeDeadline = DateTime.now().add(const Duration(seconds: 15));
    while (received.length == countAtPause && DateTime.now().isBefore(resumeDeadline)) {
      await Future<void>.delayed(const Duration(milliseconds: 100));
    }
    expect(received.length, greaterThan(countAtPause), reason: 'results resume after resume()');

    await session.stop();
    expect(session.isRunning, isFalse);
    expect(model.cameraSessions, isEmpty);
    await subscription.cancel();
    await session.stop(); // idempotent
    await model.dispose();
  });

  testWidgets('disposeAll succeeds', (tester) async {
    await FlutterNativeML.disposeAll();
  });
}

Future<void> _exerciseModel(String assetPath) async {
  {
    final model = await FlutterNativeML.loadModel(assetPath: assetPath);
    debugPrint('loaded ${model.id} on ${model.acceleratorUsed}');
    expect(model.acceleratorUsed, isNot('unknown'));

    final signature = model.signature ?? await model.getSignature();
    debugPrint('signature: ${signature.toMap()}');
    expect(signature.inputs, isNotEmpty);
    expect(signature.outputs, isNotEmpty);
    for (final tensor in [...signature.inputs, ...signature.outputs]) {
      expect(tensor.name, isNotEmpty);
      expect(tensor.type, isNot(TensorDataType.unknown), reason: 'unknown type for $tensor');
    }

    // A refreshed signature must match the cached one.
    final refreshed = await model.getSignature(refresh: true);
    expect(refreshed.inputs.map((t) => t.name), signature.inputs.map((t) => t.name));

    // Single inference.
    final input = _sampleInput(signature);
    final result = await model.run(input);
    debugPrint('result: $result (${result.inferenceTime.inMicroseconds} µs) shapes=${result.outputShapes}');
    expect(result.inferenceTime, greaterThan(Duration.zero));
    expect(result.acceleratorUsed, model.acceleratorUsed);
    for (final output in signature.outputs) {
      expect(result.output, contains(output.name));
      final value = result.output[output.name];
      if (value is List && output.type != TensorDataType.string) {
        expect(value, isNotEmpty);
        expect(result.doubles(output.name), isNotNull);
      }
    }

    // A second run returns the same output for the same input.
    final again = await model.run(input);
    expect(again.output.keys, unorderedEquals(result.output.keys));

    // Error handling.
    await expectLater(
      model.run({'this-input-does-not-exist': Float32List(4)}),
      throwsA(isA<NativeMLException>().having((e) => e.code, 'code', anyOf('INPUT_MISMATCH', 'MISSING_INPUT'))),
    );
    final first = signature.inputs.first;
    if (first.type != TensorDataType.image &&
        first.type != TensorDataType.string &&
        first.type != TensorDataType.dictionary &&
        first.type != TensorDataType.sequence &&
        first.shape.isNotEmpty) {
      await expectLater(
        model.run({...input, first.name: Float32List(first.elementCount + 1)}),
        throwsA(isA<NativeMLException>().having((e) => e.code, 'code', 'SHAPE_MISMATCH')),
      );
    }

    // Streaming: push more frames than the queue holds and make sure results arrive in order.
    final received = <InferenceResult>[];
    final errors = <Object>[];
    final subscription = model.startStream(maxQueueSize: 2).listen(received.add, onError: errors.add);
    final pushes = <StreamPushResult>[];
    for (var i = 0; i < 6; i++) {
      pushes.add(await model.pushStreamInput(input));
    }
    expect(pushes.map((p) => p.frameId), [0, 1, 2, 3, 4, 5]);
    final deadline = DateTime.now().add(const Duration(seconds: 30));
    while (received.length + (pushes.last.droppedFrames) < 6 && DateTime.now().isBefore(deadline)) {
      await Future<void>.delayed(const Duration(milliseconds: 50));
      if (received.isNotEmpty && received.last.frameId == 5) break;
    }
    debugPrint('stream: ${received.length} results, frames ${received.map((r) => r.frameId).toList()}, '
        'dropped ${received.isEmpty ? 0 : received.last.droppedFrames}');
    expect(errors, isEmpty);
    expect(received, isNotEmpty);
    expect(received.last.frameId, 5, reason: 'the newest frame must always be processed');
    final ids = received.map((r) => r.frameId!).toList();
    expect(ids, orderedEquals(ids.toList()..sort()));
    await model.stopStream();
    expect(model.isStreaming, isFalse);
    await subscription.cancel();

    // Loading with an explicit compute unit and disposing twice.
    final cpuModel = await FlutterNativeML.loadModel(assetPath: assetPath, computeUnits: ComputeUnit.cpuOnly);
    expect(cpuModel.acceleratorUsed, startsWith('CPU'));
    final cpuResult = await cpuModel.run(input);
    expect(cpuResult.output.keys, unorderedEquals(result.output.keys));
    await cpuModel.dispose();
    await cpuModel.dispose();

    await model.dispose();
    expect(model.isDisposed, isTrue);
    expect(() => model.run(input), throwsStateError);
  }
}
