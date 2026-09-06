import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_native_ml/flutter_native_ml.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const channel = MethodChannel('flutter_native_ml');
  final messenger = TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  final calls = <MethodCall>[];

  Map<String, dynamic> signatureMap() => {
        'inputs': [
          {
            'index': 0,
            'name': 'input',
            'shape': [1, 4],
            'dataType': 'float32',
            'quantizationScale': null,
            'quantizationZeroPoint': null,
          },
          {
            'index': 1,
            'name': 'mask',
            'shape': [-1, 4],
            'dataType': 'uint8',
            'quantizationScale': 0.5,
            'quantizationZeroPoint': 3,
          },
        ],
        'outputs': [
          {'index': 0, 'name': 'probs', 'shape': [1, 3], 'dataType': 'float32'},
        ],
        'signatureKeys': ['serving_default'],
        'signatures': {
          'serving_default': {
            'inputs': {'x': 'input'},
            'outputs': {'y': 'probs'},
          },
        },
        'metadata': {'runtime': 'LiteRT 1.4.2'},
      };

  void mockHandler(Future<Object?> Function(MethodCall call) handler) {
    messenger.setMockMethodCallHandler(channel, (call) async {
      calls.add(call);
      return handler(call);
    });
  }

  setUp(calls.clear);

  tearDown(() {
    messenger.setMockMethodCallHandler(channel, null);
  });

  group('loadModel', () {
    test('sends arguments and parses the native response', () async {
      mockHandler((call) async {
        if (call.method == 'loadModel') {
          return {
            'modelId': 'abc',
            'acceleratorUsed': 'GPU',
            'signature': signatureMap(),
          };
        }
        return null;
      });

      final model = await FlutterNativeML.loadModel(
        assetPath: 'assets/models/model.tflite',
        computeUnits: ComputeUnit.cpuAndGpu,
        numThreads: 2,
        allowFp16: true,
      );

      expect(calls.single.method, 'loadModel');
      expect(calls.single.arguments, {
        'assetPath': 'assets/models/model.tflite',
        'computeUnits': 'cpuAndGpu',
        'numThreads': 2,
        'allowFp16': true,
      });
      expect(model.id, 'abc');
      expect(model.acceleratorUsed, 'GPU');
      expect(model.signature, isNotNull);
      expect(model.signature!.inputs.map((t) => t.name), ['input', 'mask']);
      expect(model.signature!.inputs[1].hasDynamicShape, isTrue);
      expect(model.signature!.inputs[1].isQuantized, isTrue);
      expect(model.signature!.inputs[1].type, TensorDataType.uint8);
      expect(model.signature!.outputs.single.elementCount, 3);
      expect(model.signature!.signatureKeys, ['serving_default']);
      expect(model.signature!.metadata['runtime'], 'LiteRT 1.4.2');
    });

    test('accepts a bare id from older native implementations', () async {
      mockHandler((call) async => 'legacy-id');
      final model = await FlutterNativeML.loadModel(filePath: '/tmp/model.tflite');
      expect(model.id, 'legacy-id');
      expect(model.signature, isNull);
      expect(calls.single.arguments['filePath'], '/tmp/model.tflite');
      expect(calls.single.arguments.containsKey('assetPath'), isFalse);
    });

    test('rejects missing or conflicting paths', () async {
      expect(() => FlutterNativeML.loadModel(), throwsArgumentError);
      expect(
        () => FlutterNativeML.loadModel(assetPath: 'a', filePath: 'b'),
        throwsArgumentError,
      );
      expect(
        () => FlutterNativeML.loadModel(assetPath: 'a', numThreads: 0),
        throwsArgumentError,
      );
    });

    test('wraps platform errors in NativeMLException', () async {
      mockHandler((call) async {
        throw PlatformException(code: 'MODEL_NOT_FOUND', message: 'nope', details: 'x');
      });
      try {
        await FlutterNativeML.loadModel(assetPath: 'missing.tflite');
        fail('expected an exception');
      } on NativeMLException catch (e) {
        expect(e.code, 'MODEL_NOT_FOUND');
        expect(e.message, 'nope');
        expect(e.details, 'x');
        expect(e.toString(), contains('MODEL_NOT_FOUND'));
      }
    });

    test('reports a missing plugin as NativeMLException', () async {
      // No handler registered -> MissingPluginException.
      expect(
        () => FlutterNativeML.loadModel(assetPath: 'a.tflite'),
        throwsA(isA<NativeMLException>().having((e) => e.code, 'code', 'MISSING_PLUGIN')),
      );
    });
  });

  group('NativeMLModel', () {
    test('getSignature is cached until refreshed', () async {
      mockHandler((call) async => signatureMap());
      final model = NativeMLModel.fromId('m1', channel);

      final first = await model.getSignature();
      final second = await model.getSignature();
      expect(identical(first, second), isTrue);
      expect(calls.length, 1);
      expect(calls.single.arguments, {'modelId': 'm1'});

      await model.getSignature(refresh: true);
      expect(calls.length, 2);
    });

    test('run encodes inputs and parses typed outputs', () async {
      mockHandler((call) async {
        expect(call.method, 'run');
        return {
          'output': {
            'probs': Float32List.fromList([0.1, 0.7, 0.2]),
            'label': 'cat',
            'count': 3,
          },
          'outputShapes': {'probs': [1, 3]},
          'inferenceTime': 1234.6,
          'nativeInferenceTime': 1000.0,
          'acceleratorUsed': 'CPU (XNNPACK, 4 threads)',
        };
      });
      final model = NativeMLModel.fromId('m1', channel);
      final result = await model.run(
        {
          'input': Float32List.fromList([1, 2, 3, 4]),
          'mask': const TensorData([1, 1, 1, 1, 0, 0, 0, 0], shape: [2, 4]),
          'image': ImageInput(Uint8List(16), width: 2, height: 2, format: ImagePixelFormat.bgra),
        },
        signatureKey: 'serving_default',
      );

      final args = calls.single.arguments as Map;
      expect(args['modelId'], 'm1');
      expect(args['signatureKey'], 'serving_default');
      final input = args['input'] as Map;
      expect(input['input'], isA<Float32List>());
      expect(input['mask'], {
        'data': [1, 1, 1, 1, 0, 0, 0, 0],
        'shape': [2, 4],
      });
      expect((input['image'] as Map)['format'], 'bgra');
      expect((input['image'] as Map)['width'], 2);

      expect(result.inferenceTime, const Duration(microseconds: 1235));
      expect(result.nativeInferenceTime, const Duration(milliseconds: 1));
      expect(result.acceleratorUsed, 'CPU (XNNPACK, 4 threads)');
      expect(result.outputShapes['probs'], [1, 3]);
      expect(result.output['probs'], isA<Float32List>());
      expect(result.doubles('probs'), hasLength(3));
      expect(result.doubles('probs')![1], closeTo(0.7, 1e-6));
      expect(result.argmax('probs'), 1);
      expect(result.doubles('label'), isNull);
      expect(result.output['label'], 'cat');
      expect(result.output['count'], 3);
    });

    test('run rejects empty and null inputs', () async {
      mockHandler((call) async => null);
      final model = NativeMLModel.fromId('m1', channel);
      expect(() => model.run({}), throwsArgumentError);
      expect(() => model.run({'x': null}), throwsArgumentError);
    });

    test('dispose is idempotent and blocks later use', () async {
      mockHandler((call) async => null);
      final model = NativeMLModel.fromId('m1', channel);

      await model.dispose();
      await model.dispose();
      expect(calls.where((c) => c.method == 'dispose').length, 1);
      expect(model.isDisposed, isTrue);
      expect(() => model.run({'x': [1.0]}), throwsStateError);
      expect(() => model.getSignature(), throwsStateError);
      expect(() => model.startStream(), throwsStateError);
    });

    test('dispose tolerates a model that is already gone natively', () async {
      mockHandler((call) async {
        throw PlatformException(code: 'MODEL_NOT_FOUND');
      });
      final model = NativeMLModel.fromId('m1', channel);
      await model.dispose();
      expect(model.isDisposed, isTrue);
    });
  });

  group('streaming', () {
    const streamChannel = EventChannel('flutter_native_ml_stream/m1');
    late StreamController<Object?> nativeEvents;

    setUp(() {
      nativeEvents = StreamController<Object?>.broadcast();
      messenger.setMockStreamHandler(
        streamChannel,
        MockStreamHandler.inline(
          onListen: (arguments, events) {
            expect(arguments, 'm1');
            nativeEvents.stream.listen(
              events.success,
              onError: (Object e) => events.error(code: 'INFERENCE_FAILED', message: '$e'),
              onDone: events.endOfStream,
            );
          },
        ),
      );
    });

    tearDown(() async {
      messenger.setMockStreamHandler(streamChannel, null);
      await nativeEvents.close();
    });

    test('delivers results, pushes frames and stops', () async {
      mockHandler((call) async {
        switch (call.method) {
          case 'startStream':
            return {'channel': 'flutter_native_ml_stream/m1', 'maxQueueSize': 3};
          case 'streamInput':
            return {'frameId': 7, 'queueSize': 1, 'droppedFrames': 2};
          default:
            return null;
        }
      });
      final model = NativeMLModel.fromId('m1', channel);
      final received = <InferenceResult>[];
      final errors = <Object>[];
      final stream = model.startStream(maxQueueSize: 3);
      expect(model.isStreaming, isTrue);
      expect(identical(stream, model.startStream()), isTrue);
      final subscription = stream.listen(received.add, onError: errors.add);

      final push = await model.pushStreamInput({'input': [1.0, 2.0, 3.0, 4.0]});
      expect(push.frameId, 7);
      expect(push.queueSize, 1);
      expect(push.droppedFrames, 2);
      expect(calls.map((c) => c.method), ['startStream', 'streamInput']);
      expect(calls.first.arguments, {'modelId': 'm1', 'maxQueueSize': 3});

      nativeEvents.add({
        'output': {'probs': Float32List.fromList([0.2, 0.8])},
        'inferenceTime': 500.0,
        'acceleratorUsed': 'GPU',
        'frameId': 7,
        'droppedFrames': 2,
      });
      nativeEvents.add('garbage');
      await pumpEventQueue();

      expect(received, hasLength(1));
      expect(received.single.frameId, 7);
      expect(received.single.droppedFrames, 2);
      expect(received.single.doubles('probs'), [closeTo(0.2, 1e-6), closeTo(0.8, 1e-6)]);
      expect(errors.single, isA<FormatException>());

      await model.stopStream();
      expect(model.isStreaming, isFalse);
      expect(calls.last.method, 'stopStream');
      expect(calls.last.arguments, {'modelId': 'm1'});
      await subscription.cancel();
      expect(() => model.pushStreamInput({'input': [1.0]}), throwsStateError);
    });

    test('surfaces native stream errors as NativeMLException', () async {
      mockHandler((call) async => null);
      final model = NativeMLModel.fromId('m1', channel);
      final errors = <Object>[];
      model.startStream().listen((_) {}, onError: errors.add);
      await model.pushStreamInput({'input': [1.0]});
      nativeEvents.addError('boom');
      await pumpEventQueue();
      expect(errors.single, isA<NativeMLException>().having((e) => e.code, 'code', 'INFERENCE_FAILED'));
      await model.stopStream();
    });

    test('a failing startStream reports the error and resets', () async {
      mockHandler((call) async {
        if (call.method == 'startStream') {
          throw PlatformException(code: 'MODEL_NOT_FOUND');
        }
        return null;
      });
      final model = NativeMLModel.fromId('m1', channel);
      final errors = <Object>[];
      model.startStream().listen((_) {}, onError: errors.add);
      await pumpEventQueue();
      expect(errors.single, isA<NativeMLException>().having((e) => e.code, 'code', 'MODEL_NOT_FOUND'));
      expect(model.isStreaming, isFalse);
    });

    test('pushStreamInput before startStream throws', () async {
      mockHandler((call) async => null);
      final model = NativeMLModel.fromId('m1', channel);
      expect(() => model.pushStreamInput({'input': [1.0]}), throwsStateError);
    });

    test('dispose tears the stream down', () async {
      mockHandler((call) async => null);
      final model = NativeMLModel.fromId('m1', channel);
      final done = Completer<void>();
      model.startStream().listen((_) {}, onDone: done.complete);
      await pumpEventQueue();
      await model.dispose();
      await done.future;
      expect(model.isStreaming, isFalse);
      expect(calls.map((c) => c.method), ['startStream', 'dispose']);
    });
  });

  group('static helpers', () {
    test('getPlatformVersion and getDeviceCapabilities', () async {
      mockHandler((call) async {
        switch (call.method) {
          case 'getPlatformVersion':
            return 'Android 15';
          case 'getDeviceCapabilities':
            return {
              'platform': 'android',
              'osVersion': '15',
              'device': 'Google Pixel',
              'cpuCount': 8,
              'runtimeVersion': 'LiteRT 1.4.2',
              'gpuAvailable': true,
              'nnapiAvailable': true,
              'neuralEngineAvailable': false,
              'isEmulator': false,
              'supportedComputeUnits': ['all', 'cpuOnly', 'cpuAndGpu', 'cpuAndNeuralEngine', 'bogus'],
            };
          default:
            return null;
        }
      });
      expect(await FlutterNativeML.getPlatformVersion(), 'Android 15');
      final caps = await FlutterNativeML.getDeviceCapabilities();
      expect(caps.platform, 'android');
      expect(caps.cpuCount, 8);
      expect(caps.gpuAvailable, isTrue);
      expect(caps.nnapiAvailable, isTrue);
      expect(caps.neuralEngineAvailable, isFalse);
      expect(caps.supportedComputeUnits, ComputeUnit.values);
      expect(caps.raw['device'], 'Google Pixel');
    });

    test('disposeAll forwards to the platform', () async {
      mockHandler((call) async => null);
      await FlutterNativeML.disposeAll();
      expect(calls.single.method, 'disposeAll');
    });
  });

  group('models', () {
    test('TensorDataType.fromName normalises platform names', () {
      expect(TensorDataType.fromName('FLOAT32'), TensorDataType.float32);
      expect(TensorDataType.fromName('double'), TensorDataType.float64);
      expect(TensorDataType.fromName('int'), TensorDataType.int32);
      expect(TensorDataType.fromName('half'), TensorDataType.float16);
      expect(TensorDataType.fromName(null), TensorDataType.unknown);
      expect(TensorDataType.fromName('whatever'), TensorDataType.unknown);
    });

    test('TensorInfo keeps platform extras and legacy dtype key', () {
      final info = TensorInfo.fromMap({
        'name': 'image',
        'shape': [224, 224, 4],
        'dtype': 'IMAGE',
        'isOptional': true,
        'imageWidth': 224,
        'pixelFormat': 'bgra',
      });
      expect(info.type, TensorDataType.image);
      expect(info.isOptional, isTrue);
      expect(info.extra, {'imageWidth': 224, 'pixelFormat': 'bgra'});
      expect(info.elementCount, 224 * 224 * 4);
      expect(info.toMap()['imageWidth'], 224);
      expect(const TensorInfo(name: 's', shape: [], dataType: 'string').elementCount, 1);
    });

    test('InferenceResult tolerates missing fields', () {
      final result = InferenceResult.fromMap({});
      expect(result.output, isEmpty);
      expect(result.inferenceTime, Duration.zero);
      expect(result.acceleratorUsed, 'unknown');
      expect(result.frameId, isNull);
      expect(result.argmax('x'), isNull);
    });

    test('ModelSignature lookups', () {
      final signature = ModelSignature.fromMap(signatureMap());
      expect(signature.input('mask')?.dataType, 'uint8');
      expect(signature.output('probs')?.shape, [1, 3]);
      expect(signature.input('nope'), isNull);
      expect(signature.toMap()['signatureKeys'], ['serving_default']);
    });
  });
}
