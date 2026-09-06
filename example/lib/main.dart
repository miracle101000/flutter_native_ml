import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_native_ml/flutter_native_ml.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Native ML',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const DemoPage(),
    );
  }
}

class DemoPage extends StatefulWidget {
  const DemoPage({super.key});

  @override
  State<DemoPage> createState() => _DemoPageState();
}

class _DemoPageState extends State<DemoPage> {
  static const _defaultAsset = 'assets/models/model';

  final _pathController = TextEditingController();
  NativeMLModel? _model;
  ModelSignature? _signature;
  DeviceCapabilities? _capabilities;
  ComputeUnit _computeUnit = ComputeUnit.all;
  StreamSubscription<InferenceResult>? _streamSubscription;
  int _streamResults = 0;
  int _streamDropped = 0;
  String _status = 'Load a model to get started.';
  String _result = '';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _loadCapabilities();
  }

  @override
  void dispose() {
    _streamSubscription?.cancel();
    _model?.dispose();
    _pathController.dispose();
    super.dispose();
  }

  Future<void> _loadCapabilities() async {
    try {
      final capabilities = await FlutterNativeML.getDeviceCapabilities();
      if (mounted) setState(() => _capabilities = capabilities);
    } catch (e) {
      _setStatus('Could not read device capabilities: $e');
    }
  }

  void _setStatus(String status, {bool busy = false, String? result}) {
    if (!mounted) return;
    setState(() {
      _status = status;
      _busy = busy;
      if (result != null) _result = result;
    });
  }

  Future<void> _loadModel() async {
    _setStatus('Loading model...', busy: true);
    try {
      final path = _pathController.text.trim();
      final model = path.isNotEmpty
          ? await FlutterNativeML.loadModel(filePath: path, computeUnits: _computeUnit)
          : await FlutterNativeML.loadModel(
              assetPath: Platform.isIOS ? '$_defaultAsset.mlmodel' : '$_defaultAsset.tflite',
              computeUnits: _computeUnit,
            );
      _model = model;
      _signature = model.signature;
      _setStatus('Model loaded on ${model.acceleratorUsed}.', result: _describeSignature(_signature));
    } on NativeMLException catch (e) {
      _setStatus('Load failed [${e.code}]: ${e.message}');
    } catch (e) {
      _setStatus('Load failed: $e');
    }
  }

  Future<void> _getSignature() async {
    final model = _model;
    if (model == null) return;
    _setStatus('Reading signature...', busy: true);
    try {
      _signature = await model.getSignature(refresh: true);
      _setStatus('Signature ready.', result: _describeSignature(_signature));
    } catch (e) {
      _setStatus('Error reading signature: $e');
    }
  }

  String _describeSignature(ModelSignature? signature) {
    if (signature == null) return '';
    final sb = StringBuffer('Inputs:\n');
    for (final t in signature.inputs) {
      sb.writeln('  • $t');
    }
    sb.writeln('Outputs:');
    for (final t in signature.outputs) {
      sb.writeln('  • $t');
    }
    if (signature.signatureKeys.isNotEmpty) {
      sb.writeln('Signature keys: ${signature.signatureKeys}');
    }
    if (signature.metadata.isNotEmpty) {
      sb.writeln('Metadata: ${signature.metadata}');
    }
    return sb.toString();
  }

  /// Builds a dummy input (0.5 everywhere) matching the first model input.
  Map<String, Object> _sampleInput(ModelSignature signature) {
    final input = signature.inputs.first;
    final count = input.elementCount;
    switch (input.type) {
      case TensorDataType.uint8:
      case TensorDataType.int8:
        return {input.name: Uint8List(count)};
      case TensorDataType.int32:
      case TensorDataType.int64:
        return {input.name: Int32List(count)};
      case TensorDataType.string:
        return {input.name: List<String>.filled(count, 'hello')};
      case TensorDataType.image:
        final width = (input.extra['imageWidth'] as num?)?.toInt() ?? 224;
        final height = (input.extra['imageHeight'] as num?)?.toInt() ?? 224;
        return {
          input.name: ImageInput(
            Uint8List(width * height * 4),
            width: width,
            height: height,
            format: ImagePixelFormat.rgba,
          ),
        };
      default:
        return {input.name: Float32List.fromList(List<double>.filled(count, 0.5))};
    }
  }

  Future<void> _runInference() async {
    final model = _model;
    final signature = _signature;
    if (model == null || signature == null || signature.inputs.isEmpty) return;
    _setStatus('Running inference...', busy: true);
    try {
      final result = await model.run(_sampleInput(signature));
      final sb = StringBuffer()
        ..writeln('Accelerator: ${result.acceleratorUsed}')
        ..writeln('Time: ${result.inferenceTime.inMicroseconds} µs'
            '${result.nativeInferenceTime != null ? ' (native ${result.nativeInferenceTime!.inMicroseconds} µs)' : ''}')
        ..writeln('Outputs:');
      result.output.forEach((name, value) {
        final shape = result.outputShapes[name];
        final preview = value is List ? value.take(10).toList() : value;
        sb.writeln('  • $name${shape != null ? ' $shape' : ''}: $preview${value is List && value.length > 10 ? ' …' : ''}');
      });
      _setStatus('Inference complete.', result: sb.toString());
    } on NativeMLException catch (e) {
      _setStatus('Inference failed [${e.code}]: ${e.message}');
    } catch (e) {
      _setStatus('Inference failed: $e');
    }
  }

  Future<void> _toggleStream() async {
    final model = _model;
    final signature = _signature;
    if (model == null || signature == null) return;
    if (model.isStreaming) {
      await _streamSubscription?.cancel();
      _streamSubscription = null;
      await model.stopStream();
      _setStatus('Stream stopped after $_streamResults results ($_streamDropped dropped).');
      return;
    }
    _streamResults = 0;
    _streamDropped = 0;
    _streamSubscription = model.startStream(maxQueueSize: 2).listen(
      (result) {
        _streamResults++;
        _streamDropped = result.droppedFrames;
        _setStatus(
          'Streaming: frame ${result.frameId} in ${result.inferenceTime.inMicroseconds} µs '
          '($_streamResults results, $_streamDropped dropped)',
          result: result.output.keys.map((k) => '$k: ${(result.output[k] as List?)?.take(5).toList()}').join('\n'),
        );
      },
      onError: (Object e) => _setStatus('Stream error: $e'),
    );
    // Push a burst of frames faster than most models can process to show frame dropping.
    final input = _sampleInput(signature);
    for (var i = 0; i < 10; i++) {
      try {
        await model.pushStreamInput(input);
      } catch (e) {
        _setStatus('Could not push frame: $e');
        break;
      }
    }
    setState(() {});
  }

  Future<void> _disposeModel() async {
    final model = _model;
    if (model == null) return;
    await _streamSubscription?.cancel();
    _streamSubscription = null;
    await model.dispose();
    setState(() {
      _model = null;
      _signature = null;
      _result = '';
    });
    _setStatus('Model disposed.');
  }

  @override
  Widget build(BuildContext context) {
    final model = _model;
    final caps = _capabilities;
    return Scaffold(
      appBar: AppBar(title: const Text('Flutter Native ML')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (caps != null)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text(
                  '${caps.platform} ${caps.osVersion ?? ''} · ${caps.device ?? ''}\n'
                  'Runtime: ${caps.runtimeVersion ?? 'n/a'} · CPUs: ${caps.cpuCount}\n'
                  'GPU: ${caps.gpuAvailable} · NNAPI: ${caps.nnapiAvailable} · Neural Engine: ${caps.neuralEngineAvailable}',
                ),
              ),
            ),
          const SizedBox(height: 12),
          TextField(
            controller: _pathController,
            enabled: model == null && !_busy,
            decoration: const InputDecoration(
              labelText: 'Load from file path (optional)',
              hintText: 'Leave empty to use assets/models/model.tflite | .mlmodel',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<ComputeUnit>(
            initialValue: _computeUnit,
            decoration: const InputDecoration(labelText: 'Compute units', border: OutlineInputBorder()),
            items: [
              for (final unit in caps?.supportedComputeUnits ?? ComputeUnit.values)
                DropdownMenuItem(value: unit, child: Text(unit.name)),
            ],
            onChanged: model == null && !_busy ? (v) => setState(() => _computeUnit = v ?? ComputeUnit.all) : null,
          ),
          const SizedBox(height: 16),
          Text(_status, textAlign: TextAlign.center, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 12),
          if (_busy) const Center(child: CircularProgressIndicator()),
          const SizedBox(height: 12),
          FilledButton(
            onPressed: model == null && !_busy ? _loadModel : null,
            child: const Text('1. Load model'),
          ),
          const SizedBox(height: 8),
          FilledButton.tonal(
            onPressed: model != null && !_busy ? _getSignature : null,
            child: const Text('2. Refresh signature'),
          ),
          const SizedBox(height: 8),
          FilledButton.tonal(
            onPressed: model != null && _signature != null && !_busy ? _runInference : null,
            child: const Text('3. Run inference'),
          ),
          const SizedBox(height: 8),
          FilledButton.tonal(
            onPressed: model != null && _signature != null && !_busy ? _toggleStream : null,
            child: Text(model?.isStreaming == true ? '4. Stop stream' : '4. Stream 10 frames'),
          ),
          const SizedBox(height: 8),
          FilledButton.tonal(
            onPressed: model != null && _signature != null && !_busy
                ? () => Navigator.of(context).push(
                      MaterialPageRoute<void>(builder: (_) => CameraPage(model: model)),
                    )
                : null,
            child: const Text('5. Live camera (zero-copy)'),
          ),
          const SizedBox(height: 8),
          OutlinedButton(
            onPressed: model != null && !_busy ? _disposeModel : null,
            child: const Text('6. Dispose model'),
          ),
          const SizedBox(height: 20),
          if (_result.isNotEmpty)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: SelectableText(_result, style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
              ),
            ),
        ],
      ),
    );
  }
}

/// Streams camera frames into [model] natively and overlays the live results.
class CameraPage extends StatefulWidget {
  final NativeMLModel model;

  const CameraPage({super.key, required this.model});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  NativeCameraSession? _session;
  StreamSubscription<InferenceResult>? _subscription;
  InferenceResult? _latest;
  String? _error;
  CameraLens _lens = CameraLens.back;
  int _results = 0;
  DateTime? _windowStart;
  double _fps = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _start();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _subscription?.cancel();
    _session?.stop();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final session = _session;
    if (session == null || !session.isRunning) return;
    if (state == AppLifecycleState.resumed) {
      session.resume();
    } else if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      session.pause();
    }
  }

  Future<void> _start() async {
    try {
      if (!await FlutterNativeML.requestCameraPermission()) {
        setState(() => _error = 'Camera permission was not granted.');
        return;
      }
      final session = await widget.model.startCamera(
        lens: _lens,
        preprocessing: CameraPreprocessing.zeroToOne,
      );
      _subscription = session.results.listen(
        (result) {
          _results++;
          final now = DateTime.now();
          final start = _windowStart ??= now;
          final elapsed = now.difference(start).inMilliseconds;
          if (elapsed >= 1000) {
            _fps = _results * 1000 / elapsed;
            _results = 0;
            _windowStart = now;
          }
          if (mounted) setState(() => _latest = result);
        },
        onError: (Object e) {
          if (mounted) setState(() => _error = '$e');
        },
      );
      if (mounted) setState(() => _session = session);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  Future<void> _switchLens() async {
    await _subscription?.cancel();
    await _session?.stop();
    setState(() {
      _session = null;
      _latest = null;
      _error = null;
      _lens = _lens == CameraLens.back ? CameraLens.front : CameraLens.back;
    });
    await _start();
  }

  String _describe(InferenceResult result) {
    final sb = StringBuffer()
      ..writeln('${result.acceleratorUsed} · ${result.inferenceTime.inMilliseconds} ms · ${_fps.toStringAsFixed(1)} fps')
      ..writeln('frame ${result.frameId} (${result.frame?.width}x${result.frame?.height}), dropped ${result.droppedFrames}');
    for (final name in result.output.keys) {
      final values = result.doubles(name);
      if (values != null && values.isNotEmpty) {
        final best = result.argmax(name)!;
        sb.writeln('$name: argmax $best = ${values[best].toStringAsFixed(3)}');
      } else {
        sb.writeln('$name: ${result.output[name]}');
      }
    }
    return sb.toString();
  }

  @override
  Widget build(BuildContext context) {
    final session = _session;
    final latest = _latest;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live camera'),
        actions: [
          IconButton(icon: const Icon(Icons.cameraswitch), onPressed: session == null ? null : _switchLens),
        ],
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (session != null) NativeCameraPreview(session: session) else const ColoredBox(color: Colors.black),
          if (session == null && _error == null) const Center(child: CircularProgressIndicator()),
          Positioned(
            left: 12,
            right: 12,
            bottom: 12,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(8)),
              child: Text(
                _error ?? (latest == null ? 'Waiting for the first frame…' : _describe(latest)),
                style: const TextStyle(color: Colors.white, fontFamily: 'monospace', fontSize: 12),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
