import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_native_ml/flutter_native_ml.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});
  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  NativeMLModel? _model;
  ModelSignature? _signature;
  StreamSubscription<InferenceResult>? _streamSub;
  String _status = 'App Started. Please load a model.';
  String _resultText = '';
  bool _isLoading = false;

  @override
  void dispose() {
    _streamSub?.cancel();
    _model?.dispose();
    super.dispose();
  }

  void _updateStatus(String status, {bool loading = false, String result = ''}) {
    setState(() {
      _status = status;
      _isLoading = loading;
      if (result.isNotEmpty) _resultText = result;
    });
  }

  Future<void> _loadModel() async {
    _updateStatus('Loading model...', loading: true);
    try {
      final assetPath = Platform.isIOS
          ? 'assets/models/sentiment.mlmodelc'
          : 'assets/models/sentiment.tflite';

      _model = await FlutterNativeML.loadModel(assetPath: assetPath);
      _updateStatus('Model loaded! Retrieve signature next.');
    } catch (e) {
      _updateStatus('Error loading model: $e');
    }
  }

  Future<void> _getSignature() async {
    if (_model == null) return;
    _updateStatus('Getting model signature...', loading: true);
    try {
      _signature = await _model!.getSignature();
      final sb = StringBuffer();
      sb.write('Inputs:\n');
      for (var io in _signature!.inputs) {
        sb.write('- ${io.name}: shape=${io.shape}, type=${io.dataType}\n');
      }
      sb.write('\nOutputs:\n');
      for (var io in _signature!.outputs) {
        sb.write('- ${io.name}: shape=${io.shape}, type=${io.dataType}\n');
      }
      _updateStatus('Signature ready! You can now run inference.', result: sb.toString());
    } catch (e) {
      _updateStatus('Error getting signature: $e');
    }
  }

  Future<void> _runInference() async {
    if (_model == null || _signature == null) return;
    _updateStatus('Running inference...', loading: true);
    try {
      final inputInfo = _signature!.inputs.first;
      final name = inputInfo.name;
      final size = inputInfo.shape.reduce((a, b) => a * b);
      final inputData = List<double>.filled(size, 0.5);

      final result = await _model!.run({name: inputData});

      final sb = StringBuffer()
        ..write('Accelerator: ${result.acceleratorUsed}\n')
        ..write('Time: ${result.inferenceTime.inMilliseconds} ms\n\nOutputs:\n');
      result.output.forEach((k, v) {
        sb.write('- $k: ${v is List ? v.take(10).toList() : v}\n');
      });

      _updateStatus('Inference complete!', result: sb.toString());
    } catch (e) {
      _updateStatus('Error running inference: $e');
    }
  }

  Future<void> _disposeModel() async {
    if (_model == null) return;
    await _model!.dispose();
    _model = null;
    _signature = null;
    _resultText = '';
    _updateStatus('Model disposed. Load a new one.');
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Native ML v1.0 Example')),
        body: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(_status, textAlign: TextAlign.center, style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 12),
                if (_isLoading) const Center(child: CircularProgressIndicator()),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: _model == null && !_isLoading ? _loadModel : null,
                  child: const Text('1. Load Model'),
                ),
                ElevatedButton(
                  onPressed: _model != null && _signature == null && !_isLoading ? _getSignature : null,
                  child: const Text('2. Get Signature'),
                ),
                ElevatedButton(
                  onPressed: _model != null && _signature != null && !_isLoading ? _runInference : null,
                  child: const Text('3. Run Inference'),
                ),
                ElevatedButton(
                  onPressed: _model != null && !_isLoading ? _disposeModel : null,
                  child: const Text('4. Dispose Model'),
                ),
                const SizedBox(height: 20),
                if (_resultText.isNotEmpty)
                  Card(
                    elevation: 2,
                    child: Padding(
                      padding: const EdgeInsets.all(12),
                      child: SelectableText(_resultText),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
