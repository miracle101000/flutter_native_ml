class TensorInfo {
  final String name;
  final List<int> shape;
  final String dataType;

  TensorInfo({required this.name, required this.shape, required this.dataType});

  factory TensorInfo.fromMap(Map<dynamic, dynamic> map) {
    return TensorInfo(
      name: map['name'],
      shape: List<int>.from(map['shape']),
      dataType: map['dataType'],
    );
  }

  @override
  String toString() {
    return 'Tensor(name: $name, shape: $shape, type: $dataType)';
  }
}

class ModelSignature {
  final List<TensorInfo> inputs;
  final List<TensorInfo> outputs;

  ModelSignature({required this.inputs, required this.outputs});

  factory ModelSignature.fromMap(Map<dynamic, dynamic> map) {
    return ModelSignature(
      inputs: (map['inputs'] as List).map((i) => TensorInfo.fromMap(i)).toList(),
      outputs: (map['outputs'] as List).map((o) => TensorInfo.fromMap(o)).toList(),
    );
  }
}

enum ComputeUnit {
  all,
  cpuOnly,
}

class InferenceResult {
  final Map<String, dynamic> output;
  final Duration inferenceTime;
  final String acceleratorUsed;

  InferenceResult({
    required this.output,
    required this.inferenceTime,
    required this.acceleratorUsed,
  });

  factory InferenceResult.fromMap(Map<dynamic, dynamic> map) {
    return InferenceResult(
      output: Map<String, dynamic>.from(map['output']),
      inferenceTime: Duration(microseconds: (map['inferenceTime'] as double).round()),
      acceleratorUsed: map['acceleratorUsed'],
    );
  }
}