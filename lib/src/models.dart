import 'dart:typed_data';

/// Data types a tensor / feature can have, normalised across platforms.
enum TensorDataType {
  float16,
  float32,
  float64,
  int8,
  uint8,
  int16,
  int32,
  int64,
  bool,
  string,
  image,
  dictionary,
  sequence,
  unknown;

  /// Parses the platform-provided type name (case-insensitive).
  static TensorDataType fromName(String? name) {
    switch (name?.toLowerCase()) {
      case 'float16':
      case 'half':
        return TensorDataType.float16;
      case 'float32':
      case 'float':
        return TensorDataType.float32;
      case 'float64':
      case 'double':
        return TensorDataType.float64;
      case 'int8':
        return TensorDataType.int8;
      case 'uint8':
        return TensorDataType.uint8;
      case 'int16':
        return TensorDataType.int16;
      case 'int32':
      case 'int':
        return TensorDataType.int32;
      case 'int64':
      case 'long':
        return TensorDataType.int64;
      case 'bool':
      case 'boolean':
        return TensorDataType.bool;
      case 'string':
        return TensorDataType.string;
      case 'image':
        return TensorDataType.image;
      case 'dictionary':
        return TensorDataType.dictionary;
      case 'sequence':
        return TensorDataType.sequence;
      default:
        return TensorDataType.unknown;
    }
  }
}

/// Where a model is allowed to run.
///
/// | Value                  | Android (LiteRT)                          | iOS (Core ML)            |
/// |------------------------|-------------------------------------------|--------------------------|
/// | [all]                  | GPU delegate when supported, otherwise CPU | CPU + GPU + Neural Engine |
/// | [cpuOnly]              | CPU (XNNPACK, multi-threaded)             | CPU only                 |
/// | [cpuAndGpu]            | GPU delegate, falls back to CPU           | CPU + GPU                |
/// | [cpuAndNeuralEngine]   | NNAPI delegate (API 27+), falls back to CPU | CPU + Neural Engine    |
enum ComputeUnit {
  all,
  cpuOnly,
  cpuAndGpu,
  cpuAndNeuralEngine;

  /// Parses a value produced by [name]; returns null for unknown names.
  static ComputeUnit? tryParse(String? name) {
    for (final unit in ComputeUnit.values) {
      if (unit.name == name) return unit;
    }
    return null;
  }
}

/// Describes one model input or output.
class TensorInfo {
  /// Tensor / feature name used as the key in [NativeMLModel.run].
  final String name;

  /// Declared shape. A negative dimension (`-1`) means the size is dynamic.
  final List<int> shape;

  /// Normalised, lowercase data type name (`float32`, `uint8`, `image`, ...).
  final String dataType;

  /// Position of the tensor in the model (Android only).
  final int? index;

  /// Whether the input may be omitted (Core ML optional inputs).
  final bool isOptional;

  /// Quantization scale for quantized tensors, otherwise null.
  final double? quantizationScale;

  /// Quantization zero point for quantized tensors, otherwise null.
  final int? quantizationZeroPoint;

  /// Extra platform-specific details, for example `imageWidth`, `imageHeight`,
  /// `pixelFormat` (Core ML image inputs) or `keyType` (dictionary features).
  final Map<String, dynamic> extra;

  const TensorInfo({
    required this.name,
    required this.shape,
    required this.dataType,
    this.index,
    this.isOptional = false,
    this.quantizationScale,
    this.quantizationZeroPoint,
    this.extra = const {},
  });

  factory TensorInfo.fromMap(Map<dynamic, dynamic> map) {
    const known = {
      'name',
      'shape',
      'dataType',
      'dtype',
      'index',
      'isOptional',
      'quantizationScale',
      'quantizationZeroPoint',
    };
    final extra = <String, dynamic>{};
    for (final entry in map.entries) {
      final key = entry.key.toString();
      if (!known.contains(key)) extra[key] = entry.value;
    }
    return TensorInfo(
      name: map['name']?.toString() ?? '',
      shape: _toIntList(map['shape']),
      dataType: (map['dataType'] ?? map['dtype'])?.toString().toLowerCase() ?? 'unknown',
      index: (map['index'] as num?)?.toInt(),
      isOptional: map['isOptional'] == true,
      quantizationScale: (map['quantizationScale'] as num?)?.toDouble(),
      quantizationZeroPoint: (map['quantizationZeroPoint'] as num?)?.toInt(),
      extra: extra,
    );
  }

  /// The parsed [dataType].
  TensorDataType get type => TensorDataType.fromName(dataType);

  /// True when any dimension of [shape] is dynamic.
  bool get hasDynamicShape => shape.any((d) => d < 0);

  /// True when the tensor carries quantization parameters.
  bool get isQuantized => quantizationScale != null && quantizationScale != 0;

  /// Number of elements implied by [shape]. Dynamic dimensions count as 1,
  /// a scalar (empty shape) counts as 1.
  int get elementCount => shape.fold<int>(1, (acc, d) => acc * (d < 0 ? 1 : d));

  Map<String, dynamic> toMap() => {
        'name': name,
        'shape': shape,
        'dataType': dataType,
        if (index != null) 'index': index,
        'isOptional': isOptional,
        if (quantizationScale != null) 'quantizationScale': quantizationScale,
        if (quantizationZeroPoint != null) 'quantizationZeroPoint': quantizationZeroPoint,
        ...extra,
      };

  @override
  String toString() => 'Tensor(name: $name, shape: $shape, type: $dataType'
      '${isOptional ? ', optional' : ''}'
      '${isQuantized ? ', scale: $quantizationScale, zeroPoint: $quantizationZeroPoint' : ''})';
}

/// The inputs and outputs a loaded model expects and produces.
class ModelSignature {
  final List<TensorInfo> inputs;
  final List<TensorInfo> outputs;

  /// TensorFlow Lite SignatureDef keys (Android only, usually `serving_default`).
  final List<String> signatureKeys;

  /// SignatureDef name aliases (Android): `{key: {'inputs': {alias: tensorName}, 'outputs': {...}}}`.
  final Map<String, dynamic> signatures;

  /// Free-form metadata (runtime version, Core ML author/description/version,
  /// classifier labels, ...).
  final Map<String, dynamic> metadata;

  const ModelSignature({
    required this.inputs,
    required this.outputs,
    this.signatureKeys = const [],
    this.signatures = const {},
    this.metadata = const {},
  });

  factory ModelSignature.fromMap(Map<dynamic, dynamic> map) {
    List<TensorInfo> parse(Object? value) {
      if (value is! List) return const [];
      return value
          .whereType<Map>()
          .map((m) => TensorInfo.fromMap(m))
          .toList(growable: false);
    }

    return ModelSignature(
      inputs: parse(map['inputs']),
      outputs: parse(map['outputs']),
      signatureKeys: (map['signatureKeys'] as List?)?.map((e) => e.toString()).toList() ?? const [],
      signatures: _toStringKeyedMap(map['signatures']),
      metadata: _toStringKeyedMap(map['metadata']),
    );
  }

  /// Looks up an input by [name]; returns null when it does not exist.
  TensorInfo? input(String name) => _find(inputs, name);

  /// Looks up an output by [name]; returns null when it does not exist.
  TensorInfo? output(String name) => _find(outputs, name);

  static TensorInfo? _find(List<TensorInfo> list, String name) {
    for (final t in list) {
      if (t.name == name) return t;
    }
    return null;
  }

  Map<String, dynamic> toMap() => {
        'inputs': inputs.map((t) => t.toMap()).toList(),
        'outputs': outputs.map((t) => t.toMap()).toList(),
        'signatureKeys': signatureKeys,
        'signatures': signatures,
        'metadata': metadata,
      };

  @override
  String toString() => 'ModelSignature(inputs: $inputs, outputs: $outputs)';
}

/// Input data with an explicit shape, for models with dynamic dimensions.
///
/// ```dart
/// await model.run({
///   'input': TensorData(Float32List(2 * 128), shape: [2, 128]),
/// });
/// ```
class TensorData {
  /// The flattened values: a `List<num>`, `List<bool>`, `List<String>` or a
  /// typed list (`Float32List`, `Float64List`, `Int32List`, `Int64List`,
  /// `Uint8List`).
  final Object data;

  /// Shape to resize the input tensor to before running. Must contain only
  /// positive dimensions whose product equals the number of elements in [data].
  final List<int>? shape;

  const TensorData(this.data, {this.shape});

  Map<String, dynamic> toMap() => {
        'data': data,
        if (shape != null) 'shape': shape,
      };
}

/// Pixel layouts accepted by [ImageInput].
enum ImagePixelFormat {
  /// 4 bytes per pixel, R-G-B-A.
  rgba,

  /// 4 bytes per pixel, B-G-R-A (Core ML's native layout).
  bgra,

  /// 4 bytes per pixel, A-R-G-B.
  argb,

  /// 3 bytes per pixel, R-G-B.
  rgb,

  /// 1 byte per pixel.
  grayscale,

  /// PNG / JPEG / HEIC bytes; decoded and scaled natively.
  encoded,
}

/// Image data for Core ML image inputs (iOS) or `uint8` image tensors (Android).
///
/// On iOS the bytes are converted into a `CVPixelBuffer` matching the model's
/// image constraint. Encoded images ([ImagePixelFormat.encoded]) are decoded
/// and scaled to the size the model expects.
class ImageInput {
  final Uint8List bytes;
  final int? width;
  final int? height;
  final ImagePixelFormat format;

  const ImageInput(
    this.bytes, {
    this.width,
    this.height,
    this.format = ImagePixelFormat.rgba,
  });

  /// Convenience constructor for PNG/JPEG data.
  const ImageInput.encoded(this.bytes)
      : width = null,
        height = null,
        format = ImagePixelFormat.encoded;

  Map<String, dynamic> toMap() => {
        'data': bytes,
        if (width != null) 'width': width,
        if (height != null) 'height': height,
        'format': format.name,
        'kind': 'image',
      };
}

/// Information about the camera frame an [InferenceResult] was computed from.
class CameraFrameInfo {
  /// Size of the upright source frame (after rotation), in pixels.
  final int width;
  final int height;

  /// Rotation that was applied to make the frame upright.
  final int rotationDegrees;

  /// Capture timestamp on the device's monotonic clock.
  final Duration? timestamp;

  const CameraFrameInfo({
    required this.width,
    required this.height,
    this.rotationDegrees = 0,
    this.timestamp,
  });

  factory CameraFrameInfo.fromMap(Map<dynamic, dynamic> map) {
    final micros = map['timestampMicros'];
    return CameraFrameInfo(
      width: (map['width'] as num?)?.toInt() ?? 0,
      height: (map['height'] as num?)?.toInt() ?? 0,
      rotationDegrees: (map['rotationDegrees'] as num?)?.toInt() ?? 0,
      timestamp: micros is num ? Duration(microseconds: micros.round()) : null,
    );
  }

  @override
  String toString() => 'CameraFrameInfo(${width}x$height, rotation: $rotationDegrees°)';
}

/// The outcome of a single inference.
class InferenceResult {
  /// Output values keyed by output name.
  ///
  /// Numeric tensors arrive as typed lists (`Float32List`, `Int32List`,
  /// `Int64List`, `Uint8List`, `Float64List`) which implement `List<double>` /
  /// `List<int>`. Core ML can also return `String`, `int`, `double`,
  /// `Map<String, double>` (classifier probabilities), lists, or an image map.
  final Map<String, dynamic> output;

  /// Shape of every output as reported after running (may differ from the
  /// declared shape for models with dynamic outputs).
  final Map<String, List<int>> outputShapes;

  /// Wall-clock time spent in the native prediction call.
  final Duration inferenceTime;

  /// Time reported by the runtime itself, when available (Android).
  final Duration? nativeInferenceTime;

  /// Human-readable description of the compute unit that was used.
  final String acceleratorUsed;

  /// Sequence number of the frame when produced by a stream.
  final int? frameId;

  /// Frames dropped by the stream queue so far (streams only).
  final int droppedFrames;

  /// The camera frame this result was computed from (camera sessions only).
  final CameraFrameInfo? frame;

  const InferenceResult({
    required this.output,
    required this.inferenceTime,
    required this.acceleratorUsed,
    this.outputShapes = const {},
    this.nativeInferenceTime,
    this.frameId,
    this.droppedFrames = 0,
    this.frame,
  });

  factory InferenceResult.fromMap(Map<dynamic, dynamic> map) {
    final shapes = <String, List<int>>{};
    final rawShapes = map['outputShapes'];
    if (rawShapes is Map) {
      for (final entry in rawShapes.entries) {
        shapes[entry.key.toString()] = _toIntList(entry.value);
      }
    }
    return InferenceResult(
      output: _toStringKeyedMap(map['output']),
      outputShapes: shapes,
      inferenceTime: _microsToDuration(map['inferenceTime']) ?? Duration.zero,
      nativeInferenceTime: _microsToDuration(map['nativeInferenceTime']),
      acceleratorUsed: map['acceleratorUsed']?.toString() ?? 'unknown',
      frameId: (map['frameId'] as num?)?.toInt(),
      droppedFrames: (map['droppedFrames'] as num?)?.toInt() ?? 0,
      frame: map['frame'] is Map ? CameraFrameInfo.fromMap(map['frame'] as Map) : null,
    );
  }

  /// Returns the output [name] as a `List<double>`, converting integer or
  /// boolean values when necessary. Returns null if the output is missing or
  /// not numeric.
  List<double>? doubles(String name) {
    final value = output[name];
    if (value is List<double>) return value;
    if (value is List) {
      final out = <double>[];
      for (final v in value) {
        if (v is num) {
          out.add(v.toDouble());
        } else if (v is bool) {
          out.add(v ? 1 : 0);
        } else {
          return null;
        }
      }
      return out;
    }
    return null;
  }

  /// Returns the output [name] as a `List<int>`; returns null if missing or
  /// not numeric.
  List<int>? ints(String name) {
    final value = output[name];
    if (value is List<int>) return value;
    if (value is List) {
      final out = <int>[];
      for (final v in value) {
        if (v is num) {
          out.add(v.toInt());
        } else if (v is bool) {
          out.add(v ? 1 : 0);
        } else {
          return null;
        }
      }
      return out;
    }
    return null;
  }

  /// Index of the largest value in output [name] (handy for classifiers).
  int? argmax(String name) {
    final values = doubles(name);
    if (values == null || values.isEmpty) return null;
    var best = 0;
    for (var i = 1; i < values.length; i++) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }

  @override
  String toString() => 'InferenceResult(accelerator: $acceleratorUsed, '
      'time: ${inferenceTime.inMicroseconds}µs, outputs: ${output.keys.toList()})';
}

/// Result of queueing a frame on a stream.
class StreamPushResult {
  /// Sequence number assigned to the frame; matches [InferenceResult.frameId].
  final int frameId;

  /// Frames currently waiting to be processed.
  final int queueSize;

  /// Total frames dropped since the stream started.
  final int droppedFrames;

  const StreamPushResult({
    required this.frameId,
    required this.queueSize,
    required this.droppedFrames,
  });

  factory StreamPushResult.fromMap(Map<dynamic, dynamic>? map) => StreamPushResult(
        frameId: (map?['frameId'] as num?)?.toInt() ?? -1,
        queueSize: (map?['queueSize'] as num?)?.toInt() ?? 0,
        droppedFrames: (map?['droppedFrames'] as num?)?.toInt() ?? 0,
      );
}

/// Hardware / runtime information about the current device.
class DeviceCapabilities {
  /// `android` or `ios`.
  final String platform;
  final String? osVersion;
  final String? device;
  final int cpuCount;

  /// Runtime version, e.g. `LiteRT 1.4.2` or `Core ML (iOS 17.4)`.
  final String? runtimeVersion;

  /// GPU acceleration is available (LiteRT GPU delegate / Metal).
  final bool gpuAvailable;

  /// Android NNAPI is available (API 27+). Always false on iOS.
  final bool nnapiAvailable;

  /// Apple Neural Engine is believed to be available. Always false on Android.
  final bool neuralEngineAvailable;

  final bool isEmulator;

  /// Compute units that make sense on this device.
  final List<ComputeUnit> supportedComputeUnits;

  /// The raw map returned by the platform.
  final Map<String, dynamic> raw;

  const DeviceCapabilities({
    required this.platform,
    this.osVersion,
    this.device,
    this.cpuCount = 1,
    this.runtimeVersion,
    this.gpuAvailable = false,
    this.nnapiAvailable = false,
    this.neuralEngineAvailable = false,
    this.isEmulator = false,
    this.supportedComputeUnits = const [ComputeUnit.all, ComputeUnit.cpuOnly],
    this.raw = const {},
  });

  factory DeviceCapabilities.fromMap(Map<dynamic, dynamic> map) {
    final units = <ComputeUnit>[];
    final rawUnits = map['supportedComputeUnits'];
    if (rawUnits is List) {
      for (final u in rawUnits) {
        final parsed = ComputeUnit.tryParse(u?.toString());
        if (parsed != null) units.add(parsed);
      }
    }
    return DeviceCapabilities(
      platform: map['platform']?.toString() ?? 'unknown',
      osVersion: map['osVersion']?.toString(),
      device: map['device']?.toString(),
      cpuCount: (map['cpuCount'] as num?)?.toInt() ?? 1,
      runtimeVersion: map['runtimeVersion']?.toString(),
      gpuAvailable: map['gpuAvailable'] == true,
      nnapiAvailable: map['nnapiAvailable'] == true,
      neuralEngineAvailable: map['neuralEngineAvailable'] == true,
      isEmulator: map['isEmulator'] == true,
      supportedComputeUnits: units.isEmpty ? const [ComputeUnit.all, ComputeUnit.cpuOnly] : units,
      raw: _toStringKeyedMap(map),
    );
  }

  @override
  String toString() => 'DeviceCapabilities($platform $osVersion, $device, '
      'gpu: $gpuAvailable, nnapi: $nnapiAvailable, ane: $neuralEngineAvailable, '
      'runtime: $runtimeVersion)';
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

List<int> _toIntList(Object? value) {
  if (value is List) {
    return value.map((e) => e is num ? e.toInt() : int.tryParse(e.toString()) ?? 0).toList(growable: false);
  }
  return const [];
}

Map<String, dynamic> _toStringKeyedMap(Object? value) {
  if (value is Map) {
    return value.map((key, v) => MapEntry(key.toString(), v));
  }
  return <String, dynamic>{};
}

Duration? _microsToDuration(Object? value) {
  if (value is num) return Duration(microseconds: value.round());
  return null;
}
