// CLI that prepares models for flutter_native_ml.
//
// Usage (from your app's root):
//   dart run flutter_native_ml:ml_builder -s path/to/model.h5 -o assets/models/
//
// * `.mlmodel` / `.mlpackage`  -> compiled `.mlmodelc` (macOS only; needs Xcode tools).
//   Note: you can also ship the `.mlmodel` file directly; the plugin compiles
//   it on device and caches the result.
// * `.h5` / `.keras` / SavedModel directory -> `.tflite` (needs Python 3 + TensorFlow).
import 'dart:io';

import 'package:args/args.dart';
import 'package:path/path.dart' as p;

Future<void> main(List<String> arguments) async {
  final parser = ArgParser()
    ..addOption(
      'source',
      abbr: 's',
      help: 'Source model: model.mlmodel, model.mlpackage, model.h5, model.keras or a SavedModel directory.',
    )
    ..addOption(
      'output-dir',
      abbr: 'o',
      help: 'Directory that receives the converted model.',
      defaultsTo: 'models_out',
    )
    ..addFlag(
      'quantize-fp16',
      help: 'Apply float16 quantization (TensorFlow Lite only).',
      negatable: false,
    )
    ..addOption(
      'python',
      help: 'Python interpreter to use for TensorFlow conversion.',
      defaultsTo: 'python3',
    )
    ..addFlag(
      'install-tensorflow',
      help: 'Run `pip install tensorflow` when TensorFlow is not importable.',
      negatable: false,
    )
    ..addFlag('help', abbr: 'h', help: 'Show this help message.', negatable: false);

  final ArgResults args;
  try {
    args = parser.parse(arguments);
  } on FormatException catch (e) {
    stderr.writeln(e.message);
    stderr.writeln(parser.usage);
    exitCode = 64;
    return;
  }

  if (args['help'] as bool || !args.wasParsed('source')) {
    stdout.writeln('ml_builder: prepares models for flutter_native_ml.\n');
    stdout.writeln(parser.usage);
    exitCode = args['help'] as bool ? 0 : 64;
    return;
  }

  final source = args['source'] as String;
  final outputDir = args['output-dir'] as String;
  final quantizeFp16 = args['quantize-fp16'] as bool;
  final python = args['python'] as String;
  final installTensorFlow = args['install-tensorflow'] as bool;

  final sourceType = FileSystemEntity.typeSync(source);
  if (sourceType == FileSystemEntityType.notFound) {
    stderr.writeln('Source not found: $source');
    exitCode = 66;
    return;
  }

  await Directory(outputDir).create(recursive: true);
  final extension = p.extension(source).toLowerCase();

  stdout.writeln('Converting $source ...');
  if (extension == '.mlmodel' || extension == '.mlpackage') {
    exitCode = await convertCoreML(source, outputDir);
  } else if (extension == '.h5' ||
      extension == '.keras' ||
      sourceType == FileSystemEntityType.directory) {
    exitCode = await convertTensorFlow(
      source,
      outputDir,
      quantizeFp16: quantizeFp16,
      python: python,
      installTensorFlow: installTensorFlow,
    );
  } else if (extension == '.tflite' || extension == '.mlmodelc') {
    stdout.writeln('$source is already in a native format; nothing to do.');
  } else {
    stderr.writeln('Unsupported source: $source. Provide a .mlmodel, .mlpackage, .h5, .keras or a SavedModel directory.');
    exitCode = 65;
  }
}

Future<int> convertCoreML(String source, String outputDir) async {
  if (!Platform.isMacOS) {
    stderr.writeln('Core ML compilation requires macOS with Xcode command line tools.');
    stderr.writeln('Alternatively ship the .mlmodel as an asset; flutter_native_ml compiles it on device.');
    return 69;
  }
  final baseName = p.basenameWithoutExtension(source);
  final outputPath = p.join(outputDir, '$baseName.mlmodelc');
  stdout.writeln('Compiling Core ML model to $outputPath ...');
  final result = await Process.run('xcrun', ['coremlcompiler', 'compile', source, outputDir]);
  stdout.write(result.stdout);
  stderr.write(result.stderr);
  if (result.exitCode != 0) {
    stderr.writeln('Core ML compilation failed (exit code ${result.exitCode}). Is Xcode installed?');
    return result.exitCode;
  }
  stdout.writeln('Done: $outputPath');
  stdout.writeln('Note: .mlmodelc is a directory. To bundle it as a Flutter asset you must list the directory');
  stdout.writeln('and every sub-directory under `flutter: assets:`. Shipping the single-file .mlmodel is simpler.');
  return 0;
}

Future<int> convertTensorFlow(
  String source,
  String outputDir, {
  required bool quantizeFp16,
  required String python,
  required bool installTensorFlow,
}) async {
  final pythonPath = await _which(python);
  if (pythonPath == null) {
    stderr.writeln('$python was not found on PATH. Install Python 3 to convert TensorFlow models.');
    return 69;
  }

  final hasTensorFlow = (await Process.run(python, ['-c', 'import tensorflow'])).exitCode == 0;
  if (!hasTensorFlow) {
    if (!installTensorFlow) {
      stderr.writeln('TensorFlow is not importable from $python.');
      stderr.writeln('Install it (for example `$python -m pip install tensorflow`) or pass --install-tensorflow.');
      return 69;
    }
    stdout.writeln('Installing TensorFlow with pip (this can take a while) ...');
    final install = await Process.run(python, ['-m', 'pip', 'install', '-q', 'tensorflow']);
    stdout.write(install.stdout);
    stderr.write(install.stderr);
    if (install.exitCode != 0) {
      stderr.writeln('pip install failed (exit code ${install.exitCode}).');
      return install.exitCode;
    }
  }

  final baseName = p.basenameWithoutExtension(source);
  final outputPath = p.join(outputDir, '$baseName.tflite');
  final tempDir = await Directory.systemTemp.createTemp('flutter_native_ml_');
  final script = File(p.join(tempDir.path, 'convert.py'))..writeAsStringSync(_converterScript);
  try {
    stdout.writeln('Running TensorFlow Lite converter ...');
    final result = await Process.run(python, [
      script.path,
      source,
      outputPath,
      quantizeFp16 ? 'true' : 'false',
    ]);
    stdout.write(result.stdout);
    stderr.write(result.stderr);
    if (result.exitCode != 0) {
      stderr.writeln('TensorFlow Lite conversion failed (exit code ${result.exitCode}).');
      return result.exitCode;
    }
    stdout.writeln('Done: $outputPath');
    return 0;
  } finally {
    await tempDir.delete(recursive: true);
  }
}

Future<String?> _which(String command) async {
  final tool = Platform.isWindows ? 'where' : 'which';
  final result = await Process.run(tool, [command]);
  if (result.exitCode != 0) return null;
  final path = (result.stdout as String).trim();
  return path.isEmpty ? null : path;
}

const _converterScript = r'''
import os
import sys

import tensorflow as tf

source_path, output_path = sys.argv[1], sys.argv[2]
quantize = sys.argv[3] == "true"

if os.path.isdir(source_path):
    print(f"Loading SavedModel: {source_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(source_path)
else:
    print(f"Loading Keras model: {source_path}")
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(source_path))

if quantize:
    print("Applying float16 quantization")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

with open(output_path, "wb") as f:
    f.write(converter.convert())
print(f"Wrote {output_path}")
''';
