import 'dart:io';
import 'package:args/args.dart';
import 'package:process_run/shell.dart';
import 'package:path/path.dart' as p;

final shell = Shell();

void main(List<String> arguments) async {
  final parser = ArgParser()
    ..addOption('source', abbr: 's', help: 'Source model file (e.g., model.mlmodel, model.h5, or a SavedModel directory).', mandatory: true)
    ..addOption('output-dir', abbr: 'o', help: 'Output directory for converted models.', defaultsTo: 'models_out')
    ..addFlag('quantize-fp16', help: 'Apply float16 quantization (TFLite only).', negatable: false)
    ..addFlag('help', abbr: 'h', help: 'Show this help message.', negatable: false);

  final ArgResults argResults;
  try {
    argResults = parser.parse(arguments);
  } on FormatException catch (e) {
    print(e.message);
    print(parser.usage);
    exit(1);
  }

  if (argResults['help'] as bool) {
    print('flutter_ml_builder: A tool to prepare models for native execution.');
    print(parser.usage);
    exit(0);
  }

  final source = argResults['source'] as String;
  final outputDir = argResults['output-dir'] as String;
  final quantizeFp16 = argResults['quantize-fp16'] as bool;

  print('Starting model conversion for: $source');
  await Directory(outputDir).create(recursive: true);

  if (source.endsWith('.mlmodel')) {
    await _convertCoreML(source, outputDir);
  } else if (source.endsWith('.h5') || Directory(source).existsSync()) {
    await _convertTensorFlow(source, outputDir, quantizeFp16: quantizeFp16);
  } else {
    print('Unsupported source type: $source. Provide a .mlmodel, .h5, or SavedModel directory.');
    exit(1);
  }
}

Future<void> _convertCoreML(String source, String outputDir) async {
  if (!Platform.isMacOS) {
    print('❌ CoreML compilation is only supported on macOS.');
    return;
  }
  final baseName = p.basenameWithoutExtension(source);
  final outputPath = p.join(outputDir, '$baseName.mlmodelc');
  print('Found CoreML model. Compiling to $outputPath...');
  try {
    await shell.run('xcrun coremlcompiler compile "$source" "$outputPath"');
    print('✅ CoreML compilation successful.');
  } catch (e) {
    print('❌ Error compiling CoreML model. Make sure Xcode Command Line Tools are installed.');
    print(e);
  }
}

Future<bool> isInstalled(String command) async {
  final path = await which(command);
  return path != null;
}


Future<void> _convertTensorFlow(String source, String outputDir, {required bool quantizeFp16}) async {
   if (!await isInstalled('python3') || !await isInstalled('pip3')) {
    print('❌ python3 and pip3 must be installed and in your PATH to convert TensorFlow models.');
    return;
  }

  print('⏳ Converting TensorFlow model. This may install/use tensorflow packages.');
  final baseName = p.basenameWithoutExtension(source);
  final outputPath = p.join(outputDir, '$baseName.tflite');
  
  final scriptContent = """
import tensorflow as tf
import sys
import os

source_path = sys.argv[1]
output_path = sys.argv[2]
quantize = sys.argv[3] == 'true'

try:
    if os.path.isdir(source_path):
        print(f"Loading from SavedModel directory: {source_path}")
        converter = tf.lite.TFLiteConverter.from_saved_model(source_path)
    else:
        print(f"Loading from Keras/H5 file: {source_path}")
        converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(source_path))

    if quantize:
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Successfully converted model to {output_path}")
except Exception as e:
    print(f"❌ An error occurred during TFLite conversion: {e}")
    sys.exit(1)
""";

  final scriptFile = File('.tflite_converter.py')..writeAsStringSync(scriptContent);
  
  try {
    print('Checking for TensorFlow dependency...');
    await shell.run('pip3 install -q tensorflow');
    print('Running conversion script...');
    await shell.run('python3 "${scriptFile.path}" "$source" "$outputPath" $quantizeFp16');
  } catch(e) {
    print('❌ TensorFlow Lite conversion failed.');
    print('Please ensure your Python environment is correctly set up.');
  } finally {
    await scriptFile.delete();
  }
}