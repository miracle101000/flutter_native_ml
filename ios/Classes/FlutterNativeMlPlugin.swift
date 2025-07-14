import Flutter
import UIKit
import CoreML

// Struct to hold model + signature
struct LoadedModel {
  let model: MLModel
  let signature: [String: Any]
}

// Plugin supports MethodChannel + dynamic EventChannels per model
public class FlutterNativeMlPlugin: NSObject, FlutterPlugin, FlutterStreamHandler {

  private var loadedModels: [String: LoadedModel] = [:]
  private var eventSinks: [String: FlutterEventSink] = [:]

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "flutter_native_ml", binaryMessenger: registrar.messenger())
    let instance = FlutterNativeMlPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    DispatchQueue.global(qos: .userInitiated).async {
      switch call.method {
      case "loadModel":
        self.handleLoadModel(call: call, result: result, registrar: registrar)
      case "getSignature":
        self.handleGetSignature(call: call, result: result)
      case "run":
        self.handleRun(call: call, result: result)
      case "dispose":
        self.handleDispose(call: call, result: result)
      case "startStream":
        self.handleStartStream(call: call, result: result)
      case "stopStream":
        self.handleStopStream(call: call, result: result)
      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }

  private func handleLoadModel(call: FlutterMethodCall, result: @escaping FlutterResult, registrar: FlutterPluginRegistrar) {
    guard let args = call.arguments as? [String: Any],
          let assetPath = args["assetPath"] as? String,
          let computeUnitsString = args["computeUnits"] as? String else {
      return result(FlutterError(code: "INVALID_ARGS", message: nil, details: nil))
    }

    do {
      let key = FlutterDartProject.lookupKey(forAsset: assetPath)
      guard let path = Bundle.main.path(forResource: key, ofType: nil),
            let url = URL(string: "file://\(path)") else {
        return result(FlutterError(code: "MODEL_NOT_FOUND", message: assetPath, details: nil))
      }

      let config = MLModelConfiguration()
      config.computeUnits = (computeUnitsString == "cpuOnly") ? .cpuOnly : .all
      let model = try MLModel(contentsOf: url, configuration: config)
      let signature = generateSignature(for: model)
      let modelId = UUID().uuidString
      loadedModels[modelId] = LoadedModel(model: model, signature: signature)

      // Dynamically register EventChannel for streaming
      let eventChannel = FlutterEventChannel(
        name: "flutter_native_ml_stream/\(modelId)",
        binaryMessenger: registrar.messenger()
      )
      eventChannel.setStreamHandler(self)

      DispatchQueue.main.async { result(modelId) }
    } catch {
      DispatchQueue.main.async {
        result(FlutterError(code: "LOAD_FAILED", message: error.localizedDescription, details: nil))
      }
    }
  }

    private func handleGetSignature(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let modelId = (call.arguments as? [String: Any])?["modelId"] as? String,
              let loadedModel = loadedModels[modelId] else {
            return result(FlutterError(code: "MODEL_NOT_FOUND", message: "Signature request for unknown model ID", details: nil))
        }
        DispatchQueue.main.async { result(loadedModel.signature) }
    }

    private func handleRun(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let modelId = args["modelId"] as? String,
              let inputMap = args["input"] as? [String: Any] else {
            return result(FlutterError(code: "INVALID_ARGUMENTS", message: "Missing modelId or input map", details: nil))
        }

        guard let loadedModel = loadedModels[modelId] else {
            return result(FlutterError(code: "MODEL_NOT_LOADED", message: "No model loaded for ID: \(modelId).", details: nil))
        }

        do {
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: convertToMLMultiArray(inputMap: inputMap))

            let startTime = CACurrentMediaTime()
            let outputProvider = try loadedModel.model.prediction(from: inputProvider)
            let latency = (CACurrentMediaTime() - startTime) * 1_000_000

            let outputDict = self.convertMLOutputToDict(outputProvider)

            let resultPayload: [String: Any] = [
                "output": outputDict,
                "inferenceTime": latency,
                "acceleratorUsed": getAcceleratorName(config: loadedModel.model.configuration)
            ]
            DispatchQueue.main.async { result(resultPayload) }
        } catch {
            DispatchQueue.main.async {
                result(FlutterError(code: "INFERENCE_FAILED", message: error.localizedDescription, details: nil))
            }
        }
    }

    private func handleDispose(...) {
    loadedModels.removeValue(forKey: modelId)
    eventSinks.removeValue(forKey: modelId)
    DispatchQueue.main.async { result(nil) }
  }

    // MARK: - Streaming
  public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
    guard let modelId = arguments as? String else {
      return FlutterError(code: "BAD_ARGS", message: "Model ID required to start stream", details: nil)
    }
    eventSinks[modelId] = events
    return nil
  }

  public func onCancel(withArguments arguments: Any?) -> FlutterError? {
    guard let modelId = arguments as? String else {
      return FlutterError(code: "BAD_ARGS", message: "Model ID required to stop stream", details: nil)
    }
    eventSinks.removeValue(forKey: modelId)
    return nil
  }

  private func handleStartStream(call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let modelId = (call.arguments as? [String: Any])?["modelId"] as? String else {
      return result(FlutterError(code: "BAD_ARGS", message: "Missing modelId", details: nil))
    }
    result(nil)

    // Example: periodically send timestamps
    Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { timer in
      guard let sink = self.eventSinks[modelId] else {
        timer.invalidate()
        return
      }
      DispatchQueue.main.async {
        sink(["time": CACurrentMediaTime()])
      }
    }
  }

  private func handleStopStream(call: FlutterMethodCall, result: @escaping FlutterResult) {
    // Dart-side will cancel the stream and the timer stops itself
    result(nil)
  }

    // MARK: - Helpers
    private func generateSignature(for model: MLModel) -> [String: Any] {
        let inputs = model.modelDescription.inputDescriptionsByName.map { (name, desc) -> [String: Any] in
            let shape = desc.multiArrayConstraint?.shape.map { $0.intValue } ?? []
            return ["name": name, "shape": shape, "dataType": desc.type.toString()]
        }
        let outputs = model.modelDescription.outputDescriptionsByName.map { (name, desc) -> [String: Any] in
            let shape = desc.multiArrayConstraint?.shape.map { $0.intValue } ?? []
            return ["name": name, "shape": shape, "dataType": desc.type.toString()]
        }
        return ["inputs": inputs, "outputs": outputs]
    }

    private func convertToMLMultiArray(inputMap: [String: Any]) throws -> [String: MLMultiArray] {
        var multiArrayDict = [String: MLMultiArray]()
        for (key, value) in inputMap {
            if let list = value as? [Double] {
                let shape: [NSNumber] = [1, NSNumber(value: list.count)]
                let multiArray = try MLMultiArray(shape: shape, dataType: .float32)
                for (index, element) in list.enumerated() {
                    multiArray[index] = NSNumber(value: element)
                }
                multiArrayDict[key] = multiArray
            }
        }
        return multiArrayDict
    }

    private func convertMLOutputToDict(_ provider: MLFeatureProvider) -> [String: Any] {
        var outputDict = [String: Any]()
        for featureName in provider.featureNames {
            guard let featureValue = provider.featureValue(for: featureName) else { continue }
            if featureValue.type == .multiArray, let multiArray = featureValue.multiArrayValue {
                let pointer = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: multiArray.count)
                let buffer = UnsafeBufferPointer(start: pointer, count: multiArray.count)
                outputDict[featureName] = Array(buffer)
            }
        }
        return outputDict
    }

    private func getAcceleratorName(config: MLModelConfiguration) -> String {
        switch config.computeUnits {
        case .cpuOnly: return "CPU"
        case .cpuAndGPU: return "GPU"
        case .all: return "Neural Engine (ANE)"
        @unknown default: return "Unknown"
        }
    }
}

// Helper extension
extension MLFeatureType {
    func toString() -> String {
        switch self {
        case .double: return "float64"
        case .float: return "float32"
        case .int64: return "int64"
        case .string: return "string"
        case .multiArray: return "multiArray"
        case .dictionary: return "dictionary"
        case .image: return "image"
        case .sequence: return "sequence"
        @unknown default: return "unknown"
        }
    }
}
