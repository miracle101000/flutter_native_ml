import CoreGraphics
import CoreML
import CoreVideo
import Flutter
import QuartzCore
import UIKit

/// An error raised by the native layer. Surfaces in Dart as a `PlatformException` with `code`.
struct NativeMLError: Error {
    let code: String
    let message: String
    let details: Any?

    init(_ code: String, _ message: String, details: Any? = nil) {
        self.code = code
        self.message = message
        self.details = details
    }
}

/// A loaded Core ML model together with its streaming state.
final class LoadedModel {
    let id: String
    let model: MLModel
    let computeUnits: MLComputeUnits
    let acceleratorLabel: String
    let signature: [String: Any]

    /// Serial queue: all predictions for this model run here, in order.
    let queue: DispatchQueue

    // Streaming state. `sink`, `eventChannel` and `streamHandler` are main-thread only;
    // the remaining fields are guarded by the plugin's `stateQueue`.
    var eventChannel: FlutterEventChannel?
    var streamHandler: ModelStreamHandler?
    var sink: FlutterEventSink?
    var pending: [(frameId: Int64, input: [String: Any])] = []
    var maxQueueSize = 2
    var streamActive = false
    var droppedFrames: Int64 = 0
    var nextFrameId: Int64 = 0
    var processing = false
    var disposed = false

    init(id: String, model: MLModel, computeUnits: MLComputeUnits, acceleratorLabel: String, signature: [String: Any]) {
        self.id = id
        self.model = model
        self.computeUnits = computeUnits
        self.acceleratorLabel = acceleratorLabel
        self.signature = signature
        self.queue = DispatchQueue(label: "flutter_native_ml.model.\(id)", qos: .userInitiated)
    }
}

/// Bridges one model's `FlutterEventChannel` to the plugin.
final class ModelStreamHandler: NSObject, FlutterStreamHandler {
    private let onListenHandler: (@escaping FlutterEventSink) -> Void
    private let onCancelHandler: () -> Void

    init(onListen: @escaping (@escaping FlutterEventSink) -> Void, onCancel: @escaping () -> Void) {
        onListenHandler = onListen
        onCancelHandler = onCancel
    }

    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        onListenHandler(events)
        return nil
    }

    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        onCancelHandler()
        return nil
    }
}

/// iOS implementation of `flutter_native_ml`, backed by Core ML.
///
/// Threading model:
///  * Method calls arrive on the main thread. Cheap calls (streaming bookkeeping,
///    disposal) are handled there directly.
///  * Model loading and capability checks run on a concurrent work queue.
///  * Every model has its own serial queue on which all predictions (single runs and
///    stream frames) execute in order.
///  * Results are always delivered to Flutter on the main thread.
public class FlutterNativeMlPlugin: NSObject, FlutterPlugin {
    private static let methodChannelName = "flutter_native_ml"
    private static let streamChannelPrefix = "flutter_native_ml_stream/"
    private static let defaultMaxQueueSize = 2

    private var registrar: FlutterPluginRegistrar?
    private var models: [String: LoadedModel] = [:]
    private let stateQueue = DispatchQueue(label: "flutter_native_ml.state")
    private let workQueue = DispatchQueue(label: "flutter_native_ml.work", qos: .userInitiated, attributes: .concurrent)

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: methodChannelName, binaryMessenger: registrar.messenger())
        let instance = FlutterNativeMlPlugin()
        instance.registrar = registrar
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func detachFromEngine(for registrar: FlutterPluginRegistrar) {
        disposeAll()
        self.registrar = nil
    }

    // MARK: - Dispatch

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        let args = call.arguments as? [String: Any] ?? [:]
        switch call.method {
        case "getPlatformVersion":
            result("iOS " + UIDevice.current.systemVersion)
        case "getDeviceCapabilities":
            workQueue.async { self.reply(result) { self.deviceCapabilities() } }
        case "loadModel":
            workQueue.async { self.reply(result) { try self.loadModel(args) } }
        case "getSignature":
            do {
                let model = try requireModel(args)
                result(model.signature)
            } catch {
                fail(result, error)
            }
        case "run":
            do {
                let model = try requireModel(args)
                guard let input = args["input"] as? [String: Any] else {
                    throw NativeMLError("INVALID_ARGS", "'input' must be a map of input name to data")
                }
                model.queue.async { self.reply(result) { try self.run(model: model, input: input) } }
            } catch {
                fail(result, error)
            }
        case "startStream":
            reply(result) { try self.startStream(args) }
        case "streamInput":
            reply(result) { try self.streamInput(args) }
        case "stopStream":
            reply(result) { try self.stopStream(args) }
        case "dispose":
            reply(result) { try self.dispose(args) }
        case "disposeAll":
            disposeAll()
            result(nil)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    /// Runs `block` on the current thread and delivers its outcome on the main thread.
    private func reply(_ result: @escaping FlutterResult, _ block: () throws -> Any?) {
        do {
            let value = try block()
            deliver(result, value)
        } catch {
            fail(result, error)
        }
    }

    private func deliver(_ result: @escaping FlutterResult, _ value: Any?) {
        if Thread.isMainThread {
            result(value)
        } else {
            DispatchQueue.main.async { result(value) }
        }
    }

    private func fail(_ result: @escaping FlutterResult, _ error: Error) {
        deliver(result, flutterError(from: error))
    }

    private func flutterError(from error: Error) -> FlutterError {
        if let native = error as? NativeMLError {
            return FlutterError(code: native.code, message: native.message, details: native.details)
        }
        let nsError = error as NSError
        return FlutterError(code: "NATIVE_ERROR", message: error.localizedDescription, details: "\(nsError.domain) (\(nsError.code))")
    }

    private func requireModel(_ args: [String: Any]) throws -> LoadedModel {
        guard let modelId = args["modelId"] as? String else {
            throw NativeMLError("INVALID_ARGS", "'modelId' is required")
        }
        guard let model = stateQueue.sync(execute: { models[modelId] }) else {
            throw NativeMLError("MODEL_NOT_FOUND", "No loaded model with id '\(modelId)'")
        }
        if stateQueue.sync(execute: { model.disposed }) {
            throw NativeMLError("MODEL_DISPOSED", "Model \(modelId) has been disposed")
        }
        return model
    }

    // MARK: - Loading

    private func loadModel(_ args: [String: Any]) throws -> [String: Any] {
        guard let registrar = registrar else {
            throw NativeMLError("NOT_ATTACHED", "Plugin is not attached to a Flutter engine")
        }
        let url = try resolveModelURL(
            assetPath: args["assetPath"] as? String,
            filePath: args["filePath"] as? String,
            registrar: registrar
        )
        let compiledURL = try compileIfNeeded(url)

        let computeUnits = mapComputeUnits(args["computeUnits"] as? String ?? "all")
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        configuration.allowLowPrecisionAccumulationOnGPU = args["allowFp16"] as? Bool ?? false

        let model: MLModel
        do {
            model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } catch {
            throw NativeMLError("LOAD_FAILED", "Could not load '\(compiledURL.lastPathComponent)': \(error.localizedDescription)")
        }

        let id = UUID().uuidString
        let loaded = LoadedModel(
            id: id,
            model: model,
            computeUnits: computeUnits,
            acceleratorLabel: acceleratorName(computeUnits),
            signature: buildSignature(model)
        )
        stateQueue.sync { models[id] = loaded }
        return ["modelId": id, "acceleratorUsed": loaded.acceleratorLabel, "signature": loaded.signature]
    }

    private func resolveModelURL(assetPath: String?, filePath: String?, registrar: FlutterPluginRegistrar) throws -> URL {
        if let filePath = filePath, !filePath.isEmpty {
            let path = filePath.hasPrefix("file://") ? String(filePath.dropFirst(7)) : filePath
            guard FileManager.default.fileExists(atPath: path) else {
                throw NativeMLError("MODEL_NOT_FOUND", "No model file at '\(path)'")
            }
            return URL(fileURLWithPath: path)
        }
        guard let assetPath = assetPath, !assetPath.isEmpty else {
            throw NativeMLError("INVALID_ARGS", "Provide either 'assetPath' or 'filePath'")
        }
        if assetPath.hasPrefix("/"), FileManager.default.fileExists(atPath: assetPath) {
            return URL(fileURLWithPath: assetPath)
        }

        let key = registrar.lookupKey(forAsset: assetPath)
        if let url = Bundle.main.url(forResource: key, withExtension: nil) {
            return url
        }
        // Directories (e.g. compiled .mlmodelc bundles) are not always resolved by url(forResource:).
        let candidate = Bundle.main.bundleURL.appendingPathComponent(key)
        if FileManager.default.fileExists(atPath: candidate.path) {
            return candidate
        }
        throw NativeMLError(
            "MODEL_NOT_FOUND",
            "Asset '\(assetPath)' (bundle key '\(key)') was not found. Declare it under `flutter: assets:` in pubspec.yaml. "
                + "Tip: compiled .mlmodelc models are directories, which Flutter does not bundle as a single asset; "
                + "ship the .mlmodel file instead and the plugin compiles it on device."
        )
    }

    /// Compiles `.mlmodel` / `.mlpackage` sources on device and caches the result.
    private func compileIfNeeded(_ url: URL) throws -> URL {
        let ext = url.pathExtension.lowercased()
        guard ext == "mlmodel" || ext == "mlpackage" else { return url }

        let cacheDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("flutter_native_ml", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let baseName = url.deletingPathExtension().lastPathComponent
        let cached = cacheDirectory.appendingPathComponent("\(baseName)-\(fingerprint(of: url)).mlmodelc", isDirectory: true)
        if FileManager.default.fileExists(atPath: cached.path) {
            return cached
        }

        let compiled: URL
        do {
            compiled = try compileModel(at: url)
        } catch {
            throw NativeMLError("COMPILE_FAILED", "Could not compile '\(url.lastPathComponent)': \(error.localizedDescription)")
        }
        do {
            if FileManager.default.fileExists(atPath: cached.path) {
                try FileManager.default.removeItem(at: cached)
            }
            try FileManager.default.moveItem(at: compiled, to: cached)
            return cached
        } catch {
            return compiled // Caching failed; use the temporary compiled model instead.
        }
    }

    private func compileModel(at url: URL) throws -> URL {
        if #available(iOS 16.0, *) {
            final class Box { var result: Result<URL, Error>? }
            let box = Box()
            let semaphore = DispatchSemaphore(value: 0)
            Task {
                do {
                    box.result = .success(try await MLModel.compileModel(at: url))
                } catch {
                    box.result = .failure(error)
                }
                semaphore.signal()
            }
            semaphore.wait()
            return try box.result!.get()
        } else {
            return try MLModel.compileModel(at: url)
        }
    }

    /// A stable fingerprint of a model file (path, size and a sample of its bytes) used as cache key.
    private func fingerprint(of url: URL) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        func mix(_ bytes: [UInt8]) {
            for byte in bytes {
                hash ^= UInt64(byte)
                hash = hash &* 0x0000_0100_0000_01b3
            }
        }
        mix(Array(url.path.utf8))
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
        if isDirectory.boolValue {
            let enumerator = FileManager.default.enumerator(atPath: url.path)
            while let entry = enumerator?.nextObject() as? String {
                mix(Array(entry.utf8))
                if let size = try? FileManager.default.attributesOfItem(atPath: url.appendingPathComponent(entry).path)[.size] as? NSNumber {
                    mix(Array("\(size.int64Value)".utf8))
                }
            }
        } else if let handle = try? FileHandle(forReadingFrom: url) {
            defer { try? handle.close() }
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? 0
            mix(Array("\(size)".utf8))
            let sample = 64 * 1024
            mix(Array(handle.readData(ofLength: sample)))
            if size > Int64(sample) * 2 {
                handle.seek(toFileOffset: UInt64(size - Int64(sample)))
                mix(Array(handle.readData(ofLength: sample)))
            }
        }
        return String(hash, radix: 16)
    }

    private func mapComputeUnits(_ name: String) -> MLComputeUnits {
        switch name {
        case "cpuOnly":
            return .cpuOnly
        case "cpuAndGpu", "cpuAndGPU":
            return .cpuAndGPU
        case "cpuAndNeuralEngine", "cpuAndAne":
            if #available(iOS 16.0, *) {
                return .cpuAndNeuralEngine
            }
            return .all // Older systems: `.all` already includes the Neural Engine.
        default:
            return .all
        }
    }

    private func acceleratorName(_ units: MLComputeUnits) -> String {
        if #available(iOS 16.0, *), units == .cpuAndNeuralEngine {
            return "CPU+ANE"
        }
        switch units {
        case .cpuOnly:
            return "CPU"
        case .cpuAndGPU:
            return "CPU+GPU"
        case .all:
            return "CPU+GPU+ANE"
        default:
            return "unknown"
        }
    }

    // MARK: - Signature

    private func buildSignature(_ model: MLModel) -> [String: Any] {
        let description = model.modelDescription
        let inputs = description.inputDescriptionsByName
            .sorted { $0.key < $1.key }
            .map { featureInfo(name: $0.key, description: $0.value) }
        let outputs = description.outputDescriptionsByName
            .sorted { $0.key < $1.key }
            .map { featureInfo(name: $0.key, description: $0.value) }

        var metadata: [String: Any] = [:]
        for (key, value) in description.metadata {
            metadata[key.rawValue] = "\(value)"
        }
        if #available(iOS 14.0, *), let labels = description.classLabels {
            metadata["classLabels"] = labels.map { "\($0)" }
        }
        if let name = description.predictedFeatureName {
            metadata["predictedFeatureName"] = name
        }
        if let name = description.predictedProbabilitiesName {
            metadata["predictedProbabilitiesName"] = name
        }
        metadata["runtime"] = "Core ML (iOS \(UIDevice.current.systemVersion))"

        return [
            "inputs": inputs,
            "outputs": outputs,
            "signatureKeys": [String](),
            "signatures": [String: Any](),
            "metadata": metadata,
        ]
    }

    private func featureInfo(name: String, description: MLFeatureDescription) -> [String: Any] {
        var info: [String: Any] = ["name": name, "isOptional": description.isOptional, "shape": [Int]()]
        switch description.type {
        case .multiArray:
            if let constraint = description.multiArrayConstraint {
                info["shape"] = constraint.shape.map { $0.intValue }
                info["dataType"] = dataTypeName(constraint.dataType)
                switch constraint.shapeConstraint.type {
                case .enumerated:
                    info["isFlexible"] = true
                    info["enumeratedShapes"] = constraint.shapeConstraint.enumeratedShapes.map { $0.map { $0.intValue } }
                case .range:
                    info["isFlexible"] = true
                    info["shapeRanges"] = constraint.shapeConstraint.sizeRangeForDimension.map { value -> [String: Int] in
                        let range = value.rangeValue
                        return ["min": range.location, "max": range.location + range.length - 1]
                    }
                case .unspecified:
                    break
                @unknown default:
                    break
                }
            } else {
                info["dataType"] = "float32"
            }
        case .image:
            info["dataType"] = "image"
            if let constraint = description.imageConstraint {
                info["shape"] = [constraint.pixelsHigh, constraint.pixelsWide, channelCount(forPixelFormat: constraint.pixelFormatType)]
                info["imageWidth"] = constraint.pixelsWide
                info["imageHeight"] = constraint.pixelsHigh
                info["pixelFormat"] = pixelFormatName(constraint.pixelFormatType)
            }
        case .string:
            info["dataType"] = "string"
        case .int64:
            info["dataType"] = "int64"
        case .double:
            info["dataType"] = "float64"
        case .dictionary:
            info["dataType"] = "dictionary"
            if let constraint = description.dictionaryConstraint {
                info["keyType"] = constraint.keyType == .int64 ? "int64" : "string"
            }
        case .sequence:
            info["dataType"] = "sequence"
            if let constraint = description.sequenceConstraint {
                info["elementType"] = constraint.valueDescription.type == .int64 ? "int64" : "string"
                info["minCount"] = constraint.countRange.location
                info["maxCount"] = constraint.countRange.location + constraint.countRange.length - 1
            }
        case .invalid:
            info["dataType"] = "unknown"
        @unknown default:
            info["dataType"] = "unknown"
        }
        return info
    }

    private func dataTypeName(_ type: MLMultiArrayDataType) -> String {
        switch type {
        case .double:
            return "float64"
        case .float32:
            return "float32"
        case .float16:
            return "float16"
        case .int32:
            return "int32"
        @unknown default:
            return "unknown"
        }
    }

    private func pixelFormatName(_ format: OSType) -> String {
        switch format {
        case kCVPixelFormatType_32BGRA:
            return "bgra"
        case kCVPixelFormatType_32ARGB:
            return "argb"
        case kCVPixelFormatType_32RGBA:
            return "rgba"
        case kCVPixelFormatType_OneComponent8:
            return "grayscale"
        default:
            let bytes = [24, 16, 8, 0].map { Character(UnicodeScalar(UInt8((format >> UInt32($0)) & 0xFF))) }
            return String(bytes)
        }
    }

    private func channelCount(forPixelFormat format: OSType) -> Int {
        format == kCVPixelFormatType_OneComponent8 ? 1 : 4
    }

    // MARK: - Inference

    /// Runs one prediction. Must be called on the model's serial queue.
    private func run(model: LoadedModel, input: [String: Any]) throws -> [String: Any] {
        let provider = try makeFeatureProvider(model: model, input: input)
        let start = CACurrentMediaTime()
        let outputProvider: MLFeatureProvider
        do {
            outputProvider = try model.model.prediction(from: provider)
        } catch {
            throw NativeMLError("INFERENCE_FAILED", error.localizedDescription)
        }
        let micros = (CACurrentMediaTime() - start) * 1_000_000
        let (output, shapes) = convertOutput(outputProvider)
        return [
            "output": output,
            "outputShapes": shapes,
            "inferenceTime": micros,
            "acceleratorUsed": model.acceleratorLabel,
        ]
    }

    private func makeFeatureProvider(model: LoadedModel, input: [String: Any]) throws -> MLFeatureProvider {
        let descriptions = model.model.modelDescription.inputDescriptionsByName
        var features: [String: MLFeatureValue] = [:]
        for (name, raw) in input {
            guard let description = descriptions[name] else {
                throw NativeMLError("INPUT_MISMATCH", "No input named '\(name)'. Available inputs: \(descriptions.keys.sorted())")
            }
            features[name] = try featureValue(for: description, name: name, raw: raw)
        }
        let missing = descriptions.values
            .filter { !$0.isOptional && features[$0.name] == nil }
            .map { $0.name }
            .sorted()
        if !missing.isEmpty {
            throw NativeMLError("MISSING_INPUT", "Missing input(s) \(missing). Provided: \(input.keys.sorted())")
        }
        do {
            return try MLDictionaryFeatureProvider(dictionary: features)
        } catch {
            throw NativeMLError("INVALID_INPUT", error.localizedDescription)
        }
    }

    /// Splits `{data:, shape:, ...}` wrappers into their parts; plain values pass through.
    private func unpack(_ raw: Any) -> (data: Any, shape: [Int]?, options: [String: Any]) {
        if let map = raw as? [String: Any], let data = map["data"] {
            let shape = (map["shape"] as? [Any])?.compactMap { ($0 as? NSNumber)?.intValue }
            return (data, shape, map)
        }
        return (raw, nil, [:])
    }

    private func featureValue(for description: MLFeatureDescription, name: String, raw: Any) throws -> MLFeatureValue {
        switch description.type {
        case .multiArray:
            guard let constraint = description.multiArrayConstraint else {
                throw NativeMLError("UNSUPPORTED_INPUT", "Input '\(name)' has no multi-array constraint")
            }
            let (data, shape, _) = unpack(raw)
            return MLFeatureValue(multiArray: try makeMultiArray(constraint: constraint, name: name, data: data, explicitShape: shape))
        case .image:
            guard let constraint = description.imageConstraint else {
                throw NativeMLError("UNSUPPORTED_INPUT", "Input '\(name)' has no image constraint")
            }
            let (data, _, options) = unpack(raw)
            return MLFeatureValue(pixelBuffer: try makePixelBuffer(constraint: constraint, name: name, data: data, options: options))
        case .string:
            guard let string = raw as? String else {
                throw NativeMLError("INVALID_INPUT", "Input '\(name)' expects a String")
            }
            return MLFeatureValue(string: string)
        case .int64:
            guard let number = raw as? NSNumber else {
                throw NativeMLError("INVALID_INPUT", "Input '\(name)' expects an integer")
            }
            return MLFeatureValue(int64: number.int64Value)
        case .double:
            guard let number = raw as? NSNumber else {
                throw NativeMLError("INVALID_INPUT", "Input '\(name)' expects a number")
            }
            return MLFeatureValue(double: number.doubleValue)
        case .dictionary:
            guard let dictionary = raw as? [AnyHashable: Any] else {
                throw NativeMLError("INVALID_INPUT", "Input '\(name)' expects a Map<String, num> or Map<int, num>")
            }
            let keyType = description.dictionaryConstraint?.keyType ?? .string
            var converted: [AnyHashable: NSNumber] = [:]
            for (key, value) in dictionary {
                guard let number = value as? NSNumber else {
                    throw NativeMLError("INVALID_INPUT", "Dictionary input '\(name)' contains a non-numeric value for key '\(key)'")
                }
                if keyType == .int64 {
                    let intKey = (key as? NSNumber)?.int64Value ?? Int64("\(key)") ?? 0
                    converted[intKey] = number
                } else {
                    converted["\(key)"] = number
                }
            }
            do {
                return try MLFeatureValue(dictionary: converted)
            } catch {
                throw NativeMLError("INVALID_INPUT", "Dictionary input '\(name)': \(error.localizedDescription)")
            }
        case .sequence:
            guard let list = raw as? [Any] else {
                throw NativeMLError("INVALID_INPUT", "Input '\(name)' expects a List<String> or List<int>")
            }
            if let strings = list as? [String] {
                return MLFeatureValue(sequence: MLSequence(strings: strings))
            }
            if let numbers = list as? [NSNumber] {
                return MLFeatureValue(sequence: MLSequence(int64s: numbers))
            }
            throw NativeMLError("INVALID_INPUT", "Sequence input '\(name)' must contain only strings or only integers")
        case .invalid:
            throw NativeMLError("UNSUPPORTED_INPUT", "Input '\(name)' has an invalid feature type")
        @unknown default:
            throw NativeMLError("UNSUPPORTED_INPUT", "Input '\(name)' has an unsupported feature type")
        }
    }

    // MARK: Multi-arrays

    private func elementCount(of data: Any, name: String) throws -> Int {
        if let typed = data as? FlutterStandardTypedData {
            return Int(typed.elementCount)
        }
        if let list = data as? [Any] {
            return list.count
        }
        if data is NSNumber {
            return 1
        }
        throw NativeMLError(
            "UNSUPPORTED_INPUT",
            "Input '\(name)' has unsupported type \(type(of: data)). Use a List<num>, a typed list (Float32List, ...) or TensorData."
        )
    }

    private func resolveShape(constraint: MLMultiArrayConstraint, name: String, count: Int, explicit: [Int]?) throws -> [Int] {
        if let explicit = explicit {
            guard !explicit.isEmpty, explicit.allSatisfy({ $0 > 0 }) else {
                throw NativeMLError("INVALID_SHAPE", "Shape for '\(name)' must have positive dimensions: \(explicit)")
            }
            let implied = explicit.reduce(1, *)
            guard implied == count else {
                throw NativeMLError("SHAPE_MISMATCH", "Shape \(explicit) for '\(name)' implies \(implied) elements but \(count) were provided")
            }
            return explicit
        }

        let declared = constraint.shape.map { $0.intValue }
        let declaredCount = declared.reduce(1, *)
        if !declared.isEmpty, declaredCount == count {
            return declared
        }

        switch constraint.shapeConstraint.type {
        case .enumerated:
            for candidate in constraint.shapeConstraint.enumeratedShapes {
                let shape = candidate.map { $0.intValue }
                if shape.reduce(1, *) == count {
                    return shape
                }
            }
        case .range:
            let ranges = constraint.shapeConstraint.sizeRangeForDimension.map { $0.rangeValue }
            let flexible = ranges.indices.filter { ranges[$0].length > 1 }
            if flexible.count == 1, ranges.count == declared.count {
                let fixed = ranges.indices.filter { $0 != flexible[0] }.reduce(1) { $0 * declared[$1] }
                if fixed > 0, count % fixed == 0 {
                    let size = count / fixed
                    let range = ranges[flexible[0]]
                    if size >= range.location, size < range.location + range.length {
                        var shape = declared
                        shape[flexible[0]] = size
                        return shape
                    }
                }
            }
        case .unspecified:
            break
        @unknown default:
            break
        }

        if declared.isEmpty {
            return [count]
        }
        throw NativeMLError(
            "SHAPE_MISMATCH",
            "Input '\(name)' expects \(declaredCount) elements (shape \(declared)) but \(count) were provided. "
                + "Pass TensorData(data, shape: [...]) to use a different shape."
        )
    }

    private func makeMultiArray(constraint: MLMultiArrayConstraint, name: String, data: Any, explicitShape: [Int]?) throws -> MLMultiArray {
        let count = try elementCount(of: data, name: name)
        let shape = try resolveShape(constraint: constraint, name: name, count: count, explicit: explicitShape)
        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: constraint.dataType)
        } catch {
            throw NativeMLError("INVALID_SHAPE", "Could not allocate a \(dataTypeName(constraint.dataType)) array of shape \(shape) for '\(name)': \(error.localizedDescription)")
        }
        try fill(array, with: data, name: name)
        return array
    }

    private func bytesPerElement(_ type: MLMultiArrayDataType) -> Int {
        switch type {
        case .double:
            return 8
        case .float32, .int32:
            return 4
        case .float16:
            return 2
        @unknown default:
            return 4
        }
    }

    /// Byte offsets of every element in linear order, or nil when the array is contiguous.
    private func elementOffsets(shape: [Int], strides: [Int], count: Int) -> [Int]? {
        var expected = 1
        var contiguous = true
        for k in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[k] != expected {
                contiguous = false
                break
            }
            expected *= shape[k]
        }
        if contiguous || shape.isEmpty {
            return nil
        }
        var offsets = [Int](repeating: 0, count: count)
        var index = [Int](repeating: 0, count: shape.count)
        for i in 0..<count {
            var offset = 0
            for k in 0..<shape.count {
                offset += index[k] * strides[k]
            }
            offsets[i] = offset
            var k = shape.count - 1
            while k >= 0 {
                index[k] += 1
                if index[k] < shape[k] {
                    break
                }
                index[k] = 0
                k -= 1
            }
        }
        return offsets
    }

    private func withWritableBytes<R>(_ array: MLMultiArray, _ body: (UnsafeMutableRawPointer) throws -> R) throws -> R {
        if #available(iOS 15.4, *) {
            return try array.withUnsafeMutableBytes { buffer, _ in
                guard let base = buffer.baseAddress else {
                    throw NativeMLError("INTERNAL", "Multi-array has no storage")
                }
                return try body(base)
            }
        } else {
            return try body(array.dataPointer)
        }
    }

    private func withReadableBytes<R>(_ array: MLMultiArray, _ body: (UnsafeRawPointer) throws -> R) throws -> R {
        if #available(iOS 15.4, *) {
            return try array.withUnsafeBytes { buffer in
                guard let base = buffer.baseAddress else {
                    throw NativeMLError("INTERNAL", "Multi-array has no storage")
                }
                return try body(base)
            }
        } else {
            return try body(UnsafeRawPointer(array.dataPointer))
        }
    }

    private func fill(_ array: MLMultiArray, with data: Any, name: String) throws {
        let type = array.dataType
        let count = array.count
        let offsets = elementOffsets(shape: array.shape.map { $0.intValue }, strides: array.strides.map { $0.intValue }, count: count)
        let elementSize = bytesPerElement(type)

        try withWritableBytes(array) { base in
            func store(_ i: Int, _ value: Double) {
                let byteOffset = (offsets?[i] ?? i) * elementSize
                switch type {
                case .float32:
                    base.storeBytes(of: Float(value), toByteOffset: byteOffset, as: Float.self)
                case .double:
                    base.storeBytes(of: value, toByteOffset: byteOffset, as: Double.self)
                case .int32:
                    let clamped = value.isFinite ? min(max(value.rounded(), Double(Int32.min)), Double(Int32.max)) : 0
                    base.storeBytes(of: Int32(clamped), toByteOffset: byteOffset, as: Int32.self)
                case .float16:
                    base.storeBytes(of: floatToHalf(Float(value)), toByteOffset: byteOffset, as: UInt16.self)
                @unknown default:
                    break
                }
            }

            if let typed = data as? FlutterStandardTypedData {
                guard Int(typed.elementCount) == count else {
                    throw NativeMLError("SHAPE_MISMATCH", "Input '\(name)' expects \(count) elements but \(typed.elementCount) were provided")
                }
                let sameLayout = (typed.type == .float32 && type == .float32)
                    || (typed.type == .float64 && type == .double)
                    || (typed.type == .int32 && type == .int32)
                if sameLayout, offsets == nil {
                    typed.data.withUnsafeBytes { source in
                        if let sourceBase = source.baseAddress {
                            base.copyMemory(from: sourceBase, byteCount: min(source.count, count * elementSize))
                        }
                    }
                    return
                }
                typed.data.withUnsafeBytes { source in
                    switch typed.type {
                    case .float32:
                        for i in 0..<count { store(i, Double(source.loadUnaligned(fromByteOffset: i * 4, as: Float.self))) }
                    case .float64:
                        for i in 0..<count { store(i, source.loadUnaligned(fromByteOffset: i * 8, as: Double.self)) }
                    case .int32:
                        for i in 0..<count { store(i, Double(source.loadUnaligned(fromByteOffset: i * 4, as: Int32.self))) }
                    case .int64:
                        for i in 0..<count { store(i, Double(source.loadUnaligned(fromByteOffset: i * 8, as: Int64.self))) }
                    case .uInt8:
                        for i in 0..<count { store(i, Double(source[i])) }
                    @unknown default:
                        break
                    }
                }
            } else if let list = data as? [Any] {
                guard list.count == count else {
                    throw NativeMLError("SHAPE_MISMATCH", "Input '\(name)' expects \(count) elements but \(list.count) were provided")
                }
                for (i, element) in list.enumerated() {
                    guard let number = element as? NSNumber else {
                        throw NativeMLError("UNSUPPORTED_INPUT", "Input '\(name)' contains a non-numeric element at index \(i)")
                    }
                    store(i, number.doubleValue)
                }
            } else if let number = data as? NSNumber, count == 1 {
                store(0, number.doubleValue)
            } else {
                throw NativeMLError("UNSUPPORTED_INPUT", "Input '\(name)' has unsupported type \(Swift.type(of: data))")
            }
        }
    }

    /// Converts a multi-array into typed data for Dart (`Float32List`, `Float64List`, `Int32List`).
    private func typedData(from array: MLMultiArray) throws -> Any {
        let count = array.count
        let offsets = elementOffsets(shape: array.shape.map { $0.intValue }, strides: array.strides.map { $0.intValue }, count: count)
        let type = array.dataType
        let elementSize = bytesPerElement(type)

        return try withReadableBytes(array) { base -> Any in
            func offset(_ i: Int) -> Int { (offsets?[i] ?? i) * elementSize }
            switch type {
            case .float32:
                var values = [Float](repeating: 0, count: count)
                for i in 0..<count { values[i] = base.load(fromByteOffset: offset(i), as: Float.self) }
                return FlutterStandardTypedData(float32: values.withUnsafeBufferPointer { Data(buffer: $0) })
            case .double:
                var values = [Double](repeating: 0, count: count)
                for i in 0..<count { values[i] = base.load(fromByteOffset: offset(i), as: Double.self) }
                return FlutterStandardTypedData(float64: values.withUnsafeBufferPointer { Data(buffer: $0) })
            case .int32:
                var values = [Int32](repeating: 0, count: count)
                for i in 0..<count { values[i] = base.load(fromByteOffset: offset(i), as: Int32.self) }
                return FlutterStandardTypedData(int32: values.withUnsafeBufferPointer { Data(buffer: $0) })
            case .float16:
                var values = [Float](repeating: 0, count: count)
                for i in 0..<count { values[i] = halfToFloat(base.load(fromByteOffset: offset(i), as: UInt16.self)) }
                return FlutterStandardTypedData(float32: values.withUnsafeBufferPointer { Data(buffer: $0) })
            @unknown default:
                var values = [Double](repeating: 0, count: count)
                for i in 0..<count { values[i] = array[i].doubleValue }
                return FlutterStandardTypedData(float64: values.withUnsafeBufferPointer { Data(buffer: $0) })
            }
        }
    }

    // MARK: Images

    private func makePixelBuffer(constraint: MLImageConstraint, name: String, data: Any, options: [String: Any]) throws -> CVPixelBuffer {
        guard let typed = data as? FlutterStandardTypedData, typed.type == .uInt8 else {
            throw NativeMLError("INVALID_INPUT", "Image input '\(name)' expects a Uint8List (use ImageInput)")
        }
        let bytes = typed.data
        let format = (options["format"] as? String)?.lowercased() ?? "rgba"
        let width = (options["width"] as? NSNumber)?.intValue ?? constraint.pixelsWide
        let height = (options["height"] as? NSNumber)?.intValue ?? constraint.pixelsHigh
        let channels: Int
        switch format {
        case "rgba", "bgra", "argb":
            channels = 4
        case "rgb":
            channels = 3
        case "grayscale", "gray":
            channels = 1
        default:
            channels = 0
        }

        let image: CGImage
        if format == "encoded" || channels == 0 || bytes.count != width * height * channels {
            guard let decoded = UIImage(data: bytes)?.cgImage else {
                if channels == 0 || format == "encoded" {
                    throw NativeMLError("INVALID_INPUT", "Image input '\(name)' could not be decoded (supported: PNG, JPEG, HEIC)")
                }
                throw NativeMLError(
                    "SHAPE_MISMATCH",
                    "Image input '\(name)': expected \(width * height * channels) bytes for \(width)x\(height) \(format) but got \(bytes.count)"
                )
            }
            image = decoded
        } else {
            image = try cgImage(fromRaw: bytes, width: width, height: height, format: format, channels: channels, name: name)
        }
        return try render(image, width: constraint.pixelsWide, height: constraint.pixelsHigh, pixelFormat: constraint.pixelFormatType, name: name)
    }

    private func cgImage(fromRaw bytes: Data, width: Int, height: Int, format: String, channels: Int, name: String) throws -> CGImage {
        let colorSpace = channels == 1 ? CGColorSpaceCreateDeviceGray() : CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: CGBitmapInfo
        switch format {
        case "bgra":
            bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        case "argb":
            bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue)
        case "rgba":
            bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        default:
            bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        }
        guard let provider = CGDataProvider(data: bytes as CFData),
              let image = CGImage(
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bitsPerPixel: channels * 8,
                  bytesPerRow: width * channels,
                  space: colorSpace,
                  bitmapInfo: bitmapInfo,
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              )
        else {
            throw NativeMLError("INVALID_INPUT", "Image input '\(name)': could not interpret \(width)x\(height) \(format) pixels")
        }
        return image
    }

    private func render(_ image: CGImage, width: Int, height: Int, pixelFormat: OSType, name: String) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, pixelFormat, attributes as CFDictionary, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw NativeMLError("INTERNAL", "Could not allocate a \(width)x\(height) pixel buffer for '\(name)' (status \(status))")
        }

        let colorSpace: CGColorSpace
        let bitmapInfo: UInt32
        switch pixelFormat {
        case kCVPixelFormatType_32BGRA:
            colorSpace = CGColorSpaceCreateDeviceRGB()
            bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        case kCVPixelFormatType_32ARGB:
            colorSpace = CGColorSpaceCreateDeviceRGB()
            bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue
        case kCVPixelFormatType_OneComponent8:
            colorSpace = CGColorSpaceCreateDeviceGray()
            bitmapInfo = CGImageAlphaInfo.none.rawValue
        default:
            throw NativeMLError("UNSUPPORTED_INPUT", "Image input '\(name)' uses pixel format \(pixelFormatName(pixelFormat)), which is not supported")
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw NativeMLError("INTERNAL", "Could not create a drawing context for '\(name)'")
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    private func imageDictionary(_ buffer: CVPixelBuffer) -> [String: Any] {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let format = CVPixelBufferGetPixelFormatType(buffer)
        let channels = channelCount(forPixelFormat: format)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        var data = Data(count: width * height * channels)
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            data.withUnsafeMutableBytes { destination in
                guard let destinationBase = destination.baseAddress else { return }
                for row in 0..<height {
                    memcpy(destinationBase + row * width * channels, base + row * bytesPerRow, width * channels)
                }
            }
        }
        return [
            "kind": "image",
            "width": width,
            "height": height,
            "format": pixelFormatName(format),
            "data": FlutterStandardTypedData(bytes: data),
        ]
    }

    // MARK: Outputs

    private func convertOutput(_ provider: MLFeatureProvider) -> ([String: Any], [String: Any]) {
        var output: [String: Any] = [:]
        var shapes: [String: Any] = [:]
        for name in provider.featureNames.sorted() {
            guard let value = provider.featureValue(for: name) else { continue }
            switch value.type {
            case .multiArray:
                if let array = value.multiArrayValue {
                    shapes[name] = array.shape.map { $0.intValue }
                    if let converted = try? typedData(from: array) {
                        output[name] = converted
                    }
                }
            case .string:
                output[name] = value.stringValue
            case .int64:
                output[name] = NSNumber(value: value.int64Value)
            case .double:
                output[name] = value.doubleValue
            case .dictionary:
                var dictionary: [String: Double] = [:]
                for (key, number) in value.dictionaryValue {
                    dictionary["\(key)"] = number.doubleValue
                }
                output[name] = dictionary
            case .image:
                if let buffer = value.imageBufferValue {
                    output[name] = imageDictionary(buffer)
                    shapes[name] = [CVPixelBufferGetHeight(buffer), CVPixelBufferGetWidth(buffer), channelCount(forPixelFormat: CVPixelBufferGetPixelFormatType(buffer))]
                }
            case .sequence:
                if let sequence = value.sequenceValue {
                    output[name] = sequence.type == .string ? sequence.stringValues : sequence.int64Values
                    shapes[name] = [sequence.type == .string ? sequence.stringValues.count : sequence.int64Values.count]
                }
            case .invalid:
                continue
            @unknown default:
                continue
            }
        }
        return (output, shapes)
    }

    // MARK: - Streaming

    /// Main thread. Registers the per-model EventChannel and activates the frame queue.
    private func startStream(_ args: [String: Any]) throws -> [String: Any] {
        let model = try requireModel(args)
        guard let registrar = registrar else {
            throw NativeMLError("NOT_ATTACHED", "Plugin is not attached to a Flutter engine")
        }
        let maxQueueSize = max(1, (args["maxQueueSize"] as? NSNumber)?.intValue ?? FlutterNativeMlPlugin.defaultMaxQueueSize)
        let channelName = FlutterNativeMlPlugin.streamChannelPrefix + model.id

        if model.eventChannel == nil {
            let channel = FlutterEventChannel(name: channelName, binaryMessenger: registrar.messenger())
            let handler = ModelStreamHandler(
                onListen: { [weak model] sink in model?.sink = sink },
                onCancel: { [weak model] in model?.sink = nil }
            )
            channel.setStreamHandler(handler)
            model.eventChannel = channel
            model.streamHandler = handler
        }
        stateQueue.sync {
            model.maxQueueSize = maxQueueSize
            model.streamActive = true
            model.droppedFrames = 0
            model.pending.removeAll()
        }
        return ["channel": channelName, "maxQueueSize": maxQueueSize]
    }

    /// Main thread. Queues a frame, dropping the oldest when the queue is full.
    private func streamInput(_ args: [String: Any]) throws -> [String: Any] {
        let model = try requireModel(args)
        guard let input = args["input"] as? [String: Any] else {
            throw NativeMLError("INVALID_ARGS", "'input' must be a map of input name to data")
        }
        var frameId: Int64 = 0
        var queueSize = 0
        var dropped: Int64 = 0
        try stateQueue.sync {
            guard model.streamActive else {
                throw NativeMLError("STREAM_NOT_ACTIVE", "Call startStream() before pushing stream input")
            }
            while model.pending.count >= model.maxQueueSize {
                model.pending.removeFirst()
                model.droppedFrames += 1
            }
            frameId = model.nextFrameId
            model.nextFrameId += 1
            model.pending.append((frameId: frameId, input: input))
            queueSize = model.pending.count
            dropped = model.droppedFrames
        }
        scheduleProcessing(model)
        return ["frameId": frameId, "queueSize": queueSize, "droppedFrames": dropped]
    }

    private func scheduleProcessing(_ model: LoadedModel) {
        let shouldStart: Bool = stateQueue.sync {
            if model.processing { return false }
            model.processing = true
            return true
        }
        guard shouldStart else { return }
        model.queue.async { self.processQueue(model) }
    }

    /// Model queue. Drains pending frames, emitting one event per frame.
    private func processQueue(_ model: LoadedModel) {
        while true {
            let next: (frameId: Int64, input: [String: Any])? = stateQueue.sync {
                guard model.streamActive, !model.disposed, !model.pending.isEmpty else { return nil }
                return model.pending.removeFirst()
            }
            guard let frame = next else { break }
            let dropped = stateQueue.sync { model.droppedFrames }
            do {
                var payload = try run(model: model, input: frame.input)
                payload["frameId"] = frame.frameId
                payload["droppedFrames"] = dropped
                emit(model) { sink in sink(payload) }
            } catch {
                let flutterError = self.flutterError(from: error)
                emit(model) { sink in
                    sink(FlutterError(code: flutterError.code, message: flutterError.message, details: ["frameId": frame.frameId]))
                }
            }
        }
        let more: Bool = stateQueue.sync {
            model.processing = false
            return model.streamActive && !model.pending.isEmpty
        }
        if more {
            scheduleProcessing(model)
        }
    }

    private func emit(_ model: LoadedModel, _ action: @escaping (FlutterEventSink) -> Void) {
        DispatchQueue.main.async {
            if let sink = model.sink {
                action(sink)
            }
        }
    }

    /// Main thread.
    private func stopStream(_ args: [String: Any]) throws -> Any? {
        guard let modelId = args["modelId"] as? String else {
            throw NativeMLError("INVALID_ARGS", "'modelId' is required")
        }
        guard let model = stateQueue.sync(execute: { models[modelId] }) else {
            return nil
        }
        let dropped: Int64 = stateQueue.sync {
            model.streamActive = false
            let total = model.droppedFrames + Int64(model.pending.count)
            model.pending.removeAll()
            return total
        }
        if let sink = model.sink {
            model.sink = nil
            sink(FlutterEndOfEventStream)
        }
        return ["droppedFrames": dropped]
    }

    // MARK: - Disposal

    /// Main thread.
    private func dispose(_ args: [String: Any]) throws -> Any? {
        guard let modelId = args["modelId"] as? String else {
            throw NativeMLError("INVALID_ARGS", "'modelId' is required")
        }
        guard let model = stateQueue.sync(execute: { models.removeValue(forKey: modelId) }) else {
            return nil // Disposing twice is not an error.
        }
        teardown(model)
        return nil
    }

    private func teardown(_ model: LoadedModel) {
        stateQueue.sync {
            model.disposed = true
            model.streamActive = false
            model.pending.removeAll()
        }
        if let sink = model.sink {
            model.sink = nil
            sink(FlutterEndOfEventStream)
        }
        model.eventChannel?.setStreamHandler(nil)
        model.eventChannel = nil
        model.streamHandler = nil
    }

    private func disposeAll() {
        let snapshot: [LoadedModel] = stateQueue.sync {
            let all = Array(models.values)
            models.removeAll()
            return all
        }
        let work = { snapshot.forEach { self.teardown($0) } }
        if Thread.isMainThread {
            work()
        } else {
            DispatchQueue.main.async(execute: work)
        }
    }

    // MARK: - Capabilities

    private func deviceCapabilities() -> [String: Any] {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machine = withUnsafePointer(to: &systemInfo.machine) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) { String(cString: $0) }
        }
        #if targetEnvironment(simulator)
            let isSimulator = true
        #else
            let isSimulator = false
        #endif
        // Core ML does not expose whether a Neural Engine is present. Every iPhone/iPad that can
        // run iOS 13+ on an A12 or newer chip has one; the simulator never does.
        let hasNeuralEngine = !isSimulator
        var units = ["all", "cpuOnly"]
        if !isSimulator {
            units.append(contentsOf: ["cpuAndGpu", "cpuAndNeuralEngine"])
        }
        return [
            "platform": "ios",
            "osVersion": UIDevice.current.systemVersion,
            "device": machine,
            "cpuCount": ProcessInfo.processInfo.activeProcessorCount,
            "runtimeVersion": "Core ML (iOS \(UIDevice.current.systemVersion))",
            "gpuAvailable": !isSimulator,
            "nnapiAvailable": false,
            "neuralEngineAvailable": hasNeuralEngine,
            "neuralEngineHeuristic": true,
            "isEmulator": isSimulator,
            "supportedComputeUnits": units,
        ]
    }

    // MARK: - Half precision helpers (portable; `Float16` is unavailable on Intel simulators)

    private func halfToFloat(_ half: UInt16) -> Float {
        let sign = UInt32(half & 0x8000) << 16
        let exponent = Int((half >> 10) & 0x1F)
        let mantissa = UInt32(half & 0x3FF)
        if exponent == 0 {
            if mantissa == 0 {
                return Float(bitPattern: sign)
            }
            var m = mantissa
            var e: UInt32 = 113
            while (m & 0x400) == 0 {
                m <<= 1
                e -= 1
            }
            m &= 0x3FF
            return Float(bitPattern: sign | (e << 23) | (m << 13))
        }
        if exponent == 31 {
            return Float(bitPattern: sign | 0x7F80_0000 | (mantissa << 13))
        }
        return Float(bitPattern: sign | (UInt32(exponent + 112) << 23) | (mantissa << 13))
    }

    private func floatToHalf(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exponent = Int32((bits >> 23) & 0xFF) - 127 + 15
        var mantissa = bits & 0x7F_FFFF
        if (bits & 0x7F80_0000) == 0x7F80_0000 {
            return sign | 0x7C00 | (mantissa != 0 ? 0x200 : 0)
        }
        if exponent >= 31 {
            return sign | 0x7C00
        }
        if exponent <= 0 {
            if exponent < -10 {
                return sign
            }
            mantissa |= 0x80_0000
            let shift = UInt32(14 - exponent)
            var half = UInt16(mantissa >> shift)
            let remainder = mantissa & ((1 << shift) - 1)
            let halfway = UInt32(1) << (shift - 1)
            if remainder > halfway || (remainder == halfway && (half & 1) == 1) {
                half += 1
            }
            return sign | half
        }
        var half = sign | UInt16(exponent << 10) | UInt16(mantissa >> 13)
        let remainder = mantissa & 0x1FFF
        if remainder > 0x1000 || (remainder == 0x1000 && (half & 1) == 1) {
            half += 1
        }
        return half
    }
}
