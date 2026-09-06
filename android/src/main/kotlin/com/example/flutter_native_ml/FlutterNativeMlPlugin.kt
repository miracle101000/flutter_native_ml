package com.example.flutter_native_ml

import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.TensorFlowLite
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/** An error raised by the native layer. Surfaces in Dart as a `PlatformException` with [code]. */
internal class NativeMlException(
    val code: String,
    message: String,
    val details: Any? = null
) : Exception(message)

/**
 * Android implementation of `flutter_native_ml`, backed by LiteRT (TensorFlow Lite).
 *
 * Threading model:
 *  * Method calls arrive on the platform (main) thread.
 *  * Every loaded model owns a single-threaded executor. All interpreter access for that model
 *    (loading, inference, streaming, disposal) happens on that thread. This is required by the GPU
 *    delegate, whose OpenGL context is bound to the thread that created it.
 *  * Results are always delivered back to Flutter on the main thread.
 */
class FlutterNativeMlPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {

    private companion object {
        const val TAG = "FlutterNativeML"
        const val METHOD_CHANNEL = "flutter_native_ml"
        const val STREAM_CHANNEL_PREFIX = "flutter_native_ml_stream/"
        const val DEFAULT_MAX_QUEUE_SIZE = 2
        const val NNAPI_MIN_API = Build.VERSION_CODES.O_MR1
    }

    private enum class Accelerator(val label: String) { GPU("GPU"), NNAPI("NNAPI"), CPU("CPU") }

    private class QueuedFrame(val frameId: Long, val input: Map<String, Any?>)

    private class ModelHolder(val id: String, val executor: ExecutorService) {
        lateinit var interpreter: Interpreter

        fun hasInterpreter(): Boolean = this::interpreter.isInitialized
        var delegates: List<Any> = emptyList()
        var accelerator: String = "CPU"
        var threads: Int = 1

        /** Keeps the (memory-mapped) model bytes alive for as long as the interpreter exists. */
        var modelBuffer: ByteBuffer? = null

        @Volatile
        var disposed = false

        // ---- Streaming state (guarded by [streamLock]) ----
        val streamLock = Any()
        var eventChannel: EventChannel? = null

        @Volatile
        var sink: EventChannel.EventSink? = null
        val queue = ArrayDeque<QueuedFrame>()
        var maxQueueSize = DEFAULT_MAX_QUEUE_SIZE
        var streamActive = false
        var droppedFrames = 0L
        val frameCounter = AtomicLong(0)
        val processing = AtomicBoolean(false)
    }

    private var channel: MethodChannel? = null
    private var binding: FlutterPlugin.FlutterPluginBinding? = null
    private val mainHandler: Handler by lazy { Handler(Looper.getMainLooper()) }
    private val models = ConcurrentHashMap<String, ModelHolder>()
    private val sharedExecutor: ExecutorService by lazy {
        Executors.newCachedThreadPool { r -> Thread(r, "flutter_native_ml-shared").apply { isDaemon = true } }
    }

    // ---------------------------------------------------------------------------------------------
    // FlutterPlugin lifecycle
    // ---------------------------------------------------------------------------------------------

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        this.binding = binding
        channel = MethodChannel(binding.binaryMessenger, METHOD_CHANNEL).also {
            it.setMethodCallHandler(this)
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        disposeAll()
        channel?.setMethodCallHandler(null)
        channel = null
        this.binding = null
    }

    // ---------------------------------------------------------------------------------------------
    // Method channel dispatch
    // ---------------------------------------------------------------------------------------------

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        try {
            when (call.method) {
                "getPlatformVersion" -> result.success("Android ${Build.VERSION.RELEASE}")
                "getDeviceCapabilities" -> runShared(result) { deviceCapabilities() }
                "loadModel" -> loadModel(call, result)
                "getSignature" -> runOnModel(call, result) { signature(it) }
                "run" -> runOnModel(call, result) { holder ->
                    val input = call.argument<Map<String, Any?>>("input")
                        ?: throw NativeMlException("INVALID_ARGS", "'input' must be a map of input name to data")
                    execute(holder, input, call.argument<String>("signatureKey"))
                }
                "startStream" -> result.success(startStream(call))
                "streamInput" -> result.success(streamInput(call))
                "stopStream" -> result.success(stopStream(call))
                "dispose" -> dispose(call, result)
                "disposeAll" -> {
                    disposeAll()
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        } catch (e: NativeMlException) {
            result.error(e.code, e.message, e.details)
        } catch (t: Throwable) {
            result.error("NATIVE_ERROR", t.message ?: t.toString(), t.stackTraceToString())
        }
    }

    /** Runs [block] on the current thread and marshals its outcome to Flutter on the main thread. */
    private fun reply(result: MethodChannel.Result, block: () -> Any?) {
        val outcome: Any? = try {
            block()
        } catch (e: NativeMlException) {
            mainHandler.post { result.error(e.code, e.message, e.details) }
            return
        } catch (t: Throwable) {
            Log.e(TAG, "Native error", t)
            mainHandler.post { result.error("NATIVE_ERROR", t.message ?: t.toString(), t.stackTraceToString()) }
            return
        }
        mainHandler.post { result.success(outcome) }
    }

    private fun runShared(result: MethodChannel.Result, block: () -> Any?) {
        sharedExecutor.execute { reply(result, block) }
    }

    private fun runOnModel(call: MethodCall, result: MethodChannel.Result, block: (ModelHolder) -> Any?) {
        val holder = requireModel(call)
        try {
            holder.executor.execute { reply(result) { block(holder) } }
        } catch (e: RejectedExecutionException) {
            throw NativeMlException("MODEL_DISPOSED", "Model ${holder.id} has been disposed")
        }
    }

    private fun requireModel(call: MethodCall): ModelHolder {
        val modelId = call.argument<String>("modelId")
            ?: throw NativeMlException("INVALID_ARGS", "'modelId' is required")
        val holder = models[modelId]
            ?: throw NativeMlException("MODEL_NOT_FOUND", "No loaded model with id '$modelId'")
        if (holder.disposed) throw NativeMlException("MODEL_DISPOSED", "Model $modelId has been disposed")
        return holder
    }

    // ---------------------------------------------------------------------------------------------
    // Loading
    // ---------------------------------------------------------------------------------------------

    private fun loadModel(call: MethodCall, result: MethodChannel.Result) {
        val binding = this.binding
            ?: throw NativeMlException("NOT_ATTACHED", "Plugin is not attached to a Flutter engine")
        val assetPath = call.argument<String>("assetPath")
        val filePath = call.argument<String>("filePath")
        if (assetPath.isNullOrBlank() && filePath.isNullOrBlank()) {
            throw NativeMlException("INVALID_ARGS", "Provide either 'assetPath' or 'filePath'")
        }
        val computeUnits = call.argument<String>("computeUnits") ?: "all"
        val numThreads = call.argument<Number>("numThreads")?.toInt()
        val allowFp16 = call.argument<Boolean>("allowFp16") ?: false

        val id = UUID.randomUUID().toString()
        val executor = Executors.newSingleThreadExecutor { r ->
            Thread(r, "flutter_native_ml-${id.take(8)}").apply { isDaemon = true }
        }
        val holder = ModelHolder(id, executor)
        executor.execute {
            reply(result) {
                try {
                    holder.modelBuffer = if (!assetPath.isNullOrBlank()) {
                        loadAssetBuffer(binding, assetPath)
                    } else {
                        loadFileBuffer(filePath!!)
                    }
                    initialiseInterpreter(holder, computeUnits, numThreads, allowFp16)
                    models[id] = holder
                    mapOf(
                        "modelId" to id,
                        "acceleratorUsed" to holder.accelerator,
                        "signature" to signature(holder)
                    )
                } catch (t: Throwable) {
                    holder.disposed = true
                    executor.shutdown()
                    throw t
                }
            }
        }
    }

    private fun loadAssetBuffer(binding: FlutterPlugin.FlutterPluginBinding, assetPath: String): ByteBuffer {
        // Absolute paths are treated as files even when passed as an asset path.
        if (assetPath.startsWith("/") && File(assetPath).isFile) return loadFileBuffer(assetPath)

        val key = binding.flutterAssets.getAssetFilePathByName(assetPath)
        val assets = binding.applicationContext.assets

        // Fast path: memory-map the asset directly out of the APK (works for uncompressed assets).
        try {
            return assets.openFd(key).use { afd ->
                FileInputStream(afd.fileDescriptor).use { stream ->
                    stream.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
                }
            }
        } catch (e: IOException) {
            Log.d(TAG, "Asset '$key' cannot be memory-mapped (${e.message}); copying it instead")
        }

        try {
            return assets.open(key).use { stream ->
                val bytes = stream.readBytes()
                ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
                    put(bytes)
                    rewind()
                }
            }
        } catch (e: IOException) {
            throw NativeMlException(
                "MODEL_NOT_FOUND",
                "Asset '$assetPath' (resolved to '$key') could not be opened: ${e.message}. " +
                    "Make sure the file is listed under `flutter: assets:` in pubspec.yaml."
            )
        }
    }

    private fun loadFileBuffer(path: String): ByteBuffer {
        val file = File(path.removePrefix("file://"))
        if (!file.isFile) throw NativeMlException("MODEL_NOT_FOUND", "No model file at '${file.path}'")
        try {
            return FileInputStream(file).use { stream ->
                stream.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
            }
        } catch (e: IOException) {
            throw NativeMlException("LOAD_FAILED", "Could not read '${file.path}': ${e.message}")
        }
    }

    private fun initialiseInterpreter(holder: ModelHolder, computeUnits: String, numThreads: Int?, allowFp16: Boolean) {
        val buffer = holder.modelBuffer ?: throw NativeMlException("LOAD_FAILED", "Model buffer missing")
        val threads = (numThreads ?: defaultThreadCount()).coerceIn(1, 32)
        val plan = when (computeUnits) {
            "cpuOnly" -> listOf(Accelerator.CPU)
            "cpuAndGpu" -> listOf(Accelerator.GPU, Accelerator.CPU)
            "cpuAndNeuralEngine" -> listOf(Accelerator.NNAPI, Accelerator.CPU)
            else -> listOf(Accelerator.GPU, Accelerator.CPU) // "all"
        }

        var lastError: Throwable? = null
        for (accelerator in plan) {
            val delegates = mutableListOf<Any>()
            try {
                val options = Interpreter.Options().setNumThreads(threads)
                when (accelerator) {
                    Accelerator.GPU -> {
                        val delegate = createGpuDelegate(allowFp16) ?: continue
                        delegates += delegate
                        options.addDelegate(delegate)
                    }
                    Accelerator.NNAPI -> {
                        val delegate = createNnApiDelegate(allowFp16) ?: continue
                        delegates += delegate
                        options.addDelegate(delegate)
                    }
                    Accelerator.CPU -> options.setUseXNNPACK(true)
                }

                val interpreter = Interpreter(buffer, options)
                interpreter.allocateTensors()

                if (accelerator == Accelerator.NNAPI && (delegates.first() as NnApiDelegate).hasErrors()) {
                    interpreter.close()
                    closeDelegates(delegates)
                    Log.w(TAG, "NNAPI reported errors while preparing the model; falling back")
                    continue
                }

                holder.interpreter = interpreter
                holder.delegates = delegates
                holder.threads = threads
                holder.accelerator = when (accelerator) {
                    Accelerator.CPU -> "CPU (XNNPACK, $threads threads)"
                    else -> accelerator.label
                }
                return
            } catch (t: Throwable) {
                lastError = t
                closeDelegates(delegates)
                Log.w(TAG, "Could not initialise the model with ${accelerator.label}; trying the next option", t)
            }
        }
        throw NativeMlException(
            "LOAD_FAILED",
            "Failed to load model: ${lastError?.message ?: "no compute unit could be initialised"}",
            lastError?.stackTraceToString()
        )
    }

    private fun defaultThreadCount(): Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)

    private fun createGpuDelegate(allowFp16: Boolean): GpuDelegate? = try {
        CompatibilityList().use { compatibility ->
            if (!compatibility.isDelegateSupportedOnThisDevice) {
                null
            } else {
                val options = compatibility.bestOptionsForThisDevice
                options.setQuantizedModelsAllowed(true)
                if (allowFp16) options.setPrecisionLossAllowed(true)
                GpuDelegate(options)
            }
        }
    } catch (t: Throwable) {
        Log.w(TAG, "GPU delegate unavailable: ${t.message}")
        null
    }

    private fun createNnApiDelegate(allowFp16: Boolean): NnApiDelegate? {
        if (Build.VERSION.SDK_INT < NNAPI_MIN_API) return null
        return try {
            val options = NnApiDelegate.Options()
                .setAllowFp16(allowFp16)
                .setUseNnapiCpu(false)
                .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
            NnApiDelegate(options)
        } catch (t: Throwable) {
            Log.w(TAG, "NNAPI delegate unavailable: ${t.message}")
            null
        }
    }

    private fun closeDelegates(delegates: List<Any>) {
        for (delegate in delegates) {
            try {
                when (delegate) {
                    is GpuDelegate -> delegate.close()
                    is NnApiDelegate -> delegate.close()
                    is AutoCloseable -> delegate.close()
                }
            } catch (t: Throwable) {
                Log.w(TAG, "Failed to close delegate", t)
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Signature
    // ---------------------------------------------------------------------------------------------

    private fun signature(holder: ModelHolder): Map<String, Any?> {
        val interpreter = holder.interpreter
        val inputs = (0 until interpreter.inputTensorCount).map { tensorInfo(interpreter.getInputTensor(it), it) }
        val outputs = (0 until interpreter.outputTensorCount).map { tensorInfo(interpreter.getOutputTensor(it), it) }

        val signatureKeys = try {
            interpreter.signatureKeys.toList()
        } catch (ignored: Throwable) {
            emptyList()
        }
        val signatures = LinkedHashMap<String, Any?>()
        for (key in signatureKeys) {
            try {
                val sigInputs = LinkedHashMap<String, String>()
                for (name in interpreter.getSignatureInputs(key)) {
                    sigInputs[name] = interpreter.getInputTensorFromSignature(name, key).name()
                }
                val sigOutputs = LinkedHashMap<String, String>()
                for (name in interpreter.getSignatureOutputs(key)) {
                    sigOutputs[name] = interpreter.getOutputTensorFromSignature(name, key).name()
                }
                signatures[key] = mapOf("inputs" to sigInputs, "outputs" to sigOutputs)
            } catch (t: Throwable) {
                Log.w(TAG, "Could not read signature '$key'", t)
            }
        }

        return mapOf(
            "inputs" to inputs,
            "outputs" to outputs,
            "signatureKeys" to signatureKeys,
            "signatures" to signatures,
            "metadata" to mapOf(
                "runtime" to runtimeVersion(),
                "acceleratorUsed" to holder.accelerator
            )
        )
    }

    private fun tensorInfo(tensor: Tensor, index: Int): Map<String, Any?> {
        val currentShape = tensor.shape()
        val declaredShape = try {
            tensor.shapeSignature() ?: currentShape
        } catch (ignored: Throwable) {
            currentShape
        }
        val quantization: Tensor.QuantizationParams? = try {
            tensor.quantizationParams()
        } catch (ignored: Throwable) {
            null
        }
        val quantized = quantization != null && quantization.scale != 0f
        return mapOf(
            "index" to index,
            "name" to tensor.name(),
            "shape" to declaredShape.toList(),
            "currentShape" to currentShape.toList(),
            "dataType" to dataTypeName(tensor.dataType()),
            "quantizationScale" to if (quantized) quantization.scale.toDouble() else null,
            "quantizationZeroPoint" to if (quantized) quantization.zeroPoint else null
        )
    }

    private fun dataTypeName(type: DataType): String = when (type) {
        DataType.FLOAT32 -> "float32"
        DataType.INT32 -> "int32"
        DataType.UINT8 -> "uint8"
        DataType.INT64 -> "int64"
        DataType.STRING -> "string"
        DataType.BOOL -> "bool"
        DataType.INT16 -> "int16"
        DataType.INT8 -> "int8"
    }

    private fun runtimeVersion(): String? = try {
        "LiteRT ${TensorFlowLite.runtimeVersion()}"
    } catch (ignored: Throwable) {
        null
    }

    // ---------------------------------------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------------------------------------

    /** Runs one inference. Must be called on the model's executor thread. */
    private fun execute(holder: ModelHolder, inputMap: Map<String, Any?>, signatureKey: String?): MutableMap<String, Any?> {
        val interpreter = holder.interpreter
        val inputCount = interpreter.inputTensorCount
        val pending = arrayOfNulls<Any>(inputCount)
        var needsAllocation = false

        for ((name, raw) in inputMap) {
            val index = resolveInputIndex(interpreter, name, signatureKey)
            val tensor = interpreter.getInputTensor(index)
            val (data, explicitShape) = unpackInput(raw, name)
            val count = elementCount(data, name)
            val newShape = resolveShape(tensor, name, explicitShape, count)
            if (newShape != null) {
                interpreter.resizeInput(index, newShape)
                needsAllocation = true
            }
            pending[index] = data
        }

        val missing = (0 until inputCount).filter { pending[it] == null }.map { interpreter.getInputTensor(it).name() }
        if (missing.isNotEmpty()) {
            throw NativeMlException(
                "MISSING_INPUT",
                "Missing input(s) $missing. Provided: ${inputMap.keys.toList()}"
            )
        }
        if (needsAllocation) interpreter.allocateTensors()

        val inputs = Array<Any>(inputCount) { i -> toTensorData(interpreter.getInputTensor(i), pending[i]!!) }

        val start = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(inputs, HashMap<Int, Any>())
        val wallClockMicros = (System.nanoTime() - start) / 1_000.0
        val nativeMicros = try {
            interpreter.lastNativeInferenceDurationNanoseconds?.let { it / 1_000.0 }
        } catch (ignored: Throwable) {
            null
        }

        val output = LinkedHashMap<String, Any?>()
        val shapes = LinkedHashMap<String, Any?>()
        for (i in 0 until interpreter.outputTensorCount) {
            val tensor = interpreter.getOutputTensor(i)
            output[tensor.name()] = readTensor(tensor)
            shapes[tensor.name()] = tensor.shape().toList()
        }

        return mutableMapOf(
            "output" to output,
            "outputShapes" to shapes,
            "inferenceTime" to wallClockMicros,
            "nativeInferenceTime" to nativeMicros,
            "acceleratorUsed" to holder.accelerator
        )
    }

    private fun resolveInputIndex(interpreter: Interpreter, name: String, signatureKey: String?): Int {
        try {
            return interpreter.getInputIndex(name)
        } catch (ignored: IllegalArgumentException) {
            // Not a raw tensor name; try SignatureDef names below.
        }
        val keys: List<String> = try {
            if (signatureKey != null) listOf(signatureKey) else interpreter.signatureKeys.toList()
        } catch (ignored: Throwable) {
            emptyList()
        }
        for (key in keys) {
            try {
                if (interpreter.getSignatureInputs(key).contains(name)) {
                    val tensorName = interpreter.getInputTensorFromSignature(name, key).name()
                    return interpreter.getInputIndex(tensorName)
                }
            } catch (ignored: Throwable) {
                // ignore and continue
            }
        }
        name.toIntOrNull()?.let { if (it in 0 until interpreter.inputTensorCount) return it }

        val available = (0 until interpreter.inputTensorCount).map { interpreter.getInputTensor(it).name() }
        throw NativeMlException("INPUT_MISMATCH", "No input named '$name'. Available inputs: $available")
    }

    private fun unpackInput(raw: Any?, name: String): Pair<Any, IntArray?> {
        if (raw == null) throw NativeMlException("INVALID_INPUT", "Input '$name' is null")
        if (raw is Map<*, *> && raw.containsKey("data")) {
            val data = raw["data"] ?: throw NativeMlException("INVALID_INPUT", "Input '$name' has no 'data'")
            val shape = (raw["shape"] as? List<*>)?.map { (it as Number).toInt() }?.toIntArray()
            return data to shape
        }
        return raw to null
    }

    private fun elementCount(data: Any, name: String): Int = when (data) {
        is FloatArray -> data.size
        is DoubleArray -> data.size
        is IntArray -> data.size
        is LongArray -> data.size
        is ByteArray -> data.size
        is List<*> -> data.size
        is Number, is Boolean, is String -> 1
        else -> throw NativeMlException(
            "UNSUPPORTED_INPUT",
            "Input '$name' has unsupported type ${data::class.java.simpleName}. " +
                "Use a List<num>, List<bool>, List<String> or a typed list (Float32List, Int32List, ...)."
        )
    }

    /**
     * Returns the shape the input tensor must be resized to, or null if it already fits.
     * Handles explicit shapes and models with a single dynamic (-1) dimension.
     */
    private fun resolveShape(tensor: Tensor, name: String, explicit: IntArray?, count: Int): IntArray? {
        val current = tensor.shape()
        if (explicit != null) {
            if (explicit.any { it <= 0 }) {
                throw NativeMlException("INVALID_SHAPE", "Shape for '$name' must have positive dimensions: ${explicit.toList()}")
            }
            val implied = explicit.fold(1L) { acc, dim -> acc * dim }
            if (implied != count.toLong()) {
                throw NativeMlException("SHAPE_MISMATCH", "Shape ${explicit.toList()} for '$name' implies $implied elements but $count were provided")
            }
            return if (explicit.contentEquals(current)) null else explicit
        }

        val currentCount = current.fold(1L) { acc, dim -> acc * dim }
        if (count.toLong() == currentCount) return null

        val declared = try {
            tensor.shapeSignature() ?: current
        } catch (ignored: Throwable) {
            current
        }
        val dynamicDims = declared.indices.filter { declared[it] < 0 }
        if (dynamicDims.size == 1) {
            val fixed = declared.indices.filter { it != dynamicDims[0] }.fold(1L) { acc, i -> acc * declared[i] }
            if (fixed > 0 && count % fixed == 0L) {
                return declared.copyOf().also { it[dynamicDims[0]] = (count / fixed).toInt() }
            }
        }
        val hint = if (dynamicDims.isNotEmpty()) {
            " The input has dynamic dimensions; pass {'data': ..., 'shape': [...]} to resize it."
        } else {
            ""
        }
        throw NativeMlException(
            "SHAPE_MISMATCH",
            "Input '$name' expects $currentCount elements (shape ${current.toList()}) but $count were provided.$hint"
        )
    }

    private fun toTensorData(tensor: Tensor, data: Any): Any {
        val name = tensor.name()
        if (tensor.dataType() == DataType.STRING) return buildStringArray(tensor.shape(), data, name)

        val expected = tensor.numElements()
        val count = elementCount(data, name)
        if (count != expected) {
            throw NativeMlException("SHAPE_MISMATCH", "Input '$name' expects $expected elements but $count were provided")
        }
        val buffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        when (tensor.dataType()) {
            DataType.FLOAT32 -> {
                val fb = buffer.asFloatBuffer()
                if (data is FloatArray) fb.put(data) else forEachNumber(data, name) { fb.put(it.toFloat()) }
            }
            DataType.INT32 -> {
                val ib = buffer.asIntBuffer()
                if (data is IntArray) ib.put(data) else forEachNumber(data, name) { ib.put(it.toInt()) }
            }
            DataType.INT64 -> {
                val lb = buffer.asLongBuffer()
                if (data is LongArray) lb.put(data) else forEachNumber(data, name) { lb.put(it.toLong()) }
            }
            DataType.UINT8, DataType.INT8 -> {
                if (data is ByteArray) buffer.put(data) else forEachNumber(data, name) { buffer.put(it.toInt().toByte()) }
            }
            DataType.INT16 -> {
                val sb = buffer.asShortBuffer()
                forEachNumber(data, name) { sb.put(it.toInt().toShort()) }
            }
            DataType.BOOL -> forEachNumber(data, name) { buffer.put(if (it.toDouble() != 0.0) 1 else 0) }
            else -> throw NativeMlException("UNSUPPORTED_DTYPE", "Input '$name' has unsupported type ${tensor.dataType()}")
        }
        buffer.rewind()
        return buffer
    }

    private inline fun forEachNumber(data: Any, name: String, action: (Number) -> Unit) {
        when (data) {
            is FloatArray -> for (v in data) action(v)
            is DoubleArray -> for (v in data) action(v)
            is IntArray -> for (v in data) action(v)
            is LongArray -> for (v in data) action(v)
            is ByteArray -> for (v in data) action(v.toInt() and 0xFF) // Uint8List semantics
            is Number -> action(data)
            is Boolean -> action(if (data) 1 else 0)
            is List<*> -> for (v in data) {
                when (v) {
                    is Number -> action(v)
                    is Boolean -> action(if (v) 1 else 0)
                    else -> throw NativeMlException(
                        "UNSUPPORTED_INPUT",
                        "Input '$name' contains a non-numeric element: ${v?.let { it::class.java.simpleName } ?: "null"}"
                    )
                }
            }
            else -> throw NativeMlException("UNSUPPORTED_INPUT", "Input '$name' has unsupported type ${data::class.java.simpleName}")
        }
    }

    private fun buildStringArray(shape: IntArray, data: Any, name: String): Any {
        val flat: List<String> = when (data) {
            is String -> listOf(data)
            is List<*> -> data.map { it?.toString() ?: "" }
            else -> throw NativeMlException("UNSUPPORTED_INPUT", "String input '$name' must be a String or List<String>")
        }
        val dims = if (shape.isEmpty()) intArrayOf(1) else shape
        val expected = dims.fold(1) { acc, dim -> acc * dim }
        if (flat.size != expected) {
            throw NativeMlException("SHAPE_MISMATCH", "Input '$name' expects $expected strings but ${flat.size} were provided")
        }
        var cursor = 0
        fun build(depth: Int): Any {
            if (depth == dims.size - 1) return Array(dims[depth]) { flat[cursor++] }
            val children = (0 until dims[depth]).map { build(depth + 1) }
            val array = java.lang.reflect.Array.newInstance(children[0].javaClass, dims[depth])
            children.forEachIndexed { i, child -> java.lang.reflect.Array.set(array, i, child) }
            return array
        }
        return build(0)
    }

    /** Copies an output tensor into a Dart-friendly value (typed lists where possible). */
    private fun readTensor(tensor: Tensor): Any? {
        val type = tensor.dataType()
        val source: ByteBuffer = try {
            tensor.asReadOnlyBuffer()
        } catch (t: Throwable) {
            throw NativeMlException("OUTPUT_READ_FAILED", "Could not read output '${tensor.name()}': ${t.message}")
        }
        source.order(ByteOrder.nativeOrder())
        source.rewind()
        // Derive the element count from the live byte size: it is always fresh, even for
        // outputs whose shape only becomes known after running the model.
        val count = if (type == DataType.STRING) 0 else source.remaining() / type.byteSize()
        return when (type) {
            DataType.FLOAT32 -> FloatArray(count).also { source.asFloatBuffer().get(it) }
            DataType.INT32 -> IntArray(count).also { source.asIntBuffer().get(it) }
            DataType.INT64 -> LongArray(count).also { source.asLongBuffer().get(it) }
            DataType.UINT8 -> ByteArray(count).also { source.get(it) }
            DataType.INT8 -> IntArray(count) { source.get(it).toInt() }
            DataType.INT16 -> {
                val shorts = source.asShortBuffer()
                IntArray(count) { shorts.get(it).toInt() }
            }
            DataType.BOOL -> List(count) { source.get(it) != 0.toByte() }
            DataType.STRING -> decodeStrings(source)
            else -> ByteArray(source.remaining()).also { source.get(it) }
        }
    }

    /** Decodes LiteRT's packed string tensor format: int32 count, int32 offsets[count + 1], bytes. */
    private fun decodeStrings(buffer: ByteBuffer): List<String> {
        val little = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        if (little.remaining() < 4) return emptyList()
        val count = little.getInt(0)
        if (count < 0 || 4 + (count + 1) * 4 > little.limit()) return emptyList()
        val offsets = IntArray(count + 1) { little.getInt(4 + it * 4) }
        return List(count) { i ->
            val start = offsets[i].coerceIn(0, little.limit())
            val end = offsets[i + 1].coerceIn(start, little.limit())
            val bytes = ByteArray(end - start)
            little.position(start)
            little.get(bytes)
            String(bytes, Charsets.UTF_8)
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Streaming
    // ---------------------------------------------------------------------------------------------

    /** Called on the main thread. Registers the per-model EventChannel and activates the queue. */
    private fun startStream(call: MethodCall): Map<String, Any?> {
        val holder = requireModel(call)
        val messenger = binding?.binaryMessenger
            ?: throw NativeMlException("NOT_ATTACHED", "Plugin is not attached to a Flutter engine")
        val maxQueueSize = (call.argument<Number>("maxQueueSize")?.toInt() ?: DEFAULT_MAX_QUEUE_SIZE).coerceAtLeast(1)

        synchronized(holder.streamLock) {
            if (holder.eventChannel == null) {
                holder.eventChannel = EventChannel(messenger, STREAM_CHANNEL_PREFIX + holder.id).also {
                    it.setStreamHandler(object : EventChannel.StreamHandler {
                        override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
                            holder.sink = events
                        }

                        override fun onCancel(arguments: Any?) {
                            holder.sink = null
                        }
                    })
                }
            }
            holder.maxQueueSize = maxQueueSize
            holder.streamActive = true
            holder.droppedFrames = 0
            holder.queue.clear()
        }
        return mapOf("channel" to STREAM_CHANNEL_PREFIX + holder.id, "maxQueueSize" to maxQueueSize)
    }

    /** Called on the main thread. Enqueues a frame; drops the oldest one when the queue is full. */
    private fun streamInput(call: MethodCall): Map<String, Any?> {
        val holder = requireModel(call)
        val input = call.argument<Map<String, Any?>>("input")
            ?: throw NativeMlException("INVALID_ARGS", "'input' must be a map of input name to data")
        val frameId: Long
        val queueSize: Int
        val dropped: Long
        synchronized(holder.streamLock) {
            if (!holder.streamActive) {
                throw NativeMlException("STREAM_NOT_ACTIVE", "Call startStream() before pushing stream input")
            }
            while (holder.queue.size >= holder.maxQueueSize) {
                holder.queue.pollFirst()
                holder.droppedFrames++
            }
            frameId = holder.frameCounter.getAndIncrement()
            holder.queue.addLast(QueuedFrame(frameId, input))
            queueSize = holder.queue.size
            dropped = holder.droppedFrames
        }
        scheduleProcessing(holder)
        return mapOf("frameId" to frameId, "queueSize" to queueSize, "droppedFrames" to dropped)
    }

    private fun scheduleProcessing(holder: ModelHolder) {
        if (!holder.processing.compareAndSet(false, true)) return
        try {
            holder.executor.execute { processQueue(holder) }
        } catch (ignored: RejectedExecutionException) {
            holder.processing.set(false)
        }
    }

    /** Drains the frame queue on the model thread, emitting one event per processed frame. */
    private fun processQueue(holder: ModelHolder) {
        try {
            while (true) {
                val frame: QueuedFrame
                val dropped: Long
                synchronized(holder.streamLock) {
                    if (!holder.streamActive || holder.disposed) return
                    frame = holder.queue.pollFirst() ?: return
                    dropped = holder.droppedFrames
                }
                try {
                    val payload = execute(holder, frame.input, null)
                    payload["frameId"] = frame.frameId
                    payload["droppedFrames"] = dropped
                    emit(holder) { it.success(payload) }
                } catch (e: NativeMlException) {
                    emit(holder) { it.error(e.code, e.message, mapOf("frameId" to frame.frameId, "details" to e.details)) }
                } catch (t: Throwable) {
                    Log.e(TAG, "Stream inference failed", t)
                    emit(holder) { it.error("INFERENCE_FAILED", t.message ?: t.toString(), mapOf("frameId" to frame.frameId)) }
                }
            }
        } finally {
            holder.processing.set(false)
            val more = synchronized(holder.streamLock) { holder.streamActive && holder.queue.isNotEmpty() }
            if (more) scheduleProcessing(holder)
        }
    }

    private fun emit(holder: ModelHolder, action: (EventChannel.EventSink) -> Unit) {
        mainHandler.post {
            val sink = holder.sink ?: return@post
            try {
                action(sink)
            } catch (t: Throwable) {
                Log.w(TAG, "Failed to deliver stream event", t)
            }
        }
    }

    /** Called on the main thread. */
    private fun stopStream(call: MethodCall): Any? {
        val modelId = call.argument<String>("modelId")
            ?: throw NativeMlException("INVALID_ARGS", "'modelId' is required")
        val holder = models[modelId] ?: return null
        var dropped: Long
        synchronized(holder.streamLock) {
            holder.streamActive = false
            dropped = holder.droppedFrames + holder.queue.size
            holder.queue.clear()
        }
        holder.sink?.let { sink ->
            holder.sink = null
            try {
                sink.endOfStream()
            } catch (ignored: Throwable) {
            }
        }
        return mapOf("droppedFrames" to dropped)
    }

    // ---------------------------------------------------------------------------------------------
    // Disposal
    // ---------------------------------------------------------------------------------------------

    private fun dispose(call: MethodCall, result: MethodChannel.Result) {
        val modelId = call.argument<String>("modelId")
            ?: throw NativeMlException("INVALID_ARGS", "'modelId' is required")
        val holder = models.remove(modelId)
        if (holder == null) {
            result.success(null) // Already disposed: disposing twice is not an error.
            return
        }
        teardown(holder)
        holder.executor.execute {
            releaseInterpreter(holder)
            mainHandler.post { result.success(null) }
        }
        holder.executor.shutdown()
    }

    /** Main-thread part of disposal: stop streaming and unregister the event channel. */
    private fun teardown(holder: ModelHolder) {
        holder.disposed = true
        synchronized(holder.streamLock) {
            holder.streamActive = false
            holder.queue.clear()
        }
        holder.sink?.let { sink ->
            holder.sink = null
            try {
                sink.endOfStream()
            } catch (ignored: Throwable) {
            }
        }
        holder.eventChannel?.setStreamHandler(null)
        holder.eventChannel = null
    }

    /** Model-thread part of disposal: release native resources. */
    private fun releaseInterpreter(holder: ModelHolder) {
        try {
            if (holder.hasInterpreter()) holder.interpreter.close()
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to close interpreter", t)
        }
        closeDelegates(holder.delegates)
        holder.delegates = emptyList()
        holder.modelBuffer = null
    }

    private fun disposeAll() {
        val snapshot = models.values.toList()
        models.clear()
        for (holder in snapshot) {
            if (Looper.myLooper() == Looper.getMainLooper()) {
                teardown(holder)
            } else {
                mainHandler.post { teardown(holder) }
            }
            try {
                holder.executor.execute { releaseInterpreter(holder) }
            } catch (ignored: RejectedExecutionException) {
            }
            holder.executor.shutdown()
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Capabilities
    // ---------------------------------------------------------------------------------------------

    private fun deviceCapabilities(): Map<String, Any?> {
        val gpu = try {
            CompatibilityList().use { it.isDelegateSupportedOnThisDevice }
        } catch (t: Throwable) {
            Log.w(TAG, "GPU compatibility check failed: ${t.message}")
            false
        }
        val nnapi = Build.VERSION.SDK_INT >= NNAPI_MIN_API
        val units = mutableListOf("all", "cpuOnly")
        if (gpu) units += "cpuAndGpu"
        if (nnapi) units += "cpuAndNeuralEngine"
        return mapOf(
            "platform" to "android",
            "osVersion" to Build.VERSION.RELEASE,
            "apiLevel" to Build.VERSION.SDK_INT,
            "device" to "${Build.MANUFACTURER} ${Build.MODEL}",
            "cpuCount" to Runtime.getRuntime().availableProcessors(),
            "runtimeVersion" to runtimeVersion(),
            "gpuAvailable" to gpu,
            "nnapiAvailable" to nnapi,
            "neuralEngineAvailable" to false,
            "isEmulator" to isEmulator(),
            "supportedComputeUnits" to units
        )
    }

    private fun isEmulator(): Boolean =
        Build.FINGERPRINT.startsWith("generic") ||
            Build.FINGERPRINT.contains("emulator") ||
            Build.MODEL.contains("Emulator") ||
            Build.MODEL.contains("Android SDK built for") ||
            Build.HARDWARE.contains("ranchu") ||
            Build.HARDWARE.contains("goldfish") ||
            Build.PRODUCT.contains("sdk")
}
