package com.example.flutter_native_ml;

import androidx.annotation.NonNull;
import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import com.google.ai.edge.litert.DataType;
import com.google.ai.edge.litert.Interpreter;
import com.google.ai.edge.litert.gpu.CompatibilityList;
import com.google.ai.edge.litert.gpu.GpuDelegate;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.UUID;

class FlutterNativeMlPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {
    private lateinit var channel: MethodChannel
    private lateinit var binding: FlutterPlugin.FlutterPluginBinding
    private val loadedModels = mutableMapOf<String, ModelHolder>()

    data class ModelHolder(
        val interpreter: Interpreter,
        val gpuDelegate: GpuDelegate?
    )

    override fun onAttachedToEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        this.binding = binding
        channel = MethodChannel(binding.binaryMessenger, "flutter_native_ml")
        channel.setMethodCallHandler(this)
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        try {
            when (call.method) {
                "loadModel"    -> handleLoadModel(call, result)
                "getSignature" -> handleGetSignature(call, result)
                "run"          -> handleRun(call, result)
                "dispose"      -> handleDispose(call, result)
                else           -> result.notImplemented()
            }
        } catch (e: Exception) {
            result.error("NATIVE_ERROR", e.message, e.stackTraceToString())
        }
    }

    private fun handleLoadModel(call: MethodCall, result: MethodChannel.Result) {
        val assetPath = call.argument<String>("assetPath")!!
        val computeUnits = call.argument<String>("computeUnits")!!

        val assetManager = binding.applicationContext.assets
        val assetKey = binding.flutterAssets.getAssetFilePathByName(assetPath)
        val afd = assetManager.openFd(assetKey)
        val inputStream = FileInputStream(afd.fileDescriptor)
        val buffer = inputStream.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)

        val options = Interpreter.Options()
        var gpuDelegate: GpuDelegate? = null
        val compatList = CompatibilityList()
        if (computeUnits == "all" && compatList.isDelegateSupportedOnThisDevice) {
            val delegateOptions = compatList.bestOptionsForThisDevice
            gpuDelegate = GpuDelegate(delegateOptions)
            options.addDelegate(gpuDelegate)
        } else {
            options.setNumThreads(4)  // Fallback to multi-threaded CPU
        }

        val interpreter = Interpreter(buffer, options)
        val modelId = UUID.randomUUID().toString()
        loadedModels[modelId] = ModelHolder(interpreter, gpuDelegate)
        result.success(modelId)
    }

    private fun handleGetSignature(call: MethodCall, result: MethodChannel.Result) {
        val modelId = call.argument<String>("modelId")!!
        val holder = loadedModels[modelId]
            ?: return result.error("MODEL_NOT_FOUND", "Interpreter not found", null)

        val interpreter = holder.interpreter
        val inputs = (0 until interpreter.inputTensorCount).map { i ->
            val t = interpreter.getInputTensor(i)
            mapOf(
                "index" to i,
                "name" to t.name(),
                "shape" to t.shape().toList(),
                "dtype" to t.dataType().name
            )
        }
        val outputs = (0 until interpreter.outputTensorCount).map { i ->
            val t = interpreter.getOutputTensor(i)
            mapOf(
                "index" to i,
                "name" to t.name(),
                "shape" to t.shape().toList(),
                "dtype" to t.dataType().name
            )
        }
        result.success(mapOf("inputs" to inputs, "outputs" to outputs))
    }

    private fun handleRun(call: MethodCall, result: MethodChannel.Result) {
        val modelId = call.argument<String>("modelId")!!
        val holder = loadedModels[modelId]
            ?: return result.error("MODEL_NOT_FOUND", "Interpreter not found", null)
        val interpreter = holder.interpreter
        val inputMap = call.argument<Map<String, Any>>("input")!!

        // Prepare inputs
        val inputs = Array<Any>(interpreter.inputTensorCount) { 0 }
        val nameToIdx = mutableMapOf<String, Int>()
        for (i in 0 until interpreter.inputTensorCount) {
            val t = interpreter.getInputTensor(i)
            nameToIdx[t.name()] = i
        }
        for ((name, raw) in inputMap) {
            val idx = nameToIdx[name]
                ?: return result.error("INPUT_MISMATCH", "No input named '$name'", null)
            val t = interpreter.getInputTensor(idx)
            val type = t.dataType()
            val list = (raw as List<Number>)
            val expectedSize = t.shape().reduce { a, b -> a * b }
            if (list.size != expectedSize) {
                return result.error("SHAPE_MISMATCH", "$name expects $expectedSize elements", null)
            }
            inputs[idx] = when (type) {
                DataType.FLOAT32 -> list.map { it.toFloat() }.toFloatArray()
                DataType.UINT8, DataType.INT8 -> ByteArray(list.size) { list[it].toByte() }
                else -> return result.error("UNSUPPORTED_DTYPE", "${type.name} unsupported for input", null)
            }
        }

        // Prepare outputs
        val outputs = mutableMapOf<Int, Any>()
        for (i in 0 until interpreter.outputTensorCount) {
            val t = interpreter.getOutputTensor(i)
            val type = t.dataType()
            val numElements = t.shape().reduce { a, b -> a * b }
            outputs[i] = when (type) {
                DataType.FLOAT32 -> FloatArray(numElements)
                DataType.UINT8, DataType.INT8 -> ByteArray(numElements)
                else -> return result.error("UNSUPPORTED_DTYPE", "${type.name} unsupported", null)
            }
        }

        // Resize if needed & allocate (note: this is a no-op for fixed shapes)
        for (i in 0 until interpreter.inputTensorCount) {
            val shape = interpreter.getInputTensor(i).shape()
            interpreter.resizeInput(i, shape)
        }
        interpreter.allocateTensors()

        val start = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        val latency = (System.nanoTime() - start) / 1_000.0  // µs

        // Format output
        val finalOutput = mutableMapOf<String, Any>()
        for ((i, raw) in outputs) {
            val t = interpreter.getOutputTensor(i)
            val name = t.name()
            finalOutput[name] = when (raw) {
                is FloatArray -> raw.toList()
                is ByteArray  -> raw.map { it.toInt() }
                else          -> raw
            }
        }

        val accel = if (holder.gpuDelegate != null) "GPU" else "CPU"
        result.success(mapOf(
            "output" to finalOutput,
            "inferenceTime" to latency,
            "acceleratorUsed" to accel
        ))
    }

    private fun handleDispose(call: MethodCall, result: MethodChannel.Result) {
        val modelId = call.argument<String>("modelId")!!
        loadedModels[modelId]?.let {
            it.interpreter.close()
            it.gpuDelegate?.close()
        }
        loadedModels.remove(modelId)
        result.success(null)
    }

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        loadedModels.values.forEach {
            it.interpreter.close()
            it.gpuDelegate?.close()
        }
        loadedModels.clear()
        channel.setMethodCallHandler(null)
    }
}