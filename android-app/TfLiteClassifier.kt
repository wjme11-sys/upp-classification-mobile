package com.example.clasificadorulcerasfinal

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.flex.FlexDelegate


class TfLiteClassifier(context: Context) {
    private var interpreter: Interpreter? = null
    private val modelFile = "clasificador_final_v4.tflite"
    
    // Configuración específica para tu modelo EfficientNet
    private val inputSize = 224 
    private val pixelSize = 3
    private val numClasses = 4

  init {
    try {
        val assetFileDescriptor = context.assets.openFd(modelFile)
        val inputStream = java.io.FileInputStream(assetFileDescriptor.fileDescriptor)
        val modelBuffer = inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength
        )

        val options = Interpreter.Options().apply {
            setNumThreads(4)
            // ESTA LÍNEA ES LA CLAVE:
            addDelegate(FlexDelegate()) 
        }

        interpreter = Interpreter(modelBuffer, options)
        android.util.Log.d("IA_DEBUG", "¡SÍ! El modelo cargó con FlexDelegate.")
    } catch (e: Exception) {
        android.util.Log.e("IA_DEBUG", "ERROR AL CARGAR: ${e.message}")
    }
}

    fun classify(bitmap: Bitmap): Pair<Int, Float> {
        if (interpreter == null) return Pair(-1, 0.0f)

        return try {
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            val inputBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            inputBuffer.rewind()

            val intValues = IntArray(224 * 224)
            scaledBitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224)

            // --- AJUSTE PARA EFFICIENTNET: SIN DIVIDIR POR 255 ---
            for (pixelValue in intValues) {
                val r = ((pixelValue shr 16) and 0xFF).toFloat()
                val g = ((pixelValue shr 8) and 0xFF).toFloat()
                val b = (pixelValue and 0xFF).toFloat()

                inputBuffer.putFloat(r)
                inputBuffer.putFloat(g)
                inputBuffer.putFloat(b)
            }

            val outputBuffer = Array(1) { FloatArray(4) }
            interpreter?.run(inputBuffer, outputBuffer)

            // Extraemos el array de probabilidades de la primera fila
            val resultadoArray = outputBuffer[0]

            // --- LOG PARA DEPURAR (Verás los 4 números en el Logcat) ---
            Log.d("IA_DEBUG", "Probabilidades: ${resultadoArray.joinToString(" | ")}")

            var indiceMaximo = -1
            var confianzaMaxima = 0.0f

            for (i in resultadoArray.indices) {
                if (resultadoArray[i] > confianzaMaxima) {
                    confianzaMaxima = resultadoArray[i]
                    indiceMaximo = i
                }
            }

            if (confianzaMaxima < 0.10f) return Pair(-1, confianzaMaxima)

            Pair(indiceMaximo, confianzaMaxima)

        } catch (e: Exception) {
            Log.e("IA_DEBUG", "Error en el proceso: ${e.message}")
            Pair(-1, 0.0f)
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}