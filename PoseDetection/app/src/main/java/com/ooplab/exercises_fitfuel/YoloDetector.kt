package com.ooplab.exercises_fitfuel

import android.content.Context
import android.media.Image
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

private const val TAG = "YoloDetector"
private const val INPUT_SIZE = 320
private const val NUM_CANDIDATES = 2100
private const val NUM_CLASSES = 80
private const val PERSON_CLASS = 0
private const val TV_CLASS = 62
private const val LAPTOP_CLASS = 63
private const val CONFIDENCE_THRESHOLD = 0.45f
private const val MONITOR_CONFIDENCE_THRESHOLD = 0.35f
private const val NMS_IOU_THRESHOLD = 0.45f

private data class RawDetection(
    val classId: Int,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
)

data class YoloResult(
    val personDetected: Boolean,
    val monitorDetected: Boolean,
    val personConfidence: Float?,
    val monitorConfidence: Float?,
)

class YoloDetector(context: Context) {
    private val interpreter: Interpreter

    init {
        val afd = context.assets.openFd("yolov8n_float16.tflite")
        val channel = FileInputStream(afd.fileDescriptor).channel
        val modelBuffer = channel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength,
        )
        channel.close()
        afd.close()

        interpreter = try {
            val gpuDelegate = GpuDelegate()
            val opts = Interpreter.Options().apply { addDelegate(gpuDelegate) }
            Interpreter(modelBuffer, opts).also { Log.d(TAG, "YOLO running on GPU delegate") }
        } catch (t: Throwable) {
            Log.w(TAG, "GPU delegate unavailable (${t.message}), falling back to CPU")
            Interpreter(modelBuffer, Interpreter.Options().apply { numThreads = 4 })
        }
    }

    fun detect(image: Image, rotationDegrees: Int): YoloResult {
        val input = buildInputTensor(image, rotationDegrees)
        val output = Array(1) { Array(84) { FloatArray(NUM_CANDIDATES) } }
        interpreter.run(input, output)
        return buildResult(parseAndNms(output[0]))
    }

    private fun buildInputTensor(image: Image, rotationDegrees: Int): ByteBuffer {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuf          = yPlane.buffer
        val uBuf          = uPlane.buffer
        val vBuf          = vPlane.buffer
        val yRowStride    = yPlane.rowStride
        val uvRowStride   = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val landW = image.width
        val landH = image.height

        val outW = if (rotationDegrees == 90 || rotationDegrees == 270) landH else landW
        val outH = if (rotationDegrees == 90 || rotationDegrees == 270) landW else landH

        val buf = ByteBuffer
            .allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
            .order(ByteOrder.nativeOrder())

        for (oy in 0 until INPUT_SIZE) {
            for (ox in 0 until INPUT_SIZE) {
                val sx = ox * outW / INPUT_SIZE
                val sy = oy * outH / INPUT_SIZE

                val lx: Int; val ly: Int
                when (rotationDegrees) {
                    90  -> { lx = sy;             ly = landH - 1 - sx }
                    180 -> { lx = landW - 1 - sx; ly = landH - 1 - sy }
                    270 -> { lx = landW - 1 - sy; ly = sx             }
                    else -> { lx = sx;             ly = sy             }
                }

                val lxC = lx.coerceIn(0, landW - 1)
                val lyC = ly.coerceIn(0, landH - 1)

                val yVal  = yBuf.get(lyC * yRowStride + lxC).toInt() and 0xFF
                val uvIdx = (lyC / 2) * uvRowStride + (lxC / 2) * uvPixelStride
                val uVal  = uBuf.get(uvIdx).toInt() and 0xFF
                val vVal  = vBuf.get(uvIdx).toInt() and 0xFF

                val r = (yVal + 1.402f    * (vVal - 128)).coerceIn(0f, 255f) / 255f
                val g = (yVal - 0.344136f * (uVal - 128) - 0.714136f * (vVal - 128)).coerceIn(0f, 255f) / 255f
                val b = (yVal + 1.772f    * (uVal - 128)).coerceIn(0f, 255f) / 255f

                buf.putFloat(r)
                buf.putFloat(g)
                buf.putFloat(b)
            }
        }

        buf.rewind()
        return buf
    }

    private fun parseAndNms(output84x2100: Array<FloatArray>): List<RawDetection> {
        val candidates = mutableListOf<RawDetection>()
        for (i in 0 until NUM_CANDIDATES) {
            var bestId = 0; var bestScore = 0f
            for (c in 0 until NUM_CLASSES) {
                val s = output84x2100[4 + c][i]
                if (s > bestScore) { bestScore = s; bestId = c }
            }
            val threshold = if (bestId == TV_CLASS || bestId == LAPTOP_CLASS)
                MONITOR_CONFIDENCE_THRESHOLD else CONFIDENCE_THRESHOLD
            if (bestScore < threshold) continue

            val cx = output84x2100[0][i]; val cy = output84x2100[1][i]
            val w  = output84x2100[2][i]; val h  = output84x2100[3][i]
            candidates.add(
                RawDetection(
                    classId    = bestId,
                    confidence = bestScore,
                    left   = ((cx - w / 2) / INPUT_SIZE).coerceIn(0f, 1f),
                    top    = ((cy - h / 2) / INPUT_SIZE).coerceIn(0f, 1f),
                    right  = ((cx + w / 2) / INPUT_SIZE).coerceIn(0f, 1f),
                    bottom = ((cy + h / 2) / INPUT_SIZE).coerceIn(0f, 1f),
                ),
            )
        }
        return nms(candidates)
    }

    private fun nms(candidates: List<RawDetection>): List<RawDetection> {
        val byClass = candidates.groupBy { it.classId }
        val results = mutableListOf<RawDetection>()
        for ((_, group) in byClass) {
            val sorted = group.sortedByDescending { it.confidence }
            val suppressed = BooleanArray(sorted.size)
            for (i in sorted.indices) {
                if (suppressed[i]) continue
                results.add(sorted[i])
                for (j in i + 1 until sorted.size) {
                    if (!suppressed[j] && iou(sorted[i], sorted[j]) > NMS_IOU_THRESHOLD)
                        suppressed[j] = true
                }
            }
        }
        return results
    }

    private fun iou(a: RawDetection, b: RawDetection): Float {
        val iL = max(a.left, b.left); val iT = max(a.top, b.top)
        val iR = min(a.right, b.right); val iB = min(a.bottom, b.bottom)
        if (iR <= iL || iB <= iT) return 0f
        val inter = (iR - iL) * (iB - iT)
        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)
        return inter / (aArea + bArea - inter)
    }

    private fun buildResult(detections: List<RawDetection>): YoloResult {
        var hasPerson = false; var hasMonitor = false
        var personConf: Float? = null; var monitorConf: Float? = null
        for (d in detections) {
            if (d.classId == PERSON_CLASS) {
                hasPerson = true
                if (personConf == null || d.confidence > personConf!!) personConf = d.confidence
            }
            if (d.classId == TV_CLASS || d.classId == LAPTOP_CLASS) {
                hasMonitor = true
                if (monitorConf == null || d.confidence > monitorConf!!) monitorConf = d.confidence
            }
        }
        return YoloResult(hasPerson, hasMonitor, personConf, monitorConf)
    }

    fun dispose() = interpreter.close()
}
