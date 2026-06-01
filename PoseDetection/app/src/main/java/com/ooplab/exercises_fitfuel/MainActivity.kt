package com.ooplab.exercises_fitfuel

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private enum class AppState { LIGHT_CHECK, DETECTING, POSE }

    // -------------------------------------------------------------------------
    // Views
    // -------------------------------------------------------------------------
    private lateinit var previewView: PreviewView
    private lateinit var poseOverlayView: PoseOverlayView
    private lateinit var detectionPanel: LinearLayout
    private lateinit var tvLightStatus: TextView
    private lateinit var tvPersonStatus: TextView
    private lateinit var tvMonitorStatus: TextView
    private lateinit var confirmProgress: ProgressBar
    private lateinit var tvStatusMessage: TextView
    private lateinit var tvTiltStatus: TextView

    // -------------------------------------------------------------------------
    // Tilt
    // -------------------------------------------------------------------------
    private lateinit var tiltMonitor: TiltMonitor

    // -------------------------------------------------------------------------
    // ML
    // -------------------------------------------------------------------------
    private lateinit var cameraExecutor: ExecutorService
    @Volatile private var poseLandmarker: PoseLandmarker? = null
    private var yoloDetector: YoloDetector? = null

    // -------------------------------------------------------------------------
    // Camera
    // -------------------------------------------------------------------------
    private var cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
    @Volatile private var lastImageWidth: Int = 1
    @Volatile private var lastImageHeight: Int = 1
    private var frameCounter = 0
    private val processEveryNFrames = 1

    // -------------------------------------------------------------------------
    // Smoother + leg estimator
    // -------------------------------------------------------------------------
    private val landmarkSmoother = LandmarkSmoother(minCutoff = 0.5f, beta = 0.5f)
    private val legEstimator     = LegEstimator()

    // -------------------------------------------------------------------------
    // State machine  (only written on cameraExecutor thread except resets)
    // -------------------------------------------------------------------------
    @Volatile private var appState = AppState.LIGHT_CHECK
    private var confirmationCount = 0
    private val REQUIRED_CONFIRMATIONS = 4

    // Light thresholds (same as live_guidence project)
    private val MIN_LUMINANCE = 0.25
    private val MAX_LUMINANCE = 0.85

    // -------------------------------------------------------------------------
    // Colors
    // -------------------------------------------------------------------------
    private val COLOR_DETECTED     = Color.parseColor("#43A047") // green
    private val COLOR_NOT_DETECTED = Color.parseColor("#E53935") // red
    private val COLOR_NEUTRAL      = Color.parseColor("#9E9E9E") // grey

    // =========================================================================
    // Lifecycle
    // =========================================================================

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        setupEdgeToEdge()

        previewView      = findViewById(R.id.previewCam)
        poseOverlayView  = findViewById(R.id.poseOverlay)
        detectionPanel   = findViewById(R.id.detectionPanel)
        tvLightStatus    = findViewById(R.id.tvLightStatus)
        tvPersonStatus   = findViewById(R.id.tvPersonStatus)
        tvMonitorStatus  = findViewById(R.id.tvMonitorStatus)
        confirmProgress  = findViewById(R.id.confirmProgress)
        tvStatusMessage  = findViewById(R.id.tvStatusMessage)
        tvTiltStatus     = findViewById(R.id.tvTiltStatus)

        tiltMonitor = TiltMonitor(this) { angle ->
            val ok      = TiltMonitor.isAcceptable(angle)
            val angleStr = "%.1f".format(angle)
            tvTiltStatus.setTextColor(if (ok) COLOR_DETECTED else COLOR_NOT_DETECTED)
            tvTiltStatus.text = if (ok) "● Tilt  $angleStr°"
                                else    "● Tilt  $angleStr°  ·  Hold phone upright"
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        // Load YOLO model on the executor thread so it's ready for the first frame
        cameraExecutor.execute { yoloDetector = YoloDetector(this) }

        // Show initial panel state (all neutral, checking light)
        updatePanel(lightOk = null)

        findViewById<Button>(R.id.btnSwitchCamera).setOnClickListener {
            cameraSelector = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA)
                CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            fullReset()
            setupCamera()
        }

        findViewById<Button>(R.id.btnReset).setOnClickListener {
            fullReset()
        }

        requestCameraPermission()
    }

    private fun fullReset() {
        appState = AppState.LIGHT_CHECK
        confirmationCount = 0
        poseLandmarker?.close()
        poseLandmarker = null
        landmarkSmoother.reset()
        legEstimator.reset()
        cameraExecutor.execute {
            yoloDetector?.dispose()
            yoloDetector = YoloDetector(this)
        }
        runOnUiThread {
            poseOverlayView.updateLandmarks(emptyList(), 1, 1)
            updatePanel(lightOk = null)
        }
    }

    // =========================================================================
    // Pose landmarker — created only after detection confirms both targets
    // =========================================================================

    private fun initializePoseLandmarker() {
        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder().setModelAssetPath("pose_landmarker_full.task").build()
            )
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { result, _ ->
                val landmarks = result.landmarks()
                val w = lastImageWidth
                val h = lastImageHeight
                val tSec = android.os.SystemClock.elapsedRealtime() / 1000.0
                val smoothed = if (landmarks.isNotEmpty()) {
                    val sm = landmarkSmoother.smooth(landmarks[0], tSec)
                    legEstimator.estimate(landmarks[0], sm)
                } else {
                    landmarkSmoother.reset(); emptyList()
                }
                runOnUiThread { poseOverlayView.updateLandmarks(smoothed, w, h) }
            }.build()

        poseLandmarker = PoseLandmarker.createFromOptions(this, options)
    }

    // =========================================================================
    // Camera
    // =========================================================================

    private val cameraPermissionLauncher: ActivityResultLauncher<String> =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) setupCamera()
            else Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
        }

    private fun requestCameraPermission() {
        if (hasCameraPermission()) setupCamera()
        else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun setupCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview  = Preview.Builder().build().apply { setSurfaceProvider(previewView.surfaceProvider) }
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().apply { setAnalyzer(cameraExecutor, ::analyzeImage) }
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, cameraSelector, preview, analyzer)
            } catch (e: Exception) {
                Log.e("CameraSetup", "Bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // =========================================================================
    // Frame analysis
    // =========================================================================

    @OptIn(ExperimentalGetImage::class)
    private fun analyzeImage(imageProxy: ImageProxy) {
        if (++frameCounter % processEveryNFrames != 0) { imageProxy.close(); return }

        val mediaImage = imageProxy.image
        if (mediaImage == null || imageProxy.format != ImageFormat.YUV_420_888) {
            Log.e("AnalyzeImage", "Unsupported format"); imageProxy.close(); return
        }

        when (appState) {
            AppState.LIGHT_CHECK -> runLightCheckPhase(mediaImage, imageProxy)
            AppState.DETECTING   -> runDetectionPhase(mediaImage, imageProxy)
            AppState.POSE        -> runPosePhase(mediaImage, imageProxy)
        }
    }

    // -------------------------------------------------------------------------
    // Phase 1: Light check
    // -------------------------------------------------------------------------

    private fun runLightCheckPhase(mediaImage: Image, imageProxy: ImageProxy) {
        val luminance = computeLuminance(mediaImage)
        when {
            luminance < MIN_LUMINANCE -> runOnUiThread {
                updatePanel(lightOk = false, message = "Room is too dark — turn on more lights")
            }
            luminance > MAX_LUMINANCE -> runOnUiThread {
                updatePanel(lightOk = false, message = "Too bright — reduce glare or step back")
            }
            else -> {
                // Light is good — start YOLO detection
                appState = AppState.DETECTING
                runOnUiThread { updatePanel(lightOk = true) }
            }
        }
        imageProxy.close()
    }

    // Average the Y (luma) plane, sampling every 20th pixel — same algorithm as live_guidence
    private fun computeLuminance(image: Image): Double {
        val yPlane    = image.planes[0]
        val yBuf      = yPlane.buffer
        val rowStride = yPlane.rowStride
        val width     = image.width
        val height    = image.height
        val step      = 20
        var sum   = 0L
        var count = 0
        var row = 0
        while (row < height) {
            var col = 0
            while (col < width) {
                sum += yBuf.get(row * rowStride + col).toInt() and 0xFF
                count++
                col += step
            }
            row += step
        }
        return if (count == 0) 0.5 else (sum.toDouble() / count) / 255.0
    }

    // -------------------------------------------------------------------------
    // Phase 2: YOLO detection
    // -------------------------------------------------------------------------

    private fun runDetectionPhase(mediaImage: Image, imageProxy: ImageProxy) {
        val result = yoloDetector?.detect(mediaImage, imageProxy.imageInfo.rotationDegrees)
            ?: run { imageProxy.close(); return }

        if (result.personDetected && result.monitorDetected) {
            confirmationCount++

            if (confirmationCount >= REQUIRED_CONFIRMATIONS) {
                runOnUiThread {
                    updatePanel(
                        lightOk         = true,
                        personDetected  = true,
                        monitorDetected = true,
                        confirmCount    = REQUIRED_CONFIRMATIONS,
                        message         = "Loading…"
                    )
                }
                initializePoseLandmarker()
                appState = AppState.POSE
                yoloDetector?.dispose()
                yoloDetector = null
                runOnUiThread {
                    detectionPanel.animate()
                        .alpha(0f)
                        .setDuration(500)
                        .withEndAction {
                            detectionPanel.visibility = View.GONE
                            detectionPanel.alpha = 1f
                        }.start()
                }
            } else {
                runOnUiThread {
                    updatePanel(
                        lightOk         = true,
                        personDetected  = true,
                        monitorDetected = true,
                        confirmCount    = confirmationCount,
                        message         = "Hold still…"
                    )
                }
            }
        } else {
            confirmationCount = 0
            runOnUiThread {
                updatePanel(
                    lightOk         = true,
                    personDetected  = result.personDetected,
                    monitorDetected = result.monitorDetected,
                    confirmCount    = 0
                )
            }
        }
        imageProxy.close()
    }

    // -------------------------------------------------------------------------
    // Phase 3: Pose detection
    // -------------------------------------------------------------------------

    private fun runPosePhase(mediaImage: Image, imageProxy: ImageProxy) {
        val bitmap = yuvToRgb(mediaImage, imageProxy)
        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            if (cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA)
                postScale(-1f, 1f, bitmap.width.toFloat(), bitmap.height.toFloat())
        }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        lastImageWidth  = rotated.width
        lastImageHeight = rotated.height
        poseLandmarker?.detectAsync(BitmapImageBuilder(rotated).build(), imageProxy.imageInfo.timestamp)
        imageProxy.close()
    }

    // =========================================================================
    // Panel UI helper
    // =========================================================================

    /**
     * Single method that drives all panel indicator states.
     *
     * lightOk  = null  → neutral grey (not yet checked)
     * lightOk  = false → red  (out of range, message shown)
     * lightOk  = true  → green (passed, locked)
     *
     * During LIGHT_CHECK: pass lightOk only; person/monitor stay neutral.
     * During DETECTING:   pass lightOk=true + person/monitorDetected + confirmCount.
     */
    private fun updatePanel(
        lightOk: Boolean?,
        personDetected: Boolean  = false,
        monitorDetected: Boolean = false,
        confirmCount: Int        = 0,
        message: String?         = null,
    ) {
        detectionPanel.visibility = View.VISIBLE

        // Light indicator
        tvLightStatus.setTextColor(
            when (lightOk) {
                null  -> COLOR_NEUTRAL
                true  -> COLOR_DETECTED
                false -> COLOR_NOT_DETECTED
            }
        )

        // Person / monitor indicators — grey until light passes
        val yoloActive = lightOk == true
        tvPersonStatus.setTextColor(
            when {
                !yoloActive     -> COLOR_NEUTRAL
                personDetected  -> COLOR_DETECTED
                else            -> COLOR_NOT_DETECTED
            }
        )
        tvMonitorStatus.setTextColor(
            when {
                !yoloActive      -> COLOR_NEUTRAL
                monitorDetected  -> COLOR_DETECTED
                else             -> COLOR_NOT_DETECTED
            }
        )

        // Confirmation progress bar
        val showProgress = yoloActive && personDetected && monitorDetected && confirmCount > 0
        confirmProgress.visibility = if (showProgress) View.VISIBLE else View.INVISIBLE
        confirmProgress.progress   = confirmCount

        // Status message (light guidance, "Hold still…", "Loading…")
        if (message != null) {
            tvStatusMessage.text       = message
            tvStatusMessage.visibility = View.VISIBLE
        } else {
            tvStatusMessage.visibility = View.GONE
        }
    }

    // =========================================================================
    // YUV → Bitmap
    // =========================================================================

    private fun yuvToRgb(image: Image, imageProxy: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }

    private fun setupEdgeToEdge() {
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
    }

    override fun onResume() {
        super.onResume()
        tiltMonitor.start()
    }

    override fun onPause() {
        super.onPause()
        tiltMonitor.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        poseLandmarker?.close()
        yoloDetector?.dispose()
    }
}
