package com.ooplab.exercises_fitfuel

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
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
    private lateinit var tvDistanceStatus: TextView
    private lateinit var tvSideViewStatus: TextView

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
    @Volatile private var lastImageWidth: Int  = 1
    @Volatile private var lastImageHeight: Int = 1

    // Camera2 optics — read once per camera bind, used for distance estimation
    @Volatile private var focalLengthMm: Float  = 4.25f  // sensible fallback
    @Volatile private var sensorWidthMm: Float  = 6.4f   // landscape sensor width
    @Volatile private var sensorHeightMm: Float = 4.8f   // landscape sensor height
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

    // Hip spread / torso height ratio threshold for side-view detection.
    // Accepts camera within ~30° of true side view; unaffected by torso lean.
    private val SIDE_VIEW_THRESHOLD = 0.35f

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
        tvDistanceStatus = findViewById(R.id.tvDistanceStatus)
        tvSideViewStatus = findViewById(R.id.tvSideViewStatus)

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
                val currentState = appState
                val landmarks = result.landmarks()
                val w = lastImageWidth
                val h = lastImageHeight
                val tSec = android.os.SystemClock.elapsedRealtime() / 1000.0
                val sm = if (landmarks.isNotEmpty()) {
                    landmarkSmoother.smooth(landmarks[0], tSec)
                } else {
                    landmarkSmoother.reset(); null
                }

                // Check side view on the raw-smoothed landmarks (before estimation)
                val sideOk = sm != null &&
                             currentState == AppState.POSE &&
                             checkSideView(sm)

                // Only apply leg estimation in a confirmed seated side view —
                // prevents bogus leg lines when the person faces the camera
                val smoothed = when {
                    sm == null -> emptyList()
                    sideOk     -> legEstimator.estimate(landmarks[0], sm)
                    else       -> sm
                }

                // ── Continuous side view indicator ────────────────────────────
                if (currentState == AppState.POSE) {
                    runOnUiThread {
                        tvSideViewStatus.setTextColor(
                            if (sideOk) COLOR_DETECTED else COLOR_NOT_DETECTED
                        )
                        tvSideViewStatus.text = if (sideOk) "● Side  OK"
                                               else         "● Side  Adjust angle"
                    }
                }

                // ── Distance (POSE only) ──────────────────────────────────────
                val distanceM = if (currentState == AppState.POSE && smoothed.size >= 25) {
                    val hipMidY = (smoothed[23].y + smoothed[24].y) / 2f
                    val sensorDimForY = if (h > w) sensorWidthMm else sensorHeightMm
                    DistanceEstimator.estimate(
                        noseY          = smoothed[0].y,
                        hipMidY        = hipMidY,
                        imageHeightPx  = h,
                        focalMm        = focalLengthMm,
                        sensorHeightMm = sensorDimForY,
                    )
                } else null

                runOnUiThread {
                    poseOverlayView.updateLandmarks(smoothed, w, h)
                    if (distanceM != null) {
                        tvDistanceStatus.text = "● Distance  ${"%.2f".format(distanceM)}m"
                        tvDistanceStatus.setTextColor(Color.WHITE)
                    } else {
                        tvDistanceStatus.text = "● Distance  --"
                        tvDistanceStatus.setTextColor(Color.parseColor("#9E9E9E"))
                    }
                }
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

    private fun readCameraCharacteristics() {
        try {
            val mgr = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val facing = if (cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA)
                CameraCharacteristics.LENS_FACING_BACK else CameraCharacteristics.LENS_FACING_FRONT
            val id = mgr.cameraIdList.firstOrNull { cid ->
                mgr.getCameraCharacteristics(cid).get(CameraCharacteristics.LENS_FACING) == facing
            } ?: return
            val chars = mgr.getCameraCharacteristics(id)
            chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                ?.firstOrNull()?.let { focalLengthMm = it }
            chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)?.let { sz ->
                // SizeF is always landscape (width > height) regardless of phone orientation
                sensorWidthMm  = sz.width
                sensorHeightMm = sz.height
            }
            Log.d("CameraChars", "focal=${focalLengthMm}mm  sensor=${sensorWidthMm}×${sensorHeightMm}mm")
        } catch (e: Exception) {
            Log.w("CameraChars", "Could not read characteristics: ${e.message}")
        }
    }

    private fun setupCamera() {
        readCameraCharacteristics()
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
                        .alpha(0f).setDuration(500)
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
    // Side view check
    // =========================================================================

    /**
     * Returns true when the camera is at a side-on angle to the subject.
     *
     * Uses only hip spread relative to torso height. Hips stay planted in the
     * chair regardless of how much the person leans forward — making this
     * signal stable for typical seated office postures.
     *
     * Shoulder spread is intentionally excluded: when someone leans over a desk
     * MediaPipe shifts the shoulder landmarks, causing false "not side-on" reads.
     * Ears are intentionally excluded: the far ear can be occluded by hair or
     * head angle even when the camera is not truly side-on.
     *
     * Geometry: hipSpread ≈ hipWidth × cos(angleFromFrontal) / torsoHeight.
     * With typical proportions (hip width ≈ 70 % of torso height) a threshold
     * of 0.35 accepts angles within ~30° of true side view.
     */
    private fun checkSideView(smoothed: List<LandmarkPoint>): Boolean {
        if (smoothed.size < 25) return false
        val lHip = smoothed[23]; val rHip = smoothed[24]
        val lShoulder = smoothed[11]; val rShoulder = smoothed[12]
        val torsoH = kotlin.math.abs(
            (lShoulder.y + rShoulder.y) / 2f - (lHip.y + rHip.y) / 2f
        )
        if (torsoH < 0.01f) return false
        return kotlin.math.abs(lHip.x - rHip.x) / torsoH < SIDE_VIEW_THRESHOLD
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

        // Light
        tvLightStatus.setTextColor(when (lightOk) {
            null  -> COLOR_NEUTRAL
            true  -> COLOR_DETECTED
            false -> COLOR_NOT_DETECTED
        })

        // Person / monitor — grey until light passes
        val yoloActive = lightOk == true
        tvPersonStatus.setTextColor(when {
            !yoloActive    -> COLOR_NEUTRAL
            personDetected -> COLOR_DETECTED
            else           -> COLOR_NOT_DETECTED
        })
        tvMonitorStatus.setTextColor(when {
            !yoloActive     -> COLOR_NEUTRAL
            monitorDetected -> COLOR_DETECTED
            else            -> COLOR_NOT_DETECTED
        })

        // Progress bar
        val showProgress = yoloActive && personDetected && monitorDetected && confirmCount > 0
        confirmProgress.visibility = if (showProgress) View.VISIBLE else View.INVISIBLE
        confirmProgress.progress   = confirmCount

        // Status message
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
