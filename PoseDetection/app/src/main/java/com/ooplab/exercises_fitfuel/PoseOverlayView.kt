package com.ooplab.exercises_fitfuel

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class PoseOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private val dotPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val linePaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    // Estimated landmarks drawn semi-transparent with dashed lines
    private val estimatedDotPaint = Paint().apply {
        color = Color.argb(160, 0, 220, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val estimatedLinePaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
        pathEffect = DashPathEffect(floatArrayOf(14f, 9f), 0f)
    }

    private var landmarks: List<LandmarkPoint> = emptyList()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    companion object {
        val POSE_CONNECTIONS = listOf(
            // Face
            0 to 1, 1 to 2, 2 to 3, 3 to 7,
            0 to 4, 4 to 5, 5 to 6, 6 to 8,
            9 to 10,
            // Shoulders
            11 to 12,
            // Left arm
            11 to 13, 13 to 15, 15 to 17, 17 to 19, 19 to 15, 15 to 21,
            // Right arm
            12 to 14, 14 to 16, 16 to 18, 18 to 20, 20 to 16, 16 to 22,
            // Torso
            11 to 23, 12 to 24, 23 to 24,
            // Left leg
            23 to 25, 25 to 27, 27 to 29, 29 to 31, 31 to 27,
            // Right leg
            24 to 26, 26 to 28, 28 to 30, 30 to 32, 32 to 28
        )
    }

    fun updateLandmarks(newLandmarks: List<LandmarkPoint>, imgWidth: Int, imgHeight: Int) {
        landmarks = newLandmarks
        imageWidth = imgWidth
        imageHeight = imgHeight
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (landmarks.isEmpty()) return

        val scale   = maxOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        val offsetX = (imageWidth  * scale - width)  / 2f
        val offsetY = (imageHeight * scale - height) / 2f

        fun sx(x: Float) = x * imageWidth  * scale - offsetX
        fun sy(y: Float) = y * imageHeight * scale - offsetY

        // Connections
        for ((start, end) in POSE_CONNECTIONS) {
            if (start < landmarks.size && end < landmarks.size) {
                val s     = landmarks[start]
                val e     = landmarks[end]
                val paint = if (s.estimated || e.estimated) estimatedLinePaint else linePaint
                canvas.drawLine(sx(s.x), sy(s.y), sx(e.x), sy(e.y), paint)
            }
        }

        // Dots
        for (lm in landmarks) {
            if (lm.estimated) {
                canvas.drawCircle(sx(lm.x), sy(lm.y), 9f, estimatedDotPaint)
            } else {
                canvas.drawCircle(sx(lm.x), sy(lm.y), 12f, dotPaint)
            }
        }
    }
}
