package com.ooplab.exercises_fitfuel

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

data class LandmarkPoint(val x: Float, val y: Float, val estimated: Boolean = false)

class LandmarkSmoother(
    landmarkCount: Int = 33,
    minCutoff: Float = 1.5f,
    beta: Float = 0.01f
) {
    private val xFilters = Array(landmarkCount) { OneEuroFilter(minCutoff, beta) }
    private val yFilters = Array(landmarkCount) { OneEuroFilter(minCutoff, beta) }

    fun smooth(landmarks: List<NormalizedLandmark>, timestampSec: Double): List<LandmarkPoint> =
        landmarks.mapIndexed { i, lm ->
            LandmarkPoint(
                x = xFilters[i].filter(lm.x(), timestampSec),
                y = yFilters[i].filter(lm.y(), timestampSec)
            )
        }

    fun reset() {
        xFilters.forEach { it.reset() }
        yFilters.forEach { it.reset() }
    }
}
