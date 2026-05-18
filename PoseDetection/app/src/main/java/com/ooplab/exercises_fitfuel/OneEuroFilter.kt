package com.ooplab.exercises_fitfuel

import kotlin.math.PI
import kotlin.math.abs

/**
 * One Euro Filter for smoothing a single float signal in real-time.
 *
 * minCutoff: lower → smoother when still, more lag (try 1.0–2.0)
 * beta:      higher → less lag during fast movement (try 0.005–0.05)
 * dCutoff:   cutoff for the derivative low-pass (1.0 is fine)
 */
class OneEuroFilter(
    private val minCutoff: Float = 1.5f,
    private val beta: Float = 0.01f,
    private val dCutoff: Float = 1.0f
) {
    private var xHat: Float? = null
    private var dxHat = 0f
    private var lastTimestamp = 0.0

    fun filter(x: Float, timestampSec: Double): Float {
        val prev = xHat ?: run {
            xHat = x
            lastTimestamp = timestampSec
            return x
        }
        val dt = (timestampSec - lastTimestamp).toFloat().coerceAtLeast(1e-6f)
        lastTimestamp = timestampSec

        val dx = (x - prev) / dt
        dxHat = lowPass(dx, dxHat, alpha(dCutoff, dt))

        val cutoff = minCutoff + beta * abs(dxHat)
        val result = lowPass(x, prev, alpha(cutoff, dt))
        xHat = result
        return result
    }

    fun reset() {
        xHat = null
        dxHat = 0f
        lastTimestamp = 0.0
    }

    private fun alpha(cutoff: Float, dt: Float): Float {
        val r = 2f * PI.toFloat() * cutoff * dt
        return r / (r + 1f)
    }

    private fun lowPass(x: Float, prev: Float, a: Float) = a * x + (1f - a) * prev
}
