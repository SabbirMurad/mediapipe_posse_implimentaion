package com.ooplab.exercises_fitfuel

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.*

class TiltMonitor(
    context: Context,
    private val onAngleChanged: (angle: Double) -> Unit,
) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer =
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private var fx = 0f; private var fy = 0f; private var fz = 0f
    private var initialized = false

    fun start() {
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI)
    }

    fun stop() = sensorManager.unregisterListener(this)

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ACCELEROMETER) return
        val x = event.values[0]; val y = event.values[1]; val z = event.values[2]

        if (!initialized) {
            fx = x; fy = y; fz = z; initialized = true
        } else {
            // EMA low-pass filter — same alpha as live_guidence (0.1)
            fx += ALPHA * (x - fx)
            fy += ALPHA * (y - fy)
            fz += ALPHA * (z - fz)
        }

        // Inclination from horizontal: 0° = lying flat, 90° = perfectly upright portrait
        val angle = atan2(fy.toDouble(), sqrt((fx * fx + fz * fz).toDouble())) * (180.0 / PI)
        onAngleChanged(angle)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    companion object {
        private const val ALPHA = 0.1f
        const val MIN_ANGLE = 85.0
        const val MAX_ANGLE = 95.0
        fun isAcceptable(angle: Double) = angle in MIN_ANGLE..MAX_ANGLE
    }
}
