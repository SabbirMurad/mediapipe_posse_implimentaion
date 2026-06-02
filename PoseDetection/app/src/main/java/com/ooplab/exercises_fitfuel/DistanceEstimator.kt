package com.ooplab.exercises_fitfuel

object DistanceEstimator {

    // Nose → hip midpoint accounts for ~37 % of standing body height
    // (nose ≈ 93 % from floor, hip joint ≈ 56 % from floor → delta ≈ 37 %)
    private const val TORSO_RATIO = 0.37f

    /**
     * Estimates the distance (in metres) from the camera to the subject.
     *
     * Formula:
     *   focal_length_px = (focalMm / sensorHeightMm) × imageHeightPx
     *   real_torso_cm   = personHeightCm × 0.52
     *   distance_cm     = (real_torso_cm × focal_length_px) / pixelHeight
     *
     * @param noseY          Normalised Y of nose landmark (0–1)
     * @param hipMidY        Normalised Y of hip midpoint (0–1)
     * @param imageHeightPx  Height of the camera frame in pixels
     * @param focalMm        Focal length reported by Camera2 (mm)
     * @param sensorHeightMm Physical sensor height reported by Camera2 (mm)
     * @param personHeightCm Assumed standing height of the subject (default 170 cm)
     * @return Distance in metres, or null if inputs are degenerate
     */
    fun estimate(
        noseY: Float,
        hipMidY: Float,
        imageHeightPx: Int,
        focalMm: Float,
        sensorHeightMm: Float,
        personHeightCm: Float = 170f,
    ): Float? {
        val pixelHeight = (hipMidY - noseY) * imageHeightPx
        if (pixelHeight < 20f || focalMm <= 0f || sensorHeightMm <= 0f) return null

        val focalLengthPx  = (focalMm / sensorHeightMm) * imageHeightPx
        val realTorsoHeightCm = personHeightCm * TORSO_RATIO
        val distanceCm = (realTorsoHeightCm * focalLengthPx) / pixelHeight
        return distanceCm / 100f  // → metres
    }
}
