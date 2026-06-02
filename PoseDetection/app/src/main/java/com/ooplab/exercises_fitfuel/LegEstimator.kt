package com.ooplab.exercises_fitfuel

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.sqrt

/**
 * Estimates hidden lower-body landmarks for a seated, side-view subject.
 *
 * Stateful — keeps a rolling vote for facing direction so it doesn't flip
 * frame-to-frame.  Call [reset] whenever the camera restarts.
 *
 * Each leg is computed independently from its own hip anchor so left and
 * right landmarks are always at distinct positions.
 *
 * Standing detection: if any knee is clearly visible and well below the hip
 * (drop > 50 % of torso length), the person is assumed to be standing and the
 * estimator returns the smoothed landmarks untouched.
 */
class LegEstimator {

    private var lockedFacingSign: Float? = null
    private val facingVotes = ArrayDeque<Float>()

    fun reset() {
        lockedFacingSign = null
        facingVotes.clear()
    }

    fun estimate(
        raw:      List<NormalizedLandmark>,
        smoothed: List<LandmarkPoint>,
    ): List<LandmarkPoint> {
        if (raw.size < 33 || smoothed.size < 33) return smoothed

        val out = smoothed.toMutableList()

        // ── Torso geometry ───────────────────────────────────────────────────
        val shoulderMidX = (smoothed[L_SHOULDER].x + smoothed[R_SHOULDER].x) / 2f
        val shoulderMidY = (smoothed[L_SHOULDER].y + smoothed[R_SHOULDER].y) / 2f
        val hipMidX      = (smoothed[L_HIP].x      + smoothed[R_HIP].x)      / 2f
        val hipMidY      = (smoothed[L_HIP].y       + smoothed[R_HIP].y)      / 2f
        val torsoLen     = dist(shoulderMidX, shoulderMidY, hipMidX, hipMidY)
        if (torsoLen < 0.01f) return smoothed

        val thighLen = torsoLen * THIGH_RATIO
        val shinLen  = torsoLen * SHIN_RATIO

        // ── Standing guard ────────────────────────────────────────────────────
        // If any high-confidence knee is significantly below the hip the person
        // is standing — estimation would produce a seated geometry, so bail out.
        val standingThreshold = torsoLen * 0.50f
        val standingDetected =
            (isHighConf(raw[L_KNEE]) && smoothed[L_KNEE].y - hipMidY > standingThreshold) ||
            (isHighConf(raw[R_KNEE]) && smoothed[R_KNEE].y - hipMidY > standingThreshold)
        if (standingDetected) return smoothed

        // ── Facing direction (nose offset from shoulder midpoint) ─────────────
        updateFacingVote(smoothed[NOSE].x, shoulderMidX)
        val facingSign = lockedFacingSign
            ?: if (smoothed[NOSE].x > shoulderMidX) 1f else -1f

        // ── Left leg (anchored to L_HIP) ──────────────────────────────────────
        // Trust MediaPipe's knee position if it is highly confident.
        // isKneeForward is intentionally removed — it rejected valid standing
        // positions and is not needed now that standing is caught above.
        val lKneeX: Float
        val lKneeY: Float
        if (isHighConf(raw[L_KNEE])) {
            lKneeX = smoothed[L_KNEE].x
            lKneeY = smoothed[L_KNEE].y
        } else {
            lKneeX = smoothed[L_HIP].x + facingSign * thighLen
            lKneeY = smoothed[L_HIP].y + SEATED_THIGH_DROP * torsoLen
            out[L_KNEE] = LandmarkPoint(lKneeX, lKneeY, estimated = true)
        }

        if (!isHighConf(raw[L_ANKLE])) {
            out[L_ANKLE] = LandmarkPoint(lKneeX, lKneeY + shinLen, estimated = true)
        }

        // ── Right leg (anchored to R_HIP) ─────────────────────────────────────
        val rKneeX: Float
        val rKneeY: Float
        if (isHighConf(raw[R_KNEE])) {
            rKneeX = smoothed[R_KNEE].x
            rKneeY = smoothed[R_KNEE].y
        } else {
            rKneeX = smoothed[R_HIP].x + facingSign * thighLen
            rKneeY = smoothed[R_HIP].y + SEATED_THIGH_DROP * torsoLen
            out[R_KNEE] = LandmarkPoint(rKneeX, rKneeY, estimated = true)
        }

        if (!isHighConf(raw[R_ANKLE])) {
            out[R_ANKLE] = LandmarkPoint(rKneeX, rKneeY + shinLen, estimated = true)
        }

        // ── Heels & foot indices — collapse to ankle to avoid stray lines ─────
        listOf(L_HEEL to L_ANKLE, R_HEEL to R_ANKLE,
               L_FOOT to L_ANKLE, R_FOOT to R_ANKLE)
            .forEach { (footIdx, ankleIdx) ->
                if (!isHighConf(raw[footIdx]))
                    out[footIdx] = LandmarkPoint(out[ankleIdx].x, out[ankleIdx].y, estimated = true)
            }

        return out
    }

    // ── Facing direction ──────────────────────────────────────────────────────

    private fun updateFacingVote(noseX: Float, shoulderMidX: Float) {
        facingVotes.addLast(if (noseX > shoulderMidX) 1f else -1f)
        if (facingVotes.size > VOTE_WINDOW) facingVotes.removeFirst()
        if (facingVotes.size >= VOTE_WINDOW) {
            val fraction = facingVotes.count { it > 0f }.toFloat() / facingVotes.size
            when {
                fraction >= LOCK_FRACTION      -> lockedFacingSign = 1f
                fraction <= 1f - LOCK_FRACTION -> lockedFacingSign = -1f
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun isHighConf(lm: NormalizedLandmark) =
        lm.visibility().isPresent && lm.visibility().get() >= HIGH_CONFIDENCE

    private fun dist(x1: Float, y1: Float, x2: Float, y2: Float) =
        sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    companion object {
        private const val HIGH_CONFIDENCE   = 0.65f
        private const val THIGH_RATIO       = 0.90f
        private const val SHIN_RATIO        = 0.90f
        private const val SEATED_THIGH_DROP = 0.08f
        private const val VOTE_WINDOW       = 20
        private const val LOCK_FRACTION     = 0.70f

        private const val NOSE       = 0
        private const val L_SHOULDER = 11; private const val R_SHOULDER = 12
        private const val L_HIP      = 23; private const val R_HIP      = 24
        private const val L_KNEE     = 25; private const val R_KNEE     = 26
        private const val L_ANKLE    = 27; private const val R_ANKLE    = 28
        private const val L_HEEL     = 29; private const val R_HEEL     = 30
        private const val L_FOOT     = 31; private const val R_FOOT     = 32
    }
}
