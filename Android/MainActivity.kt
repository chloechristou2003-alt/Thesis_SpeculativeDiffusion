package com.example.speculativediffusionv1

import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import android.view.inputmethod.InputMethodManager
import android.content.Context
import ai.onnxruntime.*
import ai.onnxruntime.providers.NNAPIFlags
import org.eclipse.paho.client.mqttv3.*
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.EnumSet
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

// StatusType -- Enum for the UI state
// Used by updateStatus() to change color/spinner

enum class StatusType { IDLE, LOADING, SUCCESS, ERROR }

// EulerScheduler
// Computes sigmas and timesteps for the Euler Discrete Scheduler.
// Must be in EXACT synchronization with the server (model_loader.py):
//   - timestep_spacing = "leading"
//   - steps_offset = 1

class EulerScheduler(val numSteps: Int) {
    private val sigmasFull = FloatArray(1000)
    private val sigmas     = FloatArray(numSteps + 1)
    private val timesteps  = LongArray(numSteps)
    val initNoiseSigma: Float

    init {
        val betaStart = 0.00085f
        val betaEnd   = 0.012f
        val sqrtStart = Math.sqrt(betaStart.toDouble())
        val sqrtEnd   = Math.sqrt(betaEnd.toDouble())

        // Compute the cumulative product over all 1000 training timesteps
        var cumulativeProduct = 1.0
        for (i in 0 until 1000) {
            val frac  = i.toDouble() / 999.0
            val sqrtB = sqrtStart + frac * (sqrtEnd - sqrtStart)
            val beta  = sqrtB * sqrtB
            cumulativeProduct *= (1.0 - beta)
            sigmasFull[i] = Math.sqrt((1.0 - cumulativeProduct) / cumulativeProduct).toFloat()
        }

        // leading spacing + steps_offset=1 -- same as the server
        val stepRatio = 1000 / numSteps
        for (i in 0 until numSteps) {
            timesteps[numSteps - 1 - i] = (i * stepRatio + 1).toLong()
        }
        for (i in 0 until numSteps) {
            sigmas[i] = interpolateSigma(timesteps[i].toDouble())
        }
        sigmas[numSteps] = 0.0f
        initNoiseSigma = Math.sqrt((sigmas[0] * sigmas[0] + 1.0).toDouble()).toFloat()

        if (BuildConfig.DEBUG) {
            Log.d("SCHED", "sigmas=${sigmas.toList()}")
            Log.d("SCHED", "timesteps=${timesteps.toList()}")
        }
    }

    // Linear interpolation between two neighboring sigmas
    private fun interpolateSigma(t: Double): Float {
        if (t <= 0)   return sigmasFull[0]
        if (t >= 999) return sigmasFull[999]
        val lower = Math.floor(t).toInt()
        val frac  = (t - lower).toFloat()
        return sigmasFull[lower] * (1f - frac) + sigmasFull[lower + 1] * frac
    }

    fun getTimestep(i: Int)  = timesteps[i]
    fun getSigma(i: Int)     = sigmas[i]
    fun getSigmaNext(i: Int) = sigmas[i + 1]

    // Euler step: x_{t+1} = x_t + noise_pred * dt
    fun step(noisePred: FloatArray, latents: FloatArray, stepIdx: Int, out: FloatArray) {
        val dt = getSigmaNext(stepIdx) - getSigma(stepIdx)
        for (i in latents.indices) out[i] = latents[i] + noisePred[i] * dt
    }
}

// VerifyResult
// Verification result from the server:
//   accepted: how many draft steps were accepted (prefix acceptance)
//   state:    the corrected latent state from the target UNet

class VerifyResult(val accepted: Int, val state: FloatArray)

// MainActivity -- Speculative Diffusion with fixed K
//
// Architecture:
//   - Draft:  BK-SDM-tiny ONNX (runs locally on Android)
//   - Target: SD 1.5 (runs on the Kaggle server over MQTT)
//   - Verification: batched forward pass of the target UNet
//   - K: fixed chunk size (CHUNK_SIZE)

class MainActivity : AppCompatActivity() {

    private lateinit var ortEnv: OrtEnvironment
    private var unetSession: OrtSession? = null
    private lateinit var mqttClient: MqttClient

    private val brokerUrl    = "tcp://broker.hivemq.com:1883"
    private val sessionTopic = "speculative/session_2026"

    companion object {
        // Latent space dimensions -- 64x64 = 512x512px output
        private const val LATENT_C    = 4
        private const val LATENT_H    = 32
        private const val LATENT_W    = 32
        private const val LATENT_SIZE = LATENT_C * LATENT_H * LATENT_W  // 16384
        private const val EMBED_SIZE  = 77 * 768                        // CLIP embedding

        // Diffusion hyperparameters
        private const val TOTAL_STEPS = 10      // Total denoising steps
        private const val CHUNK_SIZE  = 4       // Fixed K -- based on log analysis
        private const val CFG_SCALE   = 7.5f    // Classifier-Free Guidance scale
        private const val SEED        = 999L    // Fixed seed for reproducibility

        // Acceptance threshold -- fixed tau (server: tau = ACCEPT_THRESHOLD)
        // 0.30 = optimal for bk-sdm-tiny draft + SD 1.5 target
        private const val ACCEPT_THRESHOLD = 0.30f

        // MQTT timeouts
        private const val VERIFY_TIMEOUT_SEC       = 45L   // Normal verify
        private const val VERIFY_TIMEOUT_FIRST_SEC = 180L  // First verify (server warmup)
        private const val DECODE_TIMEOUT_SEC        = 30L  // VAE decode

        // false = production mode (no intermediate decodes -- much faster)
        // true  = debug mode (decode after each chunk -- slower)
        private const val VISUALIZE_INTERMEDIATE = true
        private const val VISUALIZE_PAUSE_MS     = 250L

        // Positive suffix: encourages a single cohesive scene (prevents split artifacts)
        private const val POSITIVE_SUFFIX = ", single scene, photorealistic, " +
                "high quality, 4k, natural lighting"

        // Negative prompt: prevents collage/split/abstract artifacts
        private const val DEFAULT_NEG_PROMPT = "multiple horizons, double horizon, " +
                "split screen, stacked images, double image, two halves, collage, diptych, " +
                "panel, grid, triptych, composite, montage, " +
                "cartoon, anime, illustration, painting, " +
                "noisy, blurry, deformed, ugly, " +
                "text, watermark, border, frame"
    }

    // Threading & synchronization
    @Volatile private var embeddingsLatch   = CountDownLatch(0)
    @Volatile private var verifyLatch: CountDownLatch? = null
    @Volatile private var verifyResult: VerifyResult?  = null
    @Volatile private var embeddingsReady   = false
    @Volatile private var isGenerating      = false
    @Volatile private var cancelRequested   = false
    @Volatile private var generationThread: Thread? = null

    // Request ID tracking (prevents stale MQTT responses)
    @Volatile private var pendingEmbeddingsId = -1
    @Volatile private var pendingVerifyId     = -1
    @Volatile private var pendingDecodeId     = -1
    @Volatile private var serverErrorMessage: String? = null
    @Volatile private var decodeLatch: CountDownLatch? = null
    @Volatile private var pendingDecodeBitmap: android.graphics.Bitmap? = null
    @Volatile private var decodeFailed = false

    // Cache for embeddings (avoids re-encoding the same prompt)
    private var currentServerModel = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    private var lastCacheKey: Triple<String, String, String>? = null

    // Scheduler and metrics
    private var scheduler: EulerScheduler? = null
    private var totalAccepted = 0
    private var totalK        = 0
    private var generationStartTimeMs = 0L

    // Reusable noise buffers (avoids allocations in the inference loop)
    private val noiseUncondBuffer   = FloatArray(LATENT_SIZE)
    private val noiseCondBuffer     = FloatArray(LATENT_SIZE)
    private val finalNoiseBuffer    = FloatArray(LATENT_SIZE)
    private val scaledLatentsBuffer = FloatArray(LATENT_SIZE)

    // Direct NIO buffers for OnnxTensor (zero-copy)
    private lateinit var sampleBuffer:      FloatBuffer
    private lateinit var timestepBuffer:    FloatBuffer
    private lateinit var uncondEmbedBuffer: FloatBuffer
    private lateinit var condEmbedBuffer:   FloatBuffer

    // Atomic counter for unique request IDs
    private val currentRequestId = java.util.concurrent.atomic.AtomicInteger(0)

    // onCreate -- Application entry point
    // Initializes ORT, MQTT and the button listener

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ortEnv = OrtEnvironment.getEnvironment()
        initReusableBuffers()
        // MQTT on a background thread -- does not block the UI
        Thread { setupMqtt() }.start()

        val generateButton = findViewById<Button>(R.id.generateButton)
        val promptInput    = findViewById<EditText>(R.id.promptInput)

        generateButton.setOnClickListener {
            // If a generation is already running, the button acts as Cancel
            if (isGenerating) {
                cancelRequested = true
                updateStatus("Canceling...", StatusType.LOADING)
                return@setOnClickListener
            }

            val userPrompt = promptInput.text.toString()
            if (userPrompt.isBlank()) {
                updateStatus("Please enter a prompt first!", StatusType.ERROR)
                return@setOnClickListener
            }

            isGenerating          = true
            cancelRequested       = false
            serverErrorMessage    = null
            generateButton.text   = "Cancel"
            updateStatus("Starting...", StatusType.LOADING)

            // Close the keyboard
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.hideSoftInputFromWindow(promptInput.windowToken, 0)

            // Reset metrics for a new generation
            totalAccepted         = 0
            totalK                = 0
            generationStartTimeMs = System.currentTimeMillis()

            // Starts the generation on a background thread
            Thread {
                var generationSucceeded = false
                try {
                    // -- STEP 1: Load the Draft ONNX (only the first time) --------
                    if (unetSession == null) {
                        updateStatus("Loading ONNX Model...", StatusType.LOADING)
                        val unetPath = copyAssetToInternal("draft_unet.onnx")
                        try { copyAssetToInternal("draft_unet.onnx.data") }
                        catch (e: java.io.FileNotFoundException) {
                            Log.d("ORT", "Single-file ONNX (no .data)")
                        }
                        loadUnetModel(unetPath)
                    }

                    // STEP 2: Initialize the Scheduler
                    if (scheduler == null) scheduler = EulerScheduler(TOTAL_STEPS)

                    // STEP 3: Wait for the MQTT connection
                    var waited = 0
                    while (!::mqttClient.isInitialized || !mqttClient.isConnected) {
                        if (waited >= 100) break
                        Thread.sleep(100); waited++
                    }
                    if (!::mqttClient.isInitialized || !mqttClient.isConnected)
                        throw RuntimeException("MQTT not connected — check network")

                    val prompt     = "$userPrompt$POSITIVE_SUFFIX"
                    val negPrompt  = DEFAULT_NEG_PROMPT
                    val currentKey = Triple(prompt, negPrompt, currentServerModel)

                    // STEP 4: Text Embeddings (server-side CLIP, cached)
                    // The server encodes the prompt with CLIP and returns
                    // [uncond_emb, cond_emb] -- cached if the same prompt is reused
                    if (lastCacheKey != currentKey) {
                        updateStatus("Downloading Embeddings...", StatusType.LOADING)
                        embeddingsReady     = false
                        val myEmbId         = currentRequestId.incrementAndGet()
                        pendingEmbeddingsId = myEmbId
                        embeddingsLatch     = CountDownLatch(1)
                        mqttClient.publish(
                            "$sessionTopic/embeddings_request",
                            MqttMessage(encodeEmbeddingsRequest(myEmbId, prompt, negPrompt))
                                .also { it.qos = 1 }
                        )
                        if (!embeddingsLatch.await(60, TimeUnit.SECONDS))
                            throw RuntimeException("Embeddings timeout")
                        lastCacheKey = currentKey
                    } else {
                        if (BuildConfig.DEBUG) Log.d("SPEC", "Reusing cached embeddings")
                    }
                    if (cancelRequested) throw InterruptedException("User cancelled")

                    // STEP 5: Initialize the Latents
                    // Random Gaussian noise, scaled by initNoiseSigma
                    var currentLatents = createRandomLatents(scheduler!!)
                    var nextLatents    = FloatArray(LATENT_SIZE)

                    // Pool for storing the trajectory: [s0, s1, ..., sK]
                    // Size CHUNK_SIZE+1 = 5 (enough for K up to 4)
                    val trajectoryPool = Array(CHUNK_SIZE + 1) { FloatArray(LATENT_SIZE) }

                    var stepIndex = 0

                    // MAIN LOOP -- Speculative Diffusion with fixed K
                    //
                    // Logic:
                    //   1. Sets a fixed K = CHUNK_SIZE
                    //   2. Draft: runs K steps locally with ONNX
                    //   3. Verify: sends the trajectory to the server for checking
                    //   4. Server returns nAccepted + corrected state
                    //   5. stepIndex += nAccepted + (0 or 1 for the fall-forward)

                    while (stepIndex < TOTAL_STEPS) {

                        // Fixed chunk size K = CHUNK_SIZE (= 4)
                        val K = CHUNK_SIZE.coerceAtMost(TOTAL_STEPS - stepIndex)
                        if (K <= 0) break

                        // Start of the pool: store the current state (s0)
                        var poolCount = 0
                        System.arraycopy(currentLatents, 0, trajectoryPool[poolCount++], 0, LATENT_SIZE)

                        updateStatus(
                            "Drafting steps ${stepIndex+1}–${stepIndex+K} [K=$K]...",
                            StatusType.LOADING
                        )
                        if (!embeddingsReady) throw RuntimeException("Missing embeddings")

                        // STEP 6: Draft Inference (locally with ONNX)
                        // Runs K forward passes of the draft UNet
                        // Stores each intermediate state in trajectoryPool
                        for (j in 0 until K) {
                            if (cancelRequested) throw InterruptedException("User cancelled")
                            val noise = runUnetInference(currentLatents, stepIndex + j, scheduler!!)
                            scheduler!!.step(noise, currentLatents, stepIndex + j, nextLatents)
                            val tmp = currentLatents; currentLatents = nextLatents; nextLatents = tmp
                            System.arraycopy(currentLatents, 0, trajectoryPool[poolCount++], 0, LATENT_SIZE)
                        }

                        if (cancelRequested) throw InterruptedException("User cancelled")

                        // STEP 7: Verification (server)
                        // Sends [s0, s1, ..., sK] to the server
                        // The server runs a batched forward pass of the SD 1.5 target
                        // Returns nAccepted (prefix) + corrected target state
                        val myReqId = currentRequestId.incrementAndGet()
                        pendingVerifyId    = myReqId
                        verifyResult       = null
                        serverErrorMessage = null
                        verifyLatch        = CountDownLatch(1)

                        val timeoutSec = if (stepIndex == 0) VERIFY_TIMEOUT_FIRST_SEC
                        else VERIFY_TIMEOUT_SEC

                        updateStatus(
                            if (stepIndex == 0) "Verifying (server warmup)..."
                            else "Verifying [K=$K]...",
                            StatusType.LOADING
                        )

                        mqttClient.publish(
                            "$sessionTopic/verify_request",
                            MqttMessage(
                                encodeVerifyRequest(myReqId, prompt, negPrompt,
                                    trajectoryPool.take(poolCount), stepIndex)
                            ).also { it.qos = 1 }
                        )

                        if (!verifyLatch!!.await(timeoutSec, TimeUnit.SECONDS))
                            throw RuntimeException("Verify timeout (${timeoutSec}s)")

                        serverErrorMessage?.let { err ->
                            serverErrorMessage = null
                            throw RuntimeException(err)
                        }

                        val result    = verifyResult ?: throw RuntimeException("No verify result")
                        val nAccepted = result.accepted
                        val corrected = result.state

                        totalK        += K
                        totalAccepted += nAccepted

                        if (BuildConfig.DEBUG) {
                            Log.d("METRICS", "step=$stepIndex K=$K acc=$nAccepted")
                        }

                        // Fall-forward: we use the target state (never a raw draft)
                        System.arraycopy(corrected, 0, currentLatents, 0, LATENT_SIZE)

                        // If accepted == K -> all correct, we advanced K steps
                        // If accepted < K  -> rejected at step nAccepted, + 1 for the fall-forward
                        stepIndex += if (nAccepted == K) K else nAccepted + 1

                        // Debug visualization
                        if (VISUALIZE_INTERMEDIATE && !cancelRequested) {
                            requestDecodeAndWait(currentLatents)?.let { bmp ->
                                runOnUiThread {
                                    findViewById<ImageView>(R.id.imageView).setImageBitmap(bmp)
                                }
                                Thread.sleep(VISUALIZE_PAUSE_MS)
                            }
                        }
                    }

                    // STEP 8: Final Decode
                    // Sends the final latents to the server for VAE decode -> PNG
                    if (!cancelRequested) {
                        updateStatus("Decoding Final Image...", StatusType.LOADING)
                        val finalBmp = requestDecodeAndWait(currentLatents, timeoutSec = 60L)
                            ?: throw RuntimeException("Final decode failed")
                        runOnUiThread {
                            findViewById<ImageView>(R.id.imageView).setImageBitmap(finalBmp)
                        }

                        val totalMs    = System.currentTimeMillis() - generationStartTimeMs
                        val acceptRate = if (totalK > 0) totalAccepted.toFloat() / totalK else 0f
                        Log.d("METRICS", "DONE | ${totalMs}ms | " +
                                "rate=${"%.3f".format(acceptRate)} ($totalAccepted/$totalK)")
                        updateStatus(
                            "Done in ${totalMs/1000}s | Accept: ${(acceptRate*100).toInt()}%",
                            StatusType.SUCCESS
                        )
                        generationSucceeded = true
                    }

                } catch (e: InterruptedException) {
                    Log.d("SPEC", "Cancelled")
                    updateStatus("Cancelled.", StatusType.ERROR)
                } catch (e: Exception) {
                    Log.e("SPEC_ERROR", "Error: ${e.message}", e)
                    updateStatus("Error: ${e.message}", StatusType.ERROR)
                } finally {
                    isGenerating    = false
                    cancelRequested = false
                    runOnUiThread { generateButton.text = "GENERATE" }
                    hideLoadingSpinner(showCheck = generationSucceeded)
                }
            }.also { generationThread = it; it.start() }
        }
    }

    // updateStatus -- Updates the status TextView and the spinner
    // Always runs on the UI thread via runOnUiThread

    private fun updateStatus(message: String, type: StatusType = StatusType.IDLE) {
        runOnUiThread {
            val tv      = findViewById<TextView>(R.id.statusText)
            val spinner = findViewById<ProgressBar>(R.id.loadingSpinner)
            val check   = findViewById<ImageView>(R.id.successCheck)
            tv?.text = message
            when (type) {
                StatusType.LOADING -> {
                    spinner?.visibility = View.VISIBLE; check?.visibility = View.GONE
                    tv?.setTextColor(android.graphics.Color.LTGRAY)
                }
                StatusType.SUCCESS -> {
                    spinner?.visibility = View.GONE; check?.visibility = View.VISIBLE
                    tv?.setTextColor(android.graphics.Color.parseColor("#4CAF50"))
                }
                StatusType.ERROR -> {
                    spinner?.visibility = View.GONE; check?.visibility = View.GONE
                    tv?.setTextColor(android.graphics.Color.RED)
                }
                StatusType.IDLE -> {
                    spinner?.visibility = View.GONE; check?.visibility = View.GONE
                    tv?.setTextColor(android.graphics.Color.WHITE)
                }
            }
        }
    }

    // hideLoadingSpinner -- Hides the spinner, shows a checkmark on success

    private fun hideLoadingSpinner(showCheck: Boolean) {
        runOnUiThread {
            findViewById<ProgressBar>(R.id.loadingSpinner)?.visibility = View.GONE
            findViewById<ImageView>(R.id.successCheck)?.visibility =
                if (showCheck) View.VISIBLE else View.GONE
        }
    }

    // requestDecodeAndWait -- Sends latents to the server for VAE decode
    // Waits for the PNG response over MQTT (blocking with CountDownLatch)
    // Returns a Bitmap, or null on timeout/error

    private fun requestDecodeAndWait(
        latents: FloatArray, timeoutSec: Long = DECODE_TIMEOUT_SEC
    ): android.graphics.Bitmap? {
        val myId = currentRequestId.incrementAndGet()
        pendingDecodeId     = myId
        pendingDecodeBitmap = null
        decodeFailed        = false
        decodeLatch         = CountDownLatch(1)
        try {
            mqttClient.publish(
                "$sessionTopic/decode_request",
                MqttMessage(encodeDecodeRequest(myId, latents)).also { it.qos = 1 }
            )
        } catch (e: Exception) {
            Log.e("SPEC", "Decode publish failed: ${e.message}"); return null
        }
        if (!decodeLatch!!.await(timeoutSec, TimeUnit.SECONDS)) {
            Log.w("SPEC", "Decode timeout for req $myId"); return null
        }
        if (decodeFailed) { Log.w("SPEC", "Decode failed req $myId"); return null }
        return pendingDecodeBitmap
    }

    // initReusableBuffers -- Creates the Direct NIO buffers once
    // Direct buffers = zero-copy transfer to OnnxTensor (no extra copy)

    private fun initReusableBuffers() {
        sampleBuffer      = ByteBuffer.allocateDirect(LATENT_SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        timestepBuffer    = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        uncondEmbedBuffer = ByteBuffer.allocateDirect(EMBED_SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        condEmbedBuffer   = ByteBuffer.allocateDirect(EMBED_SIZE * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    }

    // setupMqtt -- Connects to the HiveMQ broker and sets the callbacks
    // Runs on a background thread (called from onCreate)
    // Subscriptions: embeddings_response, verify_response, verify_error, decoded

    private fun setupMqtt() {
        try {
            mqttClient = MqttClient(
                brokerUrl,
                "AndroidClient_${System.currentTimeMillis()}",
                MemoryPersistence()
            )
            mqttClient.setCallback(object : MqttCallback {
                // If the connection is lost, tries to reconnect up to 5 times
                override fun connectionLost(cause: Throwable?) {
                    Log.w("MQTT", "Connection lost: ${cause?.message}")
                    if (cancelRequested) return
                    Thread {
                        var attempt = 0
                        while (attempt < 5) {
                            Thread.sleep(minOf(2000L * (attempt + 1), 16_000L))
                            try {
                                if (!mqttClient.isConnected) {
                                    mqttClient.reconnect()
                                    Log.d("MQTT", "Reconnected after ${attempt+1} attempt(s)")
                                    mqttClient.subscribe("$sessionTopic/embeddings_response", 1)
                                    mqttClient.subscribe("$sessionTopic/verify_response",     1)
                                    mqttClient.subscribe("$sessionTopic/verify_error",        1)
                                    mqttClient.subscribe("$sessionTopic/decoded",             1)
                                    if (isGenerating) {
                                        serverErrorMessage = "MQTT reconnected — please retry"
                                        verifyLatch?.countDown()
                                        embeddingsLatch.countDown()
                                    }
                                }
                                return@Thread
                            } catch (e: Exception) {
                                Log.w("MQTT", "Reconnect ${attempt+1} failed: ${e.message}")
                                attempt++
                            }
                        }
                        updateStatus("MQTT offline — check network", StatusType.ERROR)
                    }.start()
                }

                // Router for incoming MQTT messages
                override fun messageArrived(topic: String?, message: MqttMessage?) {
                    val payload = message?.payload ?: return
                    when (topic) {
                        "$sessionTopic/embeddings_response" -> parseEmbeddingsResponse(payload)
                        "$sessionTopic/verify_response"     -> parseVerifyResponse(payload)
                        "$sessionTopic/verify_error"        -> {
                            val buf    = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
                            val respId = buf.int
                            val isOurs = respId == pendingVerifyId ||
                                    (respId == -1 && pendingVerifyId != -1)
                            if (isOurs) {
                                val errBytes = ByteArray(buf.remaining()); buf.get(errBytes)
                                serverErrorMessage = String(errBytes, Charsets.UTF_8)
                                Log.e("SPEC_SERVER_ERROR", serverErrorMessage ?: "")
                                verifyLatch?.countDown()
                            } else {
                                Log.w("SPEC", "Stale verify_error $respId")
                            }
                        }
                        "$sessionTopic/decoded" -> parseDecodedResponse(payload)
                    }
                }

                override fun deliveryComplete(token: IMqttDeliveryToken?) {}
            })

            mqttClient.connect(MqttConnectOptions().apply {
                isCleanSession       = true
                connectionTimeout    = 30
                keepAliveInterval    = 60
                isAutomaticReconnect = true
            })
            mqttClient.subscribe("$sessionTopic/embeddings_response", 1)
            mqttClient.subscribe("$sessionTopic/verify_response",     1)
            mqttClient.subscribe("$sessionTopic/verify_error",        1)
            mqttClient.subscribe("$sessionTopic/decoded",             1)
            Log.d("MQTT", "Connected to $brokerUrl")
        } catch (e: Exception) {
            Log.e("MQTT", "setup error: ${e.message}", e)
            updateStatus("MQTT Error — check network", StatusType.ERROR)
        }
    }

    // parseDecodedResponse -- Decodes the PNG response from the server
    // Protocol: [4 bytes req_id] [PNG bytes]
    // Stores the Bitmap and counts down decodeLatch

    private fun parseDecodedResponse(payload: ByteArray) {
        val buf    = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
        val respId = buf.int
        if (respId != pendingDecodeId) { Log.w("SPEC", "Stale decode $respId"); return }
        val imgBytes = ByteArray(buf.remaining()); buf.get(imgBytes)
        val bitmap = android.graphics.BitmapFactory.decodeByteArray(imgBytes, 0, imgBytes.size)
        if (bitmap != null) {
            pendingDecodeBitmap = bitmap; decodeFailed = false
        } else {
            Log.e("SPEC_ERROR", "Failed to decode PNG")
            pendingDecodeBitmap = null; decodeFailed = true
            runOnUiThread {
                updateStatus("Error: Image data corruption.", StatusType.ERROR)
                findViewById<ImageView>(R.id.imageView)
                    .setBackgroundColor(android.graphics.Color.parseColor("#441111"))
            }
        }
        decodeLatch?.countDown()
    }

    // runUnetInference -- One forward pass of the draft ONNX UNet
    // Runs two passes (uncond + cond) and applies CFG
    //
    // Note: nearly double the time versus a single pass because of CFG
    // Alternative: a batched [uncond, cond] pass, but it needs more RAM

    private fun runUnetInference(
        latents: FloatArray, stepIdx: Int, scheduler: EulerScheduler
    ): FloatArray {
        val t0    = if (BuildConfig.DEBUG) System.currentTimeMillis() else 0L

        // Scaling: x_scaled = x / sqrt(sigma^2 + 1) -- required by the Euler scheduler
        val sigma = scheduler.getSigma(stepIdx)
        val scale = Math.sqrt((sigma * sigma + 1.0).toDouble()).toFloat()
        for (i in latents.indices) scaledLatentsBuffer[i] = latents[i] / scale

        // Fill the reusable buffers
        sampleBuffer.clear();   sampleBuffer.put(scaledLatentsBuffer);  sampleBuffer.rewind()
        timestepBuffer.clear(); timestepBuffer.put(scheduler.getTimestep(stepIdx).toFloat()); timestepBuffer.rewind()

        val sampleTen   = OnnxTensor.createTensor(ortEnv, sampleBuffer,
            longArrayOf(1, LATENT_C.toLong(), LATENT_H.toLong(), LATENT_W.toLong()))
        val timestepTen = OnnxTensor.createTensor(ortEnv, timestepBuffer, longArrayOf(1))

        try {
            // Pass 1: Unconditional (no text conditioning)
            uncondEmbedBuffer.rewind()
            val uncondTen = OnnxTensor.createTensor(ortEnv, uncondEmbedBuffer, longArrayOf(1, 77, 768))
            try {
                unetSession?.run(mapOf("sample" to sampleTen, "timestep" to timestepTen,
                    "encoder_hidden_states" to uncondTen))
                    ?.use { (it.get(0) as OnnxTensor).floatBuffer.get(noiseUncondBuffer) }
            } finally { uncondTen.close() }

            // Pass 2: Conditional (with text conditioning)
            condEmbedBuffer.rewind()
            val condTen = OnnxTensor.createTensor(ortEnv, condEmbedBuffer, longArrayOf(1, 77, 768))
            try {
                unetSession?.run(mapOf("sample" to sampleTen, "timestep" to timestepTen,
                    "encoder_hidden_states" to condTen))
                    ?.use { (it.get(0) as OnnxTensor).floatBuffer.get(noiseCondBuffer) }
            } finally { condTen.close() }

        } finally { sampleTen.close(); timestepTen.close() }

        // CFG: noise_guided = uncond + scale * (cond - uncond)
        for (i in latents.indices)
            finalNoiseBuffer[i] = noiseUncondBuffer[i] + CFG_SCALE * (noiseCondBuffer[i] - noiseUncondBuffer[i])

        if (BuildConfig.DEBUG)
            Log.d("PERF", "step=$stepIdx unet=${System.currentTimeMillis()-t0}ms")
        return finalNoiseBuffer
    }

    // parseEmbeddingsResponse -- Receives CLIP embeddings from the server
    // Protocol: [4 req_id][4 batch][4 tokens][4 dim][floats...]
    // Stores [uncond, cond] in the corresponding buffers

    private fun parseEmbeddingsResponse(payload: ByteArray) {
        val buf        = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
        val responseId = buf.int
        if (responseId != pendingEmbeddingsId) {
            Log.w("SPEC", "Stale embeddings $responseId"); return
        }
        val b = buf.int; val t = buf.int; val d = buf.int
        if (b != 2 || t != 77 || d != 768) {
            Log.e("SPEC", "Bad embeddings shape ($b,$t,$d)"); return
        }
        val embSize = t * d
        // The server sends: [uncond_emb (77x768), cond_emb (77x768)]
        val uncond = FloatArray(embSize) { buf.float }
        val cond   = FloatArray(embSize) { buf.float }
        uncondEmbedBuffer.clear(); uncondEmbedBuffer.put(uncond); uncondEmbedBuffer.rewind()
        condEmbedBuffer.clear();   condEmbedBuffer.put(cond);     condEmbedBuffer.rewind()
        pendingEmbeddingsId = -1  // Prevents processing a stale response
        embeddingsReady     = true
        embeddingsLatch.countDown()
    }

    // parseVerifyResponse -- Receives the verification result from the server
    // Protocol: [4 req_id][4 nAccepted][4 c][4 h][4 w][floats state]
    //           [4 n][n floats cos_sims][4 n][n floats rel_l2_x]
    //           [4 n][n floats rel_l2_eps]

    private fun parseVerifyResponse(payload: ByteArray) {
        val buf    = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
        val respId = buf.int
        if (respId != pendingVerifyId) { Log.w("SPEC", "Stale verify $respId"); return }
        val nAcc = buf.int; val c = buf.int; val h = buf.int; val w = buf.int
        val localState = FloatArray(c * h * w) { buf.float }
        val cosSims  = readMetricBlock(buf)
        val relL2X   = readMetricBlock(buf)
        val relL2Eps = readMetricBlock(buf)
        if (BuildConfig.DEBUG) {
            Log.d("SPEC", "accepted=$nAcc/${cosSims?.size ?: -1}")
            relL2Eps?.let { Log.d("SPEC_METRICS", "rel_l2_eps=${it.toList()}") }
        }
        verifyResult = VerifyResult(nAcc, localState)
        verifyLatch?.countDown()
    }

    // readMetricBlock -- Helper for reading a metric array from a ByteBuffer
    // Format: [4 bytes length][length * 4 bytes floats]
    // Returns null if there are not enough bytes

    private fun readMetricBlock(buf: ByteBuffer): FloatArray? {
        if (buf.remaining() < 4) return null
        val n = buf.int
        if (n < 0 || buf.remaining() < n * 4) return null
        return FloatArray(n) { buf.float }
    }

    // encodeVerifyRequest -- Encodes the verify request to binary
    // Must match exactly parse_verify_request()
    // on the server (server_speculative.py)
    //
    // Format: [4 req_id][2+N prompt][2+N neg][4 stepIdx][4 K][4 totalSteps]
    //         [4 cfg][4 threshold][8 seed][4 c][4 h][4 w][states floats]

    private fun encodeVerifyRequest(
        reqId: Int, prompt: String, negPrompt: String,
        trajectory: List<FloatArray>, stepIdx: Int
    ): ByteArray {
        val pBytes    = prompt.toByteArray(Charsets.UTF_8)
        val nBytes    = negPrompt.toByteArray(Charsets.UTF_8)
        val totalSize = 4 + 2+pBytes.size + 2+nBytes.size + 40 + trajectory.size*LATENT_SIZE*4
        val buf = ByteBuffer.allocate(totalSize).order(ByteOrder.BIG_ENDIAN)
        buf.putInt(reqId)
        buf.putShort(pBytes.size.toShort()).put(pBytes)
        buf.putShort(nBytes.size.toShort()).put(nBytes)
        buf.putInt(stepIdx)
            .putInt(trajectory.size - 1)  // K = states - 1
            .putInt(TOTAL_STEPS)
            .putFloat(CFG_SCALE)
            .putFloat(ACCEPT_THRESHOLD)
            .putLong(SEED)
            .putInt(LATENT_C).putInt(LATENT_H).putInt(LATENT_W)
        for (state in trajectory) { for (v in state) buf.putFloat(v) }
        return buf.array()
    }

    // encodeEmbeddingsRequest -- Encodes the embeddings request
    // Must match _handle_embeddings() on the server
    //
    // Format: [4 req_id][2+N prompt][2+N neg][2+N model_id]

    private fun encodeEmbeddingsRequest(reqId: Int, prompt: String, neg: String): ByteArray {
        val pBytes = prompt.toByteArray(Charsets.UTF_8)
        val nBytes = neg.toByteArray(Charsets.UTF_8)
        val mBytes = currentServerModel.toByteArray(Charsets.UTF_8)
        val buf = ByteBuffer.allocate(4 + 2+pBytes.size + 2+nBytes.size + 2+mBytes.size)
            .order(ByteOrder.BIG_ENDIAN)
        buf.putInt(reqId)
        buf.putShort(pBytes.size.toShort()).put(pBytes)
        buf.putShort(nBytes.size.toShort()).put(nBytes)
        buf.putShort(mBytes.size.toShort()).put(mBytes)
        return buf.array()
    }

    // encodeDecodeRequest -- Encodes the decode request (latents -> PNG)
    // Must match parse_decode_request() on the server
    //
    // Format: [4 req_id][4 c][4 h][4 w][latents floats]

    private fun encodeDecodeRequest(reqId: Int, latents: FloatArray): ByteArray {
        val buf = ByteBuffer.allocate(4 + 12 + latents.size*4).order(ByteOrder.BIG_ENDIAN)
        buf.putInt(reqId)
        buf.putInt(LATENT_C).putInt(LATENT_H).putInt(LATENT_W)
        for (v in latents) buf.putFloat(v)
        return buf.array()
    }

    // createRandomLatents -- Creates the initial Gaussian noise
    // Scaled by initNoiseSigma as required by the Euler scheduler
    // Fixed seed for reproducible results

    private fun createRandomLatents(scheduler: EulerScheduler): FloatArray {
        val random = java.util.Random(SEED)
        return FloatArray(LATENT_SIZE) { random.nextGaussian().toFloat() * scheduler.initNoiseSigma }
    }

    // loadUnetModel -- Loads the ONNX model and runs a warmup
    //
    // SessionOptions:
    //   - 4 threads
    //   - ALL_OPT
    //
    // Warmup: dummy forward pass for JIT compilation before the first verify
    // Without warmup, the first inference takes 30-60s or crashes

    private fun loadUnetModel(path: String) {
        // Try NNAPI first (offloads supported ops to device NPU/GPU/DSP if present).
        // ORT partitions the graph automatically -- unsupported ops fall back to CPU.
        // NNAPI support/perf varies a lot by device, so if session creation fails
        // for any reason we retry CPU-only rather than crashing generation.
        unetSession = try {
            val nnapiOpts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }
            ortEnv.createSession(path, nnapiOpts).also {
                Log.d("ORT", "Draft UNet loaded with NNAPI EP")
            }
        } catch (e: Exception) {
            Log.w("ORT", "NNAPI EP unavailable (${e.message}), falling back to CPU-only")
            val cpuOpts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            ortEnv.createSession(path, cpuOpts).also {
                Log.d("ORT", "Draft UNet loaded with CPU EP")
            }
        }

        updateStatus("Warming up model...", StatusType.LOADING)
        try {
            val t0 = System.currentTimeMillis()
            val dummySample = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(FloatArray(LATENT_SIZE)),
                longArrayOf(1, LATENT_C.toLong(), LATENT_H.toLong(), LATENT_W.toLong())
            )
            val dummyTimestep = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(floatArrayOf(1.0f)),
                longArrayOf(1)
            )
            val dummyEncoder = OnnxTensor.createTensor(
                ortEnv,
                FloatBuffer.wrap(FloatArray(77 * 768)),
                longArrayOf(1, 77, 768)
            )
            try {
                unetSession?.run(mapOf(
                    "sample"                to dummySample,
                    "timestep"              to dummyTimestep,
                    "encoder_hidden_states" to dummyEncoder
                ))?.close()
            } finally {
                dummySample.close()
                dummyTimestep.close()
                dummyEncoder.close()
            }
            Log.d("ORT", "Warmup complete in ${System.currentTimeMillis() - t0}ms")
        } catch (e: Exception) {
            Log.w("ORT", "Warmup failed (non-fatal): ${e.message}")
        }
    }

    // copyAssetToInternal -- Copies an asset to internal storage
    // Required because ORT cannot read directly from assets
    // If the file already exists, it skips the copy

    private fun copyAssetToInternal(f: String): String {
        val file = File(filesDir, f)
        if (!file.exists()) assets.open(f).use { i -> FileOutputStream(file).use { i.copyTo(it) } }
        return file.absolutePath
    }

    // onDestroy -- Cleans up resources when the application closes
    // Order: cancel -> interrupt thread -> disconnect MQTT -> close ORT

    override fun onDestroy() {
        super.onDestroy()
        cancelRequested = true
        generationThread?.interrupt()
        try { generationThread?.join(5000) } catch (e: InterruptedException) {}
        try { if (::mqttClient.isInitialized && mqttClient.isConnected) mqttClient.disconnect() } catch (e: Exception) {}
        try { unetSession?.close() } catch (e: Exception) {}
        try { if (::ortEnv.isInitialized) ortEnv.close() } catch (e: Exception) {}
    }
}
