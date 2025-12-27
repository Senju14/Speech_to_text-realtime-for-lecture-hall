/**
 * Audio utility functions
 * Critical: Downsampling from browser sample rate to 16kHz
 */

import { SAMPLE_RATE } from './config.js';

/**
 * Downsample audio buffer from source rate to target rate (16kHz)
 * This is CRITICAL for ASR quality - browser uses 44.1kHz or 48kHz by default
 * * @param {Float32Array} buffer - Input audio buffer
 * @param {number} fromRate - Source sample rate (e.g., 48000)
 * @param {number} toRate - Target sample rate (16000)
 * @returns {Float32Array} Downsampled buffer
 */
export function downsampleBuffer(buffer, fromRate, toRate = SAMPLE_RATE) {
    if (fromRate === toRate) {
        return buffer;
    }

    const sampleRateRatio = fromRate / toRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);

    let offsetResult = 0;
    let offsetBuffer = 0;

    while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);

        // Average all samples in this window to prevent aliasing (Simple Low-pass)
        let accum = 0;
        let count = 0;

        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }

        // Avoid division by zero
        if (count > 0) {
            result[offsetResult] = accum / count;
        } else {
            result[offsetResult] = 0; // Should rarely happen
        }

        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }

    return result;
}

/**
 * Convert Float32Array to Int16Array for transmission
 * Range: [-1.0, 1.0] → [-32768, 32767]
 * * @param {Float32Array} float32Array
 * @returns {Int16Array}
 */
export function convertFloat32ToInt16(float32Array) {
    const int16 = new Int16Array(float32Array.length);

    for (let i = 0; i < float32Array.length; i++) {
        // Clamp values to [-1, 1] to prevent distortion
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        // Convert to PCM 16-bit
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    return int16;
}

/**
 * Convert Int16Array to Base64 string for WebSocket transmission
 * Optimized to prevent "Maximum call stack size exceeded" errors
 * * @param {Int16Array} int16Array
 * @returns {string} Base64 encoded string
 */
export function int16ToBase64(int16Array) {
    let binary = '';
    const bytes = new Uint8Array(int16Array.buffer);
    const len = bytes.byteLength;

    // Process in chunks to avoid stack overflow with spread operator
    // or simply loop through (slower but safer). 
    // Given buffer size is small (~8KB), simple loop is fine and safe.
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }

    return btoa(binary);
}

/**
 * Calculate RMS (Root Mean Square) energy of audio buffer
 * Useful for voice activity detection visualization
 * * @param {Float32Array} buffer
 * @returns {number} RMS energy value (0.0 to 1.0)
 */
export function calculateRMS(buffer) {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
        sum += buffer[i] * buffer[i];
    }
    return Math.sqrt(sum / buffer.length);
}
