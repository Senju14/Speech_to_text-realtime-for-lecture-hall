/**
 * Configuration constants for ASR Frontend
 */

// WebSocket URL - Auto-detect based on current location
// Works for both Local (ws://) and Cloud Modal (wss://)
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
export const WS_URL = `${wsProtocol}//${window.location.host}/ws/transcribe`;

// Audio settings
export const SAMPLE_RATE = 16000;      // Model Whisper expects 16kHz
export const BUFFER_SIZE = 4096;       // Audio processing buffer size
export const CHUNK_DURATION_MS = 500;  // Target chunk duration (approx)

// WebSocket settings
export const WS_RECONNECT_DELAY_MS = 2000;  // Wait 2s before reconnecting
export const WS_MAX_RETRIES = 5;            // Increased to 5 for Cloud Cold-starts
export const WS_TIMEOUT_MS = 30000;         // 30s connection timeout (Modal needs time to boot)
