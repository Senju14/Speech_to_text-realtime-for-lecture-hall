/**
 * Audio Manager
 * Handles microphone/computer audio capture and processing
 */

import { SAMPLE_RATE, BUFFER_SIZE } from './config.js';
import { downsampleBuffer, convertFloat32ToInt16, int16ToBase64, calculateRMS } from './audio-utils.js';

export class AudioManager {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.mediaStream = null;
        this.audioProcessor = null;
        this.sourceNode = null;

        // Callbacks
        this.onAudioData = null;  // Called with processed audio data
        this.onLevelUpdate = null; // Called with audio level for meter

        this.isRecording = false;
        this.sourceSampleRate = 0;
    }

    /**
     * Initialize Audio Context
     */
    async init() {
        if (this.audioContext) return;

        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.analyser.smoothingTimeConstant = 0.8;

        console.log(`[Audio] Initialized. Browser sample rate: ${this.audioContext.sampleRate}Hz`);
        this.sourceSampleRate = this.audioContext.sampleRate;
    }

    /**
     * Get microphone stream
     */
    async getMicrophoneStream() {
        try {
            // Frontend xử lý lọc ồn để giảm tải cho Server
            return await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
        } catch (err) {
            console.error('[Audio] Microphone error:', err);
            throw new Error('Could not access microphone. Please allow permission.');
        }
    }

    /**
     * Get computer audio stream (screen capture)
     */
    async getComputerAudioStream() {
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: true,
                audio: {
                    echoCancellation: false, // Tắt khử vọng để âm thanh hệ thống rõ nhất
                    noiseSuppression: false
                }
            });

            // Stop video track immediately (chỉ lấy tiếng)
            stream.getVideoTracks().forEach(t => t.stop());

            if (stream.getAudioTracks().length === 0) {
                throw new Error('No audio captured. Check "Share audio" option.');
            }

            return stream;
        } catch (err) {
            console.error('[Audio] Computer audio error:', err);
            throw err;
        }
    }

    /**
     * Start recording from source
     * @param {string} source - 'microphone' or 'computer'
     */
    async startRecording(source = 'microphone') {
        await this.init();

        // 👇 FIX QUAN TRỌNG: Resume AudioContext nếu bị suspended
        if (this.audioContext.state === 'suspended') {
            console.log('[Audio] AudioContext suspended, resuming...');
            await this.audioContext.resume();
            console.log('[Audio] AudioContext resumed:', this.audioContext.state);
        }

        // Get stream
        this.mediaStream = source === 'computer'
            ? await this.getComputerAudioStream()
            : await this.getMicrophoneStream();

        // Create source node
        this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.sourceNode.connect(this.analyser);

        // Create script processor for audio data
        // BUFFER_SIZE nên là 4096 (định nghĩa trong config.js)
        this.audioProcessor = this.audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
        this.sourceNode.connect(this.audioProcessor);
        this.audioProcessor.connect(this.audioContext.destination);

        // Process audio
        this.audioProcessor.onaudioprocess = (e) => {
            if (!this.isRecording) return;

            const inputData = e.inputBuffer.getChannelData(0);

            // 1. Downsample (48k/44.1k -> 16k)
            const downsampled = downsampleBuffer(
                inputData,
                this.sourceSampleRate,
                SAMPLE_RATE
            );

            // 2. Convert to Int16 (PCM) & Base64
            const int16 = convertFloat32ToInt16(downsampled);
            const base64 = int16ToBase64(int16);

            // 3. Callback sends data to WebSocket
            if (this.onAudioData) {
                this.onAudioData(base64);
            }

            // 4. Update UI meter
            if (this.onLevelUpdate) {
                const rms = calculateRMS(inputData);
                this.onLevelUpdate(rms);
            }
        };

        this.isRecording = true;
        console.log(`[Audio] Recording started (${source})`);
        console.log(`[Audio] Downsampling: ${this.sourceSampleRate}Hz → ${SAMPLE_RATE}Hz`);
    }

    /**
     * Stop recording and cleanup
     */
    stopRecording() {
        this.isRecording = false;

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor = null;
        }

        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }

        console.log('[Audio] Recording stopped');
    }

    /**
     * Get current frequency data for visualization
     */
    getFrequencyData() {
        if (!this.analyser) return new Uint8Array(0);

        const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteFrequencyData(dataArray);
        return dataArray;
    }
}
