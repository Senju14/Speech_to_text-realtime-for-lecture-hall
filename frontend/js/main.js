/**
 * ASR Frontend - Main Entry Point
 * Vietnamese Speech Recognition with Downsampling
 */

import { AudioManager } from './audio-mgr.js';
import { SocketManager } from './socket-mgr.js';
import { UIManager } from './ui-mgr.js';

class ASRApp {
    constructor() {
        this.audioMgr = new AudioManager();
        this.socketMgr = new SocketManager();
        this.uiMgr = new UIManager();

        this.isRecording = false;
        this.isConnecting = false;  // Track connection state

        // Cache để lưu transcript tạm thời -> Save vào localStorage khi stop
        this.currentSessionTranscript = [];
        this.currentSessionId = null;
    }

    /**
     * Initialize application
     */
    async init() {
        console.log('✅ ASR Frontend initialized');

        // Setup callbacks
        this.setupAudioCallbacks();
        this.setupSocketCallbacks();
        this.setupUIHandlers();

        // Restore context if needed
        // this.restoreSettings();

        console.log('✅ All callbacks registered');
    }

    /**
     * Setup audio manager callbacks
     */
    setupAudioCallbacks() {
        // When audio data is ready, send via WebSocket
        this.audioMgr.onAudioData = (base64Audio) => {
            // Chỉ gửi khi đang recording thực sự
            if (this.isRecording) {
                this.socketMgr.send('audio', { audio: base64Audio });
            }
        };

        // Update UI meter
        this.audioMgr.onLevelUpdate = (rms) => {
            this.uiMgr.updateAudioMeter(rms);
        };
    }

    /**
     * Setup WebSocket callbacks
     */
    setupSocketCallbacks() {
        // Handle incoming messages
        this.socketMgr.onMessage = (data) => {
            // console.log('[App] Message:', data.type); // Comment bớt cho đỡ spam

            if (data.type === 'transcript') {
                // 1. Update UI
                this.uiMgr.addTranscriptSegment(data);

                // 2. Update Cache (chỉ lưu những đoạn có nội dung)
                if (data.source || data.target) {
                    this.updateTranscriptCache(data);
                }

            } else if (data.type === 'status' && data.status === 'error') {
                console.error('[App] Server error:', data.message);
                this.uiMgr.showNotification('Server error: ' + data.message);
            }
        };

        this.socketMgr.onConnected = () => {
            console.log('✅ [App] WebSocket connected');
            this.isConnecting = false;
            this.uiMgr.showNotification('Connected to server');
        };

        this.socketMgr.onDisconnected = () => {
            console.log('⚠️ [App] WebSocket disconnected');
            if (this.isRecording) {
                this.uiMgr.showNotification('Connection lost. Stopping...');
                this.stopRecording();
            }
        };

        this.socketMgr.onError = (err) => {
            console.error('❌ [App] WebSocket error:', err);
            this.isConnecting = false;
        };
    }

    /**
     * Setup UI event handlers
     */
    setupUIHandlers() {
        // Record button
        this.uiMgr.elements.recordBtn.onclick = () => {
            if (this.isRecording) {
                this.stopRecording();
            } else {
                this.startRecording();
            }
        };

        // Context Save Handler (Nối UI với Socket)
        this.uiMgr.onContextSave = (data) => {
            if (this.socketMgr.isConnected()) {
                this.socketMgr.send('context', data);
            } else {
                // Nếu chưa connect thì connect trước rồi gửi (hoặc lưu tạm)
                console.warn('Socket not connected, context will be sent on start');
                // TODO: Save to temp storage
            }
        };
    }

    /**
     * Update transcript cache for saving later
     */
    updateTranscriptCache(data) {
        // Tìm xem segment này đã có trong cache chưa
        const index = this.currentSessionTranscript.findIndex(t => t.segment_id === data.segment_id);

        const entry = {
            segment_id: data.segment_id,
            timestamp: data.timestamp,
            vi: data.source,
            en: data.target,
            is_final: data.is_final
        };

        if (index !== -1) {
            this.currentSessionTranscript[index] = entry;
        } else {
            this.currentSessionTranscript.push(entry);
        }
    }

    /**
     * Start recording
     */
    async startRecording() {
        if (this.isConnecting) return;
        this.isConnecting = true;

        try {
            // 1. Connect WebSocket if needed
            if (!this.socketMgr.isConnected()) {
                this.uiMgr.updateRecordButton(true); // Fake active state indicating loading
                await this.socketMgr.connect();
            }

            // 2. Get Audio Source & Permission
            const source = this.uiMgr.getAudioSource();
            await this.audioMgr.startRecording(source);

            // 3. Prepare Session Data
            this.uiMgr.clearTranscripts();
            this.currentSessionTranscript = [];
            this.currentSessionId = Date.now();

            // 4. Send Start Signal
            this.socketMgr.send('start');

            // 5. Update UI
            this.isRecording = true;
            this.isConnecting = false;
            this.uiMgr.startTimer();
            this.uiMgr.updateRecordButton(true);

            console.log('✅ Recording started');

        } catch (err) {
            console.error('Start failed:', err);
            this.isConnecting = false;
            this.isRecording = false;
            this.uiMgr.updateRecordButton(false);

            let msg = 'Could not start recording.';
            if (err.name === 'NotAllowedError') msg = 'Microphone permission denied.';
            if (err.name === 'NotFoundError') msg = 'Microphone not found.';

            alert(msg);
        }
    }

    /**
     * Stop recording
     */
    async stopRecording() {
        console.log('⏹️ Stopping...');

        // 1. Send stop signal
        if (this.socketMgr.isConnected()) {
            this.socketMgr.send('stop');
        }

        // 2. Stop audio
        this.audioMgr.stopRecording();

        // 3. Update UI
        this.isRecording = false;
        this.isConnecting = false;
        this.uiMgr.stopTimer();
        this.uiMgr.resetAudioMeter();
        this.uiMgr.updateRecordButton(false);

        // 4. Save Recording to LocalStorage
        this.saveRecordingToStorage();
    }

    /**
     * Save current session to LocalStorage for History Page
     */
    saveRecordingToStorage() {
        console.log('[Save] Attempting to save. Cache length:', this.currentSessionTranscript.length);

        if (this.currentSessionTranscript.length === 0) {
            console.log('[Save] No transcripts to save');
            return;
        }

        // Filter empty segments
        const validTranscripts = this.currentSessionTranscript.filter(t => t.vi || t.en);
        console.log('[Save] Valid transcripts:', validTranscripts.length);

        if (validTranscripts.length === 0) {
            console.log('[Save] All transcripts were empty');
            return;
        }

        try {
            const recordings = JSON.parse(localStorage.getItem('recordings') || '[]');

            const now = new Date();
            const dateStr = now.toLocaleDateString('en-CA').replace(/-/g, '.'); // YYYY.MM.DD
            const timeStr = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

            const newRecording = {
                id: this.currentSessionId,
                date: dateStr,
                time: timeStr,
                duration: this.uiMgr.elements.timer.textContent,
                transcript: validTranscripts
            };

            // Add to beginning
            recordings.unshift(newRecording);

            // Limit history to 20 items
            if (recordings.length > 20) recordings.pop();

            localStorage.setItem('recordings', JSON.stringify(recordings));
            console.log('✅ Recording saved to history:', newRecording);

            // Notify user and refresh sidebar
            this.uiMgr.showNotification(`Recording saved (${validTranscripts.length} segments)`);
            this.uiMgr.loadRecordingsSidebar();

        } catch (e) {
            console.error('Failed to save recording:', e);
        }
    }
}

// Initialize app when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing ASR App...');
    const app = new ASRApp();
    app.init();
});
