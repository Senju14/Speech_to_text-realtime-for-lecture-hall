import { AudioManager } from './audio.js';
import { SocketManager } from './socket.js';
import { UIManager } from './ui.js';
import { exportSRT, exportVTT, exportTXT, exportJSON } from './export.js';

class ASRApp {
    constructor() {
        this.audioMgr = new AudioManager();
        this.socketMgr = new SocketManager();
        this.uiMgr = new UIManager();
        this.isRecording = false;
        this.isConnecting = false;
        this.serverReady = false;
        this.transcripts = {};
        this.sessionId = null;
    }

    async init() {
        this.setupAudioCallbacks();
        this.setupSocketCallbacks();
        this.setupUIHandlers();
        this.setupExportHandlers();
        this.uiMgr.updateRecordButton(false);
        await this.loadMicrophones();
        await this.checkServerStatus();
    }

    async loadMicrophones() {
        try {
            const mics = await this.audioMgr.getAvailableMicrophones();
            const micList = document.getElementById('micList');
            if (micList && mics.length > 0) {
                micList.innerHTML = mics.map((mic, i) => `
                    <div class="dropdown-item mic-item" data-source="microphone" data-device="${mic.deviceId}">
                        ${mic.label}
                    </div>
                `).join('');
            }
        } catch (e) {
            console.log('[Mics] Could not enumerate devices');
        }
    }

    async checkServerStatus() {
        try {
            this.uiMgr.showNotification('Connecting to server...', 'info');
            const res = await fetch('/api/status');
            if (res.ok) {
                const data = await res.json();
                this.serverReady = true;
                this.uiMgr.showNotification(`Ready • ${data.model || 'Whisper'} on ${data.gpu || 'GPU'}`, 'success');
            }
        } catch (e) {
            this.uiMgr.showNotification('Server is starting up...', 'warning');
        }
    }

    setupAudioCallbacks() {
        // Receives raw ArrayBuffer from AudioManager (binary, not base64)
        this.audioMgr.onAudioData = (arrayBuffer) => {
            if (this.isRecording && this.socketMgr.isConnected()) {
                this.socketMgr.sendBinary(arrayBuffer);
            }
        };
        this.audioMgr.onLevelUpdate = (rms) => {
            if (this.isRecording) this.uiMgr.updateAudioMeter(rms);
        };
    }

    setupSocketCallbacks() {
        this.socketMgr.onMessage = (data) => {
            if (data.type === 'transcript') {
                this.uiMgr.addTranscriptSegment(data);

                if (data.is_final && (data.source || data.target)) {
                    const id = data.segment_id || Object.keys(this.transcripts).length + 1;
                    if (!this.transcripts[id]) this.transcripts[id] = { vi: '', en: '' };
                    if (data.source) this.transcripts[id].vi = data.source;
                    if (data.target) this.transcripts[id].en = data.target;
                }
            } else if (data.type === 'summary') {
                this.uiMgr.showSummary(data);
            } else if (data.type === 'log') {
                // Structured log messages from backend
                const level = data.level || 'info';
                this.uiMgr.showNotification(data.message, level);
            } else if (data.type === 'status') {
                if (data.status === 'started') {
                    const msg = data.primed
                        ? `Recording • Context primed with keywords`
                        : 'Recording...';
                    this.uiMgr.showNotification(msg, 'success');
                } else if (data.status === 'stopped') {
                    if (data.metrics) {
                        const m = data.metrics;
                        const confPct = Math.round(m.avg_confidence * 100);
                        this.uiMgr.showNotification(
                            `Stopped • ${m.segments} segments • ${confPct}% confidence`,
                            'info'
                        );
                    } else {
                        this.uiMgr.showNotification('Recording stopped', 'info');
                    }
                }
            }
        };

        this.socketMgr.onConnected = () => {
            this.isConnecting = false;
            this.serverReady = true;
        };

        this.socketMgr.onDisconnected = () => {
            if (this.isRecording) {
                this.uiMgr.showNotification('Connection lost', 'error');
                this.stopRecording();
            }
        };

        this.socketMgr.onError = () => {
            this.isConnecting = false;
            this.uiMgr.showNotification('Connection error', 'error');
        };
    }

    setupUIHandlers() {
        if (this.uiMgr.el.recordBtn) {
            this.uiMgr.el.recordBtn.onclick = () => {
                if (this.isRecording) this.stopRecording();
                else this.startRecording();
            };
        }

        this.uiMgr.onContextSave = (data) => {
            if (this.socketMgr.isConnected()) {
                this.socketMgr.send('context', data);
                this.uiMgr.showNotification('Context saved');
            }
        };

        // Manual Summarize button
        const summarizeBtn = document.getElementById('summarizeBtn');
        if (summarizeBtn) {
            summarizeBtn.onclick = () => {
                const segments = Object.values(this.transcripts).filter(t => t.vi || t.en);
                if (segments.length === 0) {
                    this.uiMgr.showNotification('No transcripts to summarize', 'warning');
                    return;
                }
                if (this.socketMgr.isConnected()) {
                    this.socketMgr.send('summarize');
                    this.uiMgr.showNotification('Generating summary...', 'info');
                } else {
                    this.uiMgr.showNotification('Not connected to server', 'error');
                }
            };
        }

        // Microphone selection
        const audioDropdown = document.getElementById('audioDropdown');
        if (audioDropdown) {
            audioDropdown.addEventListener('click', (e) => {
                const item = e.target.closest('[data-source]');
                if (!item) return;

                const source = item.dataset.source;
                const deviceId = item.dataset.device;

                if (source === 'microphone' && deviceId) {
                    this.audioMgr.setMicrophone(deviceId);
                    this.uiMgr.setAudioSource('microphone');
                    const label = item.textContent.trim();
                    document.getElementById('audioSourceText').textContent = label.substring(0, 20);
                } else if (source === 'computer') {
                    this.uiMgr.setAudioSource('computer');
                }

                audioDropdown.classList.remove('active');
            });
        }
    }

    setupExportHandlers() {
        const exportBtn = document.getElementById('exportBtn');
        const exportDropdown = document.getElementById('exportDropdown');

        if (exportBtn) {
            exportBtn.onclick = (e) => {
                e.stopPropagation();
                exportDropdown?.classList.toggle('active');
            };
        }

        if (exportDropdown) {
            exportDropdown.querySelectorAll('[data-export]').forEach(item => {
                item.onclick = () => {
                    const format = item.dataset.export;
                    const segments = Object.values(this.transcripts).filter(t => t.vi || t.en);

                    if (segments.length === 0) {
                        this.uiMgr.showNotification('No transcripts');
                        return;
                    }

                    switch (format) {
                        case 'srt': exportSRT(segments); break;
                        case 'vtt': exportVTT(segments); break;
                        case 'txt': exportTXT(segments); break;
                        case 'json': exportJSON(segments); break;
                    }

                    this.uiMgr.showNotification(`Exported ${format.toUpperCase()}`, 'success');
                    exportDropdown.classList.remove('active');
                };
            });
        }

        document.addEventListener('click', () => {
            exportDropdown?.classList.remove('active');
        });
    }

    async startRecording() {
        if (this.isConnecting) return;
        if (!this.serverReady) {
            this.uiMgr.showNotification('Server is starting up...', 'warning');
            await this.checkServerStatus();
            if (!this.serverReady) return;
        }

        this.isConnecting = true;
        this.uiMgr.updateRecordButton('connecting');

        try {
            if (!this.socketMgr.isConnected()) {
                await this.socketMgr.connect();
            }

            const source = this.uiMgr.getAudioSource();
            await this.audioMgr.startRecording(source);

            this.uiMgr.clearTranscripts();
            this.transcripts = {};
            this.sessionId = Date.now();

            const langSettings = this.uiMgr.getLanguageSettings();
            const topic = this.uiMgr.getLectureTopic();
            this.socketMgr.send('start', { ...langSettings, topic });

            this.isRecording = true;
            this.isConnecting = false;
            this.uiMgr.startTimer();
            this.uiMgr.updateRecordButton('recording');

        } catch (err) {
            this.isConnecting = false;
            this.isRecording = false;
            this.uiMgr.updateRecordButton('idle');
            this.uiMgr.resetAudioMeter();

            // Map browser errors to friendly messages
            const friendly = this._getFriendlyError(err);
            this.uiMgr.showNotification(friendly, 'error');
        }
    }

    _getFriendlyError(err) {
        const name = err.name || '';
        const msg = err.message || '';

        if (name === 'NotAllowedError' || msg.includes('Permission denied')) {
            return 'Microphone access denied. Please allow microphone permission and try again.';
        }
        if (name === 'NotFoundError' || msg.includes('Requested device not found')) {
            return 'No microphone found. Please connect a microphone.';
        }
        if (name === 'NotReadableError' || msg.includes('Could not start')) {
            return 'Microphone is in use by another application.';
        }
        if (msg.includes('timeout') || msg.includes('Timeout')) {
            return 'Connection timed out. Please try again.';
        }
        if (msg.includes('No audio')) {
            return 'No audio detected. Make sure "Share audio" is enabled.';
        }
        return msg || 'Could not start recording';
    }

    async stopRecording() {
        if (this.socketMgr.isConnected()) {
            this.socketMgr.send('stop');
        }

        this.audioMgr.stopRecording();

        this.isRecording = false;
        this.isConnecting = false;
        this.uiMgr.stopTimer();
        this.uiMgr.resetAudioMeter();
        this.uiMgr.updateRecordButton('idle');

        // Save recording after short delay
        await new Promise(r => setTimeout(r, 500));
        this.saveRecording();
    }

    saveRecording() {
        const segments = Object.values(this.transcripts).filter(t => t.vi || t.en);
        if (segments.length === 0) return;

        try {
            const recordings = JSON.parse(localStorage.getItem('recordings') || '[]');
            const now = new Date();

            const rec = {
                id: this.sessionId,
                date: now.toLocaleDateString('en-CA').replace(/-/g, '.'),
                time: now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }),
                duration: this.uiMgr.el.timer?.textContent || '00:00',
                transcript: segments.map(t => ({
                    vi: t.vi || '',
                    en: t.en || '',
                    source: t.vi || '',
                    target: t.en || ''
                }))
            };

            recordings.unshift(rec);
            if (recordings.length > 50) recordings.pop();

            localStorage.setItem('recordings', JSON.stringify(recordings));
            this.uiMgr.showNotification(`Saved ${segments.length} segments`);
            this.uiMgr.loadRecordingsSidebar();
        } catch (e) {
            console.error('[Save] Error:', e);
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    const app = new ASRApp();
    app.init();
});