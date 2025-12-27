/**
 * UI Manager - Handles DOM updates and Bilingual Display
 */

export class UIManager {
    constructor() {
        this.elements = {
            // Dropdown
            audioSourceBtn: document.getElementById('audioSourceBtn'),
            audioSourceText: document.getElementById('audioSourceText'),
            audioDropdown: document.getElementById('audioDropdown'),

            // Audio meter
            audioMeter: document.getElementById('audioMeter'),

            // Timer & Record
            timer: document.getElementById('timer'),
            recordBtn: document.getElementById('recordBtn'),

            // Panel
            panelContent: document.getElementById('panelContent'),

            // Actions
            fontDecrease: document.getElementById('fontDecrease'),
            fontIncrease: document.getElementById('fontIncrease'),
            contextBtn: document.getElementById('contextBtn'),
            copyBtn: document.getElementById('copyBtn'),
            clearBtn: document.getElementById('clearBtn'), // Added clear button

            // Sidebar
            sidebar: document.getElementById('sidebar'),
            sidebarToggle: document.getElementById('sidebarToggle'),
            recordingsList: document.getElementById('recordingsList'),

            // Modal
            contextModal: document.getElementById('contextModal'),
            closeModal: document.getElementById('closeModal'),
            keywords: document.getElementById('keywords'),
            context: document.getElementById('context'),
            clearContextBtn: document.getElementById('clearContextBtn'),
            saveContextBtn: document.getElementById('saveContextBtn'),

            // Notification
            notification: document.getElementById('notification'),

            // Clear Modal
            clearModal: document.getElementById('clearModal'),

            // Delete Recording Modal
            deleteRecordingModal: document.getElementById('deleteRecordingModal'),
        };

        this.timerInterval = null;
        this.startTime = null;
        this.currentSource = 'microphone';
        this.fontSizeLevel = 0; // 0=normal, -1=sm, 1=lg, 2=xl

        // Callbacks
        this.onContextSave = null; // Callback to Main.js

        this.setupEventHandlers();
    }

    setupEventHandlers() {
        // Dropdown toggle
        if (this.elements.audioSourceBtn) {
            this.elements.audioSourceBtn.onclick = (e) => {
                e.stopPropagation();
                this.elements.audioDropdown.classList.toggle('active');
            };
        }

        // Close dropdown when clicking outside
        document.addEventListener('click', () => {
            if (this.elements.audioDropdown) {
                this.elements.audioDropdown.classList.remove('active');
            }
        });

        // Dropdown items
        if (this.elements.audioDropdown) {
            this.elements.audioDropdown.querySelectorAll('.dropdown-item').forEach(item => {
                item.onclick = () => {
                    const source = item.dataset.source;
                    this.setAudioSource(source);
                    this.elements.audioDropdown.classList.remove('active');
                };
            });
        }

        // Font size controls
        if (this.elements.fontDecrease) this.elements.fontDecrease.onclick = () => this.changeFontSize(-1);
        if (this.elements.fontIncrease) this.elements.fontIncrease.onclick = () => this.changeFontSize(1);

        // Clear Transcript
        if (this.elements.clearBtn) {
            this.elements.clearBtn.onclick = () => {
                // Show confirmation modal
                if (this.elements.clearModal) this.elements.clearModal.classList.add('active');
            };
        }

        // Clear Modal Buttons - Bind programmatically instead of inline onclick
        const clearModalClose = this.elements.clearModal?.querySelector('.modal-close');
        const clearModalCancel = this.elements.clearModal?.querySelector('.btn-secondary');
        const clearModalConfirm = this.elements.clearModal?.querySelector('.btn-primary');

        if (clearModalClose) clearModalClose.onclick = () => this.closeClearModal();
        if (clearModalCancel) clearModalCancel.onclick = () => this.closeClearModal();
        if (clearModalConfirm) clearModalConfirm.onclick = () => this.confirmClear();

        // Copy Button
        if (this.elements.copyBtn) this.elements.copyBtn.onclick = () => this.copyTranscripts();

        // Context modal
        if (this.elements.contextBtn) this.elements.contextBtn.onclick = () => this.showContextModal();
        if (this.elements.closeModal) this.elements.closeModal.onclick = () => this.hideContextModal();
        if (this.elements.clearContextBtn) this.elements.clearContextBtn.onclick = () => this.clearContext();
        if (this.elements.saveContextBtn) this.elements.saveContextBtn.onclick = () => this.saveContext();

        // Close modal on overlay click
        [this.elements.contextModal, this.elements.clearModal].forEach(modal => {
            if (modal) {
                modal.onclick = (e) => {
                    if (e.target === modal) modal.classList.remove('active');
                };
            }
        });

        // Sidebar toggle
        if (this.elements.sidebarToggle) {
            this.elements.sidebarToggle.onclick = () => this.toggleSidebar();
        }

        // Load sidebar recordings on init
        this.loadRecordingsSidebar();
    }

    setAudioSource(source) {
        this.currentSource = source;
        this.elements.audioSourceText.textContent =
            source === 'computer' ? 'Computer Audio' : 'Microphone';
    }

    getAudioSource() {
        return this.currentSource;
    }

    updateAudioMeter(rms) {
        const normalizedLevel = Math.min(10, Math.floor(rms * 50)); // Scale RMS to 0-10
        const bars = this.elements.audioMeter.querySelectorAll('.meter-bar-h');

        bars.forEach((bar, index) => {
            bar.classList.toggle('active', index < normalizedLevel);
        });
    }

    resetAudioMeter() {
        const bars = this.elements.audioMeter.querySelectorAll('.meter-bar-h');
        bars.forEach(bar => bar.classList.remove('active'));
    }

    startTimer() {
        this.startTime = Date.now();
        this.timerInterval = setInterval(() => this.updateTimer(), 1000);
        this.updateTimer();
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    updateTimer() {
        const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        this.elements.timer.textContent =
            `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    resetTimer() {
        this.elements.timer.textContent = '00:00';
    }

    updateRecordButton(isRecording) {
        if (isRecording) {
            this.elements.recordBtn.classList.add('recording');
            // Stop Icon
            this.elements.recordBtn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="6" width="12" height="12" rx="2"/>
                </svg>
            `;
        } else {
            this.elements.recordBtn.classList.remove('recording');
            // Mic Icon
            this.elements.recordBtn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                    <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                </svg>
            `;
        }
    }

    changeFontSize(delta) {
        this.fontSizeLevel = Math.max(-1, Math.min(2, this.fontSizeLevel + delta));
        this.elements.panelContent.classList.remove('font-sm', 'font-lg', 'font-xl');

        if (this.fontSizeLevel === -1) this.elements.panelContent.classList.add('font-sm');
        else if (this.fontSizeLevel === 1) this.elements.panelContent.classList.add('font-lg');
        else if (this.fontSizeLevel === 2) this.elements.panelContent.classList.add('font-xl');
    }

    // 👇 NÂNG CẤP: Hỗ trợ hiển thị song ngữ VI - EN và trạng thái Final
    addTranscriptSegment(data) {
        const { segment_id, source, target, timestamp, is_final } = data;
        const segmentIdStr = `segment-${segment_id}`;
        let segment = document.getElementById(segmentIdStr);

        // 1. Nếu chưa có segment này thì tạo mới
        if (!segment) {
            // Remove empty state if exists
            const emptyState = this.elements.panelContent.querySelector('.empty-state');
            if (emptyState) emptyState.remove();

            segment = document.createElement('div');
            segment.id = segmentIdStr;
            segment.className = 'transcript-segment partial'; // Mặc định là partial

            // Structure: Meta + VI + EN
            segment.innerHTML = `
                <div class="segment-meta">
                    <span class="timestamp">${timestamp}</span>
                    <span class="segment-id">#${segment_id}</span>
                </div>
                <div class="vi-text"></div>
                <div class="en-text"></div>
            `;
            this.elements.panelContent.appendChild(segment);

            // Auto scroll
            this.elements.panelContent.scrollTop = this.elements.panelContent.scrollHeight;
        }

        // 2. Cập nhật nội dung - TransyncAI Style
        const viEl = segment.querySelector('.vi-text');
        const enEl = segment.querySelector('.en-text');

        // Partial: thêm "..." ở cuối để signal đang xử lý
        if (viEl) {
            viEl.textContent = is_final ? (source || "") : (source ? source + "..." : "");
        }
        // English: hiện target hoặc "..." nếu đang dịch
        if (enEl) {
            enEl.textContent = target || (source ? "..." : "");
        }

        // 3. Cập nhật trạng thái (Final / Partial)
        if (is_final) {
            segment.classList.remove('partial');
        } else {
            segment.classList.add('partial');
        }
    }

    clearTranscripts() {
        this.elements.panelContent.innerHTML = `
            <div class="empty-state">
                <p>Click the record button to start</p>
                <p>Transcription will appear here</p>
            </div>
        `;
        if (this.elements.clearModal) this.elements.clearModal.classList.remove('active');
    }

    async copyTranscripts() {
        // Copy format: [Time] VI \n EN
        const segments = Array.from(this.elements.panelContent.querySelectorAll('.transcript-segment'));

        const text = segments.map(seg => {
            const time = seg.querySelector('.timestamp')?.textContent || "";
            const vi = seg.querySelector('.vi-text')?.textContent || "";
            const en = seg.querySelector('.en-text')?.textContent || "";
            return `[${time}] ${vi}\n> ${en}\n`;
        }).join('\n');

        if (!text.trim()) {
            this.showNotification('Nothing to copy!');
            return;
        }

        try {
            await navigator.clipboard.writeText(text);
            this.showNotification('Copied transcript to clipboard!');
        } catch (err) {
            console.error('Copy failed:', err);
        }
    }

    showNotification(message) {
        this.elements.notification.textContent = message;
        this.elements.notification.classList.add('active');
        setTimeout(() => {
            this.elements.notification.classList.remove('active');
        }, 3000);
    }

    // Clear Modal Methods
    closeClearModal() {
        if (this.elements.clearModal) {
            this.elements.clearModal.classList.remove('active');
        }
    }

    confirmClear() {
        this.clearTranscripts();
        this.closeClearModal();
        this.showNotification('Transcript cleared');
    }

    // Sidebar Recordings
    loadRecordingsSidebar() {
        if (!this.elements.recordingsList) return;

        try {
            const recordings = JSON.parse(localStorage.getItem('recordings') || '[]');

            if (recordings.length === 0) {
                this.elements.recordingsList.innerHTML = `
                    <div class="empty-state" style="padding: 2rem 1rem;">
                        <p>No recordings yet</p>
                    </div>
                `;
                return;
            }

            // Show only last 5 recordings in sidebar
            const recentRecordings = recordings.slice(0, 5);

            this.elements.recordingsList.innerHTML = recentRecordings.map(rec => {
                const displayName = rec.customName || `${rec.date} ${rec.time}`;
                const segmentCount = rec.transcript?.length || 0;
                return `
                    <div class="recording-card" data-id="${rec.id}">
                        <div class="recording-info">
                            <div class="recording-date">${displayName}</div>
                            <div class="recording-duration">${rec.duration} • ${segmentCount} segments</div>
                        </div>
                        <div class="recording-card-actions">
                            <button class="btn-view-small" data-action="view" data-id="${rec.id}" title="View">
                                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                                    <circle cx="12" cy="12" r="3"/>
                                </svg>
                            </button>
                            <button class="btn-delete-small" data-action="delete" data-id="${rec.id}" title="Delete">
                                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                `;
            }).join('');

            // Attach event listeners to buttons
            this.elements.recordingsList.querySelectorAll('[data-action]').forEach(btn => {
                btn.onclick = (e) => {
                    e.stopPropagation();
                    const action = btn.dataset.action;
                    const id = btn.dataset.id;

                    if (action === 'view') {
                        this.viewRecording(id);
                    } else if (action === 'delete') {
                        this.deleteRecording(id);
                    }
                };
            });

        } catch (e) {
            console.error('Failed to load recordings sidebar:', e);
        }
    }

    viewRecording(id) {
        // Save selected ID and navigate to recordings page
        localStorage.setItem('selectedRecordingId', id);
        window.location.href = 'recordings.html';
    }

    deleteRecording(id) {
        // Store ID and show modal
        this.pendingDeleteId = id;
        this.showDeleteModal();
    }

    showDeleteModal() {
        if (this.elements.deleteRecordingModal) {
            this.elements.deleteRecordingModal.classList.add('active');

            // Setup modal buttons
            const closeBtn = document.getElementById('deleteModalClose');
            const cancelBtn = document.getElementById('deleteModalCancel');
            const confirmBtn = document.getElementById('deleteModalConfirm');

            if (closeBtn) closeBtn.onclick = () => this.hideDeleteModal();
            if (cancelBtn) cancelBtn.onclick = () => this.hideDeleteModal();
            if (confirmBtn) confirmBtn.onclick = () => this.confirmDeleteRecording();

            // Close on overlay click
            this.elements.deleteRecordingModal.onclick = (e) => {
                if (e.target === this.elements.deleteRecordingModal) {
                    this.hideDeleteModal();
                }
            };
        }
    }

    hideDeleteModal() {
        if (this.elements.deleteRecordingModal) {
            this.elements.deleteRecordingModal.classList.remove('active');
        }
        this.pendingDeleteId = null;
    }

    confirmDeleteRecording() {
        if (!this.pendingDeleteId) return;

        try {
            let recordings = JSON.parse(localStorage.getItem('recordings') || '[]');
            recordings = recordings.filter(r => r.id != this.pendingDeleteId);
            localStorage.setItem('recordings', JSON.stringify(recordings));

            this.loadRecordingsSidebar();
            this.showNotification('Recording deleted');
        } catch (e) {
            console.error('Failed to delete recording:', e);
        }

        this.hideDeleteModal();
    }

    toggleSidebar() {
        this.elements.sidebar.classList.toggle('active');
        // Reload recordings when opening sidebar
        this.loadRecordingsSidebar();
    }

    showContextModal() {
        this.elements.contextModal.classList.add('active');
    }

    hideContextModal() {
        this.elements.contextModal.classList.remove('active');
    }

    clearContext() {
        this.elements.keywords.value = '';
        this.elements.context.value = '';
    }

    saveContext() {
        const keywords = this.elements.keywords.value.split(',').map(k => k.trim()).filter(k => k);
        const context = this.elements.context.value.trim();

        if (this.onContextSave) {
            this.onContextSave({ keywords, context });
        }

        this.hideContextModal();
        this.showNotification('Context saved! AI will adapt.');
    }
}
