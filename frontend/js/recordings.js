import { exportSRT, exportVTT, exportTXT, exportJSON } from './export.js';

class RecordingsManager {
    constructor() {
        this.listEl = document.getElementById('recordingsList');
        this.contentEl = document.getElementById('transcriptContent');
        this.titleEl = document.getElementById('recordingTitle');
        this.subtitleEl = document.getElementById('recordingSubtitle');
        this.summarizeBtn = document.getElementById('summarizeBtn');
        this.selectedId = null;
        this.recordings = [];
        this.menuOpenId = null;
    }

    init() {
        this.loadRecordings();
        this.setupListeners();
        this.setupModals();
        this.setupSummary();

        const preSelected = localStorage.getItem('selectedRecordingId');
        if (preSelected) {
            localStorage.removeItem('selectedRecordingId');
            setTimeout(() => this.selectRecording(preSelected), 200);
        } else if (this.recordings.length > 0) {
            this.selectRecording(this.recordings[0].id);
        }
    }

    loadRecordings() {
        try {
            this.recordings = JSON.parse(localStorage.getItem('recordings') || '[]');
            // Sort: starred first, then by date
            this.recordings.sort((a, b) => {
                if (a.starred && !b.starred) return -1;
                if (!a.starred && b.starred) return 1;
                return (b.id || 0) - (a.id || 0);
            });
        } catch (e) {
            this.recordings = [];
        }
        this.renderList();
    }

    renderList() {
        if (!this.listEl) return;

        if (this.recordings.length === 0) {
            this.listEl.innerHTML = '<div class="empty-state"><p>No recordings</p></div>';
            return;
        }

        this.listEl.innerHTML = this.recordings.map(rec => {
            const name = rec.customName || `${rec.date} ${rec.time}`;
            const count = rec.transcript?.length || 0;
            const active = String(rec.id) === String(this.selectedId) ? 'active' : '';
            const starred = rec.starred ? '⭐ ' : '';

            return `
                <div class="recording-item ${active}" data-id="${rec.id}">
                    <div class="recording-info">
                        <div class="recording-date">${starred}${name}</div>
                        <div class="recording-time">${rec.duration || '00:00'} • ${count} segments</div>
                    </div>
                    <div class="recording-actions">
                        <button class="menu-btn" data-menu="${rec.id}" title="Options">⋮</button>
                        <div class="context-menu" id="menu-${rec.id}">
                            <div class="context-menu-item" data-action="rename" data-id="${rec.id}">Rename</div>
                            <div class="context-menu-item" data-action="star" data-id="${rec.id}">${rec.starred ? 'Unstar' : 'Star'}</div>
                            <div class="context-menu-item" data-action="summarize" data-id="${rec.id}">📝 Summarize</div>
                            <div class="context-menu-item has-submenu" data-id="${rec.id}">
                                Export ▸
                                <div class="context-submenu">
                                    <div class="context-menu-item" data-action="export-srt" data-id="${rec.id}">SRT (Subtitles)</div>
                                    <div class="context-menu-item" data-action="export-vtt" data-id="${rec.id}">VTT (WebVTT)</div>
                                    <div class="context-menu-item" data-action="export-txt" data-id="${rec.id}">TXT (Plain text)</div>
                                    <div class="context-menu-item" data-action="export-json" data-id="${rec.id}">JSON (Data)</div>
                                </div>
                            </div>
                            <div class="context-menu-item delete" data-action="delete" data-id="${rec.id}">Delete</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    setupListeners() {
        if (!this.listEl) return;

        document.addEventListener('click', () => {
            this.closeAllMenus();
        });

        this.listEl.addEventListener('click', (e) => {
            const menuBtn = e.target.closest('[data-menu]');
            if (menuBtn) {
                e.stopPropagation();
                const id = menuBtn.dataset.menu;
                this.toggleMenu(id);
                return;
            }

            const actionEl = e.target.closest('[data-action]');
            if (actionEl) {
                e.stopPropagation();
                const { action, id } = actionEl.dataset;
                this.handleAction(action, id);
                this.closeAllMenus();
                return;
            }

            const item = e.target.closest('.recording-item');
            if (item) {
                this.selectRecording(item.dataset.id);
            }
        });
    }

    toggleMenu(id) {
        const menu = document.getElementById(`menu-${id}`);
        if (!menu) return;

        const isOpen = menu.classList.contains('active');
        this.closeAllMenus();

        if (!isOpen) {
            menu.classList.add('active');
            this.menuOpenId = id;
        }
    }

    closeAllMenus() {
        document.querySelectorAll('.context-menu.active').forEach(m => m.classList.remove('active'));
        this.menuOpenId = null;
    }

    handleAction(action, id) {
        switch (action) {
            case 'rename': this.showRenameModal(id); break;
            case 'star': this.toggleStar(id); break;
            case 'summarize': this.summarizeRecording(id); break;
            case 'export': this.exportRecording(id, 'srt'); break;
            case 'export-srt': this.exportRecording(id, 'srt'); break;
            case 'export-vtt': this.exportRecording(id, 'vtt'); break;
            case 'export-txt': this.exportRecording(id, 'txt'); break;
            case 'export-json': this.exportRecording(id, 'json'); break;
            case 'delete': this.showDeleteModal(id); break;
        }
    }

    toggleStar(id) {
        const rec = this.recordings.find(r => String(r.id) === String(id));
        if (rec) {
            rec.starred = !rec.starred;
            this.saveRecordings();
            this.loadRecordings();
        }
    }

    exportRecording(id, format = 'srt') {
        const rec = this.recordings.find(r => String(r.id) === String(id));
        if (!rec || !rec.transcript) return;

        switch (format) {
            case 'srt': exportSRT(rec.transcript); break;
            case 'vtt': exportVTT(rec.transcript); break;
            case 'txt': exportTXT(rec.transcript); break;
            case 'json': exportJSON(rec.transcript); break;
            default: exportSRT(rec.transcript);
        }
    }

    setupModals() {
        const renameModal = document.getElementById('renameModal');
        const renameClose = document.getElementById('renameModalClose');
        const renameCancel = document.getElementById('renameModalCancel');
        const renameSave = document.getElementById('renameModalSave');

        if (renameClose) renameClose.onclick = () => this.hideRenameModal();
        if (renameCancel) renameCancel.onclick = () => this.hideRenameModal();
        if (renameSave) renameSave.onclick = () => this.saveRename();
        if (renameModal) renameModal.onclick = (e) => {
            if (e.target === renameModal) this.hideRenameModal();
        };

        const deleteModal = document.getElementById('deleteModal');
        const deleteClose = document.getElementById('deleteModalClose');
        const deleteCancel = document.getElementById('deleteModalCancel');
        const deleteConfirm = document.getElementById('deleteModalConfirm');

        if (deleteClose) deleteClose.onclick = () => this.hideDeleteModal();
        if (deleteCancel) deleteCancel.onclick = () => this.hideDeleteModal();
        if (deleteConfirm) deleteConfirm.onclick = () => this.confirmDelete();
        if (deleteModal) deleteModal.onclick = (e) => {
            if (e.target === deleteModal) this.hideDeleteModal();
        };
    }

    showRenameModal(id) {
        this.pendingRenameId = id;
        const rec = this.recordings.find(r => String(r.id) === String(id));
        const input = document.getElementById('renameInput');
        if (input && rec) {
            input.value = rec.customName || `${rec.date} ${rec.time}`;
        }
        document.getElementById('renameModal')?.classList.add('active');
    }

    hideRenameModal() {
        document.getElementById('renameModal')?.classList.remove('active');
        this.pendingRenameId = null;
    }

    saveRename() {
        const input = document.getElementById('renameInput');
        const newName = input?.value?.trim();
        if (!newName || !this.pendingRenameId) {
            this.hideRenameModal();
            return;
        }

        const rec = this.recordings.find(r => String(r.id) === String(this.pendingRenameId));
        if (rec) {
            rec.customName = newName;
            this.saveRecordings();
            this.renderList();
            if (String(this.selectedId) === String(this.pendingRenameId)) {
                this.selectRecording(this.pendingRenameId);
            }
        }
        this.hideRenameModal();
    }

    showDeleteModal(id) {
        this.pendingDeleteId = id;
        document.getElementById('deleteModal')?.classList.add('active');
    }

    hideDeleteModal() {
        document.getElementById('deleteModal')?.classList.remove('active');
        this.pendingDeleteId = null;
    }

    confirmDelete() {
        if (!this.pendingDeleteId) {
            this.hideDeleteModal();
            return;
        }

        this.recordings = this.recordings.filter(r => String(r.id) !== String(this.pendingDeleteId));
        this.saveRecordings();

        if (String(this.selectedId) === String(this.pendingDeleteId)) {
            this.selectedId = null;
            this.showEmpty();
        }

        this.renderList();
        this.hideDeleteModal();
    }

    saveRecordings() {
        localStorage.setItem('recordings', JSON.stringify(this.recordings));
    }

    selectRecording(id) {
        this.selectedId = id;

        this.listEl?.querySelectorAll('.recording-item').forEach(el => {
            el.classList.toggle('active', String(el.dataset.id) === String(id));
        });

        const rec = this.recordings.find(r => String(r.id) === String(id));
        if (!rec) {
            this.showEmpty();
            return;
        }

        const name = rec.customName || `${rec.date} ${rec.time}`;
        const starred = rec.starred ? '⭐ ' : '';
        if (this.titleEl) this.titleEl.textContent = starred + name;
        if (this.subtitleEl) this.subtitleEl.textContent = `${rec.duration || '00:00'} • ${rec.transcript?.length || 0} segments`;
        if (this.summarizeBtn) this.summarizeBtn.style.display = '';

        this.renderTranscript(rec.transcript || []);
    }

    renderTranscript(transcript) {
        const container = this.contentEl?.querySelector('.transcript-container-full') || this.contentEl;
        if (!container) return;

        if (!transcript || transcript.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No transcript</p></div>';
            return;
        }

        container.innerHTML = transcript.map((item, i) => {
            const vi = item.vi || item.source || '';
            const en = item.en || item.target || '';

            return `
                <div class="transcript-block">
                    <div class="vi-text">${vi}</div>
                    ${en ? `<div class="en-text">${en}</div>` : ''}
                </div>
            `;
        }).join('');
    }

    showEmpty() {
        if (this.titleEl) this.titleEl.textContent = 'Select a recording';
        if (this.subtitleEl) this.subtitleEl.textContent = '';
        if (this.summarizeBtn) this.summarizeBtn.style.display = 'none';

        const container = this.contentEl?.querySelector('.transcript-container-full') || this.contentEl;
        if (container) {
            container.innerHTML = '<div class="empty-state"><p>Select a recording</p></div>';
        }
    }

    setupSummary() {
        // Header summarize button
        if (this.summarizeBtn) {
            this.summarizeBtn.onclick = () => {
                if (this.selectedId) this.summarizeRecording(this.selectedId);
            };
        }

        // Summary modal controls
        const closeModal = () => {
            document.getElementById('summaryModal')?.classList.remove('active');
        };
        document.getElementById('closeSummaryModal')?.addEventListener('click', closeModal);
        document.getElementById('closeSummaryBtn')?.addEventListener('click', closeModal);
        document.getElementById('summaryModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'summaryModal') closeModal();
        });
        document.getElementById('copySummaryBtn')?.addEventListener('click', async () => {
            const content = document.getElementById('summaryContent')?.innerText;
            if (content) {
                await navigator.clipboard.writeText(content);
                this.showToast('Summary copied!', 'success');
            }
        });
    }

    async summarizeRecording(id) {
        const rec = this.recordings.find(r => String(r.id) === String(id));
        if (!rec || !rec.transcript || rec.transcript.length === 0) {
            this.showToast('No transcript to summarize', 'error');
            return;
        }

        // Check if we already have a cached summary
        if (rec.summary) {
            this.showSummaryModal(rec.summary);
            return;
        }

        const btn = this.summarizeBtn;
        if (btn) {
            btn.disabled = true;
            btn.textContent = '⏳ Summarizing...';
        }

        try {
            // Build transcript text for the API
            const transcriptText = rec.transcript
                .map(item => item.vi || item.source || '')
                .filter(t => t.trim())
                .join('\n');

            const response = await fetch('/api/summarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    transcript: transcriptText,
                    topic: rec.customName || '',
                }),
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            const summary = data.summary || 'No summary generated.';

            // Cache the summary
            rec.summary = summary;
            this.saveRecordings();

            this.showSummaryModal(summary);
            this.showToast('Summary generated!', 'success');
        } catch (err) {
            console.error('Summarize error:', err);
            this.showToast('Failed to generate summary. Is the server running?', 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = '📝 Summarize';
            }
        }
    }

    showSummaryModal(summary) {
        const modal = document.getElementById('summaryModal');
        const content = document.getElementById('summaryContent');
        if (!modal || !content) return;

        // Convert markdown-like formatting to HTML
        let html = summary
            .replace(/^### (.+)$/gm, '<h3>$1</h3>')
            .replace(/^## (.+)$/gm, '<h2>$1</h2>')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/^- (.+)$/gm, '<li>$1</li>')
            .replace(/\n/g, '<br>');

        content.innerHTML = html;
        modal.classList.add('active');
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        container.appendChild(toast);

        requestAnimationFrame(() => toast.classList.add('show'));
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new RecordingsManager().init();
});
