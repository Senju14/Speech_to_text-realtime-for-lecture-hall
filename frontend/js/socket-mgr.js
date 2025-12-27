/**
 * WebSocket Manager
 * Handles connection, reconnection, messaging, and Heartbeat (Keep-Alive)
 */

import { WS_URL, WS_RECONNECT_DELAY_MS, WS_MAX_RETRIES, WS_TIMEOUT_MS } from './config.js';

export class SocketManager {
    constructor() {
        this.ws = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;

        // Heartbeat timer
        this.pingInterval = null;

        // Callbacks
        this.onMessage = null;
        this.onConnected = null;
        this.onDisconnected = null;
        this.onError = null;
    }

    /**
     * Connect to WebSocket server
     * @returns {Promise<void>}
     */
    async connect() {
        return new Promise((resolve, reject) => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                resolve();
                return;
            }

            console.log(`[WS] Connecting to ${WS_URL}...`);
            this.ws = new WebSocket(WS_URL);

            // Connection timeout safety
            const timeout = setTimeout(() => {
                if (this.ws && this.ws.readyState !== WebSocket.OPEN) {
                    this.ws.close();
                    reject(new Error('Connection timeout'));
                }
            }, WS_TIMEOUT_MS);

            this.ws.onopen = () => {
                clearTimeout(timeout);
                this.connected = true;
                this.reconnectAttempts = 0;
                console.log('[WS] Connected!');

                // 👇 BẮT ĐẦU GỬI PING ĐỂ GIỮ KẾT NỐI
                this.startHeartbeat();

                if (this.onConnected) {
                    this.onConnected();
                }

                resolve();
            };

            this.ws.onclose = (event) => {
                this.connected = false;
                this.stopHeartbeat(); // 👇 DỪNG PING
                console.log(`[WS] Disconnected (Code: ${event.code})`);

                if (this.onDisconnected) {
                    this.onDisconnected();
                }

                // Auto-reconnect if not closed cleanly
                if (event.code !== 1000) {
                    this.scheduleReconnect();
                }
            };

            this.ws.onerror = (err) => {
                console.error('[WS] Error:', err);
                if (this.onError) {
                    this.onError(err);
                }
                // Don't reject here immediately as onclose usually follows
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    // Bỏ qua tin nhắn pong (chỉ để giữ kết nối)
                    if (data.type === 'pong') return;

                    if (this.onMessage) {
                        this.onMessage(data);
                    }
                } catch (err) {
                    console.error('[WS] Message parse error:', err);
                }
            };
        });
    }

    /**
     * Schedule automatic reconnection
     */
    scheduleReconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }

        if (this.reconnectAttempts >= WS_MAX_RETRIES) {
            console.error(`[WS] Max reconnection attempts (${WS_MAX_RETRIES}) reached`);
            if (this.onError) this.onError(new Error("Connection failed after max retries"));
            return;
        }

        this.reconnectAttempts++;
        console.log(`[WS] Reconnecting in ${WS_RECONNECT_DELAY_MS}ms (attempt ${this.reconnectAttempts}/${WS_MAX_RETRIES})`);

        this.reconnectTimer = setTimeout(() => {
            this.connect().catch(err => {
                console.error('[WS] Reconnection failed:', err);
            });
        }, WS_RECONNECT_DELAY_MS);
    }

    /**
     * Start sending Ping every 30s to keep connection alive on Cloud
     */
    startHeartbeat() {
        this.stopHeartbeat();
        this.pingInterval = setInterval(() => {
            if (this.isConnected()) {
                this.send('ping'); // Backend sẽ trả về 'pong'
            }
        }, 30000); // 30 giây
    }

    stopHeartbeat() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    /**
     * Send message to server
     * @param {string} type - Message type
     * @param {Object} data - Message data
     * @returns {boolean} Success
     */
    send(type, data = {}) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            // console.warn('[WS] Cannot send - not connected');
            return false;
        }

        try {
            this.ws.send(JSON.stringify({ type, ...data }));
            return true;
        } catch (e) {
            console.error("[WS] Send error:", e);
            return false;
        }
    }

    /**
     * Close connection
     */
    close() {
        this.stopHeartbeat();

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.ws) {
            // 1000 = Normal Closure
            this.ws.close(1000, "User initiated close");
            this.ws = null;
        }

        this.connected = false;
        this.reconnectAttempts = 0;
    }

    /**
     * Check if connected
     * @returns {boolean}
     */
    isConnected() {
        return this.connected && this.ws && this.ws.readyState === WebSocket.OPEN;
    }
}
