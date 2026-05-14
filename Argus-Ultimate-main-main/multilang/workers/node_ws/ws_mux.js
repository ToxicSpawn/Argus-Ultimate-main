/**
 * ARGUS WebSocket Multiplexer
 *
 * Connects to multiple exchange WebSocket feeds simultaneously and
 * re-broadcasts all messages through a single local WebSocket server
 * on localhost:9999.
 *
 * Protocol (outbound to Python):
 *   { "exchange": "kraken", "channel": "ticker", "data": {...}, "timestamp": 1234567890123 }
 *
 * Config is read from stdin as a JSON object on startup:
 *   { "feeds": [
 *       { "exchange": "kraken", "url": "wss://ws.kraken.com", "subscribe": {...} },
 *       { "exchange": "coinbase", "url": "wss://ws-feed.exchange.coinbase.com", "subscribe": {...} }
 *   ], "port": 9999 }
 *
 * If no config is provided on stdin, uses default port 9999 and waits for
 * feeds to be added via control messages.
 *
 * Install: npm install
 * Run:     node ws_mux.js
 */

'use strict';

const WebSocket = require('ws');

const DEFAULT_PORT = 9999;
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_DELAY_MS = 30000;

class WSMultiplexer {
    constructor(port) {
        this.port = port || DEFAULT_PORT;
        this.server = null;
        this.clients = new Set();
        this.feeds = new Map();        // exchange -> { ws, url, subscribe, reconnectDelay }
        this.stats = new Map();        // exchange -> { messages: 0, lastMessage: null, connected: false }
    }

    start() {
        this.server = new WebSocket.Server({ port: this.port, host: '127.0.0.1' });

        this.server.on('connection', (ws) => {
            this.clients.add(ws);
            console.log(`[MUX] Client connected (total: ${this.clients.size})`);

            ws.on('close', () => {
                this.clients.delete(ws);
                console.log(`[MUX] Client disconnected (total: ${this.clients.size})`);
            });

            ws.on('message', (data) => {
                try {
                    const msg = JSON.parse(data);
                    if (msg.command === 'add_feed') {
                        this.addFeed(msg.exchange, msg.url, msg.subscribe);
                    } else if (msg.command === 'remove_feed') {
                        this.removeFeed(msg.exchange);
                    } else if (msg.command === 'status') {
                        ws.send(JSON.stringify({ type: 'status', feeds: this.getStatus() }));
                    }
                } catch (e) {
                    // Ignore malformed control messages
                }
            });

            // Send current status to new client
            ws.send(JSON.stringify({ type: 'status', feeds: this.getStatus() }));
        });

        this.server.on('listening', () => {
            console.log(`[MUX] WebSocket server listening on ws://127.0.0.1:${this.port}`);
        });

        this.server.on('error', (err) => {
            console.error(`[MUX] Server error: ${err.message}`);
        });
    }

    addFeed(exchange, url, subscribeMsg) {
        if (this.feeds.has(exchange)) {
            console.log(`[MUX] Feed ${exchange} already exists, reconnecting...`);
            this.removeFeed(exchange);
        }

        const feedState = {
            url,
            subscribe: subscribeMsg,
            reconnectDelay: RECONNECT_DELAY_MS,
            ws: null,
        };
        this.feeds.set(exchange, feedState);
        this.stats.set(exchange, { messages: 0, lastMessage: null, connected: false });

        this._connectFeed(exchange);
    }

    removeFeed(exchange) {
        const feed = this.feeds.get(exchange);
        if (feed && feed.ws) {
            feed.ws.removeAllListeners();
            feed.ws.close();
        }
        this.feeds.delete(exchange);
        this.stats.delete(exchange);
        console.log(`[MUX] Removed feed: ${exchange}`);
    }

    _connectFeed(exchange) {
        const feed = this.feeds.get(exchange);
        if (!feed) return;

        console.log(`[MUX] Connecting to ${exchange}: ${feed.url}`);

        try {
            const ws = new WebSocket(feed.url);
            feed.ws = ws;

            ws.on('open', () => {
                console.log(`[MUX] Connected to ${exchange}`);
                feed.reconnectDelay = RECONNECT_DELAY_MS;
                const stat = this.stats.get(exchange);
                if (stat) stat.connected = true;

                // Send subscription message if provided
                if (feed.subscribe) {
                    ws.send(JSON.stringify(feed.subscribe));
                }
            });

            ws.on('message', (data) => {
                const stat = this.stats.get(exchange);
                if (stat) {
                    stat.messages++;
                    stat.lastMessage = Date.now();
                }

                let parsed;
                try {
                    parsed = JSON.parse(data);
                } catch (e) {
                    parsed = { raw: data.toString() };
                }

                const envelope = {
                    exchange,
                    channel: parsed.channel || parsed.type || 'unknown',
                    data: parsed,
                    timestamp: Date.now(),
                };

                const msg = JSON.stringify(envelope);
                for (const client of this.clients) {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(msg);
                    }
                }
            });

            ws.on('close', (code, reason) => {
                console.log(`[MUX] ${exchange} disconnected (${code}): ${reason}`);
                const stat = this.stats.get(exchange);
                if (stat) stat.connected = false;
                this._scheduleReconnect(exchange);
            });

            ws.on('error', (err) => {
                console.error(`[MUX] ${exchange} error: ${err.message}`);
                const stat = this.stats.get(exchange);
                if (stat) stat.connected = false;
            });
        } catch (err) {
            console.error(`[MUX] Failed to create WebSocket for ${exchange}: ${err.message}`);
            this._scheduleReconnect(exchange);
        }
    }

    _scheduleReconnect(exchange) {
        const feed = this.feeds.get(exchange);
        if (!feed) return;

        console.log(`[MUX] Reconnecting ${exchange} in ${feed.reconnectDelay}ms...`);
        setTimeout(() => {
            if (this.feeds.has(exchange)) {
                this._connectFeed(exchange);
            }
        }, feed.reconnectDelay);

        // Exponential backoff
        feed.reconnectDelay = Math.min(feed.reconnectDelay * 1.5, MAX_RECONNECT_DELAY_MS);
    }

    getStatus() {
        const status = {};
        for (const [exchange, stat] of this.stats.entries()) {
            status[exchange] = { ...stat };
        }
        return status;
    }

    shutdown() {
        for (const [exchange] of this.feeds.entries()) {
            this.removeFeed(exchange);
        }
        if (this.server) {
            this.server.close();
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function readStdin() {
    return new Promise((resolve) => {
        let data = '';
        const timeout = setTimeout(() => resolve(null), 500);
        process.stdin.setEncoding('utf8');
        process.stdin.on('data', (chunk) => {
            clearTimeout(timeout);
            data += chunk;
        });
        process.stdin.on('end', () => {
            clearTimeout(timeout);
            try {
                resolve(JSON.parse(data));
            } catch (e) {
                resolve(null);
            }
        });
        // If stdin is a TTY, don't wait
        if (process.stdin.isTTY) {
            clearTimeout(timeout);
            resolve(null);
        }
    });
}

async function main() {
    const config = await readStdin();
    const port = (config && config.port) || DEFAULT_PORT;

    const mux = new WSMultiplexer(port);
    mux.start();

    // Add configured feeds
    if (config && config.feeds) {
        for (const feed of config.feeds) {
            mux.addFeed(feed.exchange, feed.url, feed.subscribe || null);
        }
    }

    // Graceful shutdown
    process.on('SIGINT', () => {
        console.log('[MUX] Shutting down...');
        mux.shutdown();
        process.exit(0);
    });
    process.on('SIGTERM', () => {
        mux.shutdown();
        process.exit(0);
    });
}

main().catch((err) => {
    console.error(`[MUX] Fatal: ${err.message}`);
    process.exit(1);
});
