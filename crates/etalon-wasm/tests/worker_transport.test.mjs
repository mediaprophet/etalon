import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { MessageChannel } from 'node:worker_threads';

const source = await readFile(new URL('../js/worker_transport.js', import.meta.url), 'utf8');
const { EtalonWorkerTransport } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
async function pair(isolated, budget = 1024) {
    globalThis.crossOriginIsolated = isolated;
    const channel = new MessageChannel();
    const left = [], right = [], received = [[], []];
    let leftBarrier, rightBarrier;
    channel.port1.on('message', value => value.testBarrier ? leftBarrier() : left.push(value));
    channel.port2.on('message', value => value.testBarrier ? rightBarrier() : right.push(value));
    const a = new EtalonWorkerTransport(channel.port1, value => received[0].push(value), true, budget);
    const b = new EtalonWorkerTransport(channel.port2, value => received[1].push(value));
    async function settle() {
        let timer;
        const delivered = Promise.all([
            new Promise(resolve => { leftBarrier = resolve; }),
            new Promise(resolve => { rightBarrier = resolve; }),
        ]);
        channel.port1.postMessage({ testBarrier: true });
        channel.port2.postMessage({ testBarrier: true });
        try {
            await Promise.race([
                delivered,
                new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('MessageChannel deadline')), 2000); }),
            ]);
        } finally { clearTimeout(timer); }
        while (right.length) b.handle(right.shift());
        while (left.length) a.handle(left.shift());
    }
    await settle();
    return { a, b, received, left, right, settle, close() { channel.port1.close(); channel.port2.close(); } };
}

for (const isolated of [false, true]) {
    test(`actual MessageChannel bidirectional delivery and ownership (isolated=${isolated})`, async () => {
        const p = await pair(isolated);
        try {
            assert.equal(p.a.shared, isolated);
            assert.equal(p.b.shared, isolated);
            p.a.send({ id: 1, text: 'héllo' });
            p.b.send({ id: 2, reply: true });
            await p.settle();
            assert.deepEqual(p.received, [[{ id: 2, reply: true }], [{ id: 1, text: 'héllo' }]]);
            const bytes = new Uint8Array([1, 2, 3]);
            p.a.send({ id: 3, bytes: bytes.buffer }, [bytes.buffer]);
            assert.equal(bytes.byteLength, 0);
            await p.settle();
            assert.deepEqual([...new Uint8Array(p.received[1][1].bytes)], [1, 2, 3]);
        } finally { p.close(); }
    });
}

test('SAB saturation and oversized slot fallback preserve FIFO across both directions', async () => {
    const p = await pair(true);
    try {
        for (let i = 1; i <= 70; ++i) {
            p.a.send({ id: i, side: 'a' });
            p.b.send({ id: i + 100, side: 'b' });
        }
        p.a.send({ id: 71, text: 'x'.repeat(9000) });
        await p.settle();
        assert.deepEqual(p.received[1].map(x => x.id), Array.from({ length: 71 }, (_, i) => i + 1));
        assert.deepEqual(p.received[0].map(x => x.id), Array.from({ length: 70 }, (_, i) => i + 101));
        assert.equal(p.received[1][70].text.length, 9000);
        p.a.send({ id: 72 });
        await p.settle();
        assert.equal(p.received[1].at(-1).id, 72);
    } finally { p.close(); }
});

test('SAB cloning rejection leaves a working transferable path', async () => {
    globalThis.crossOriginIsolated = true;
    const messages = [];
    const port = { postMessage(value, transfer) {
        if (value.etalonTransport === 'attach') throw new Error('host denies SAB');
        messages.push(structuredClone(value, { transfer: transfer ?? [] }));
    } };
    const a = new EtalonWorkerTransport(port, () => {}, true);
    const received = [];
    const b = new EtalonWorkerTransport(port, value => received.push(value));
    assert.equal(a.shared, false);
    a.send({ id: 9, value: 'fallback' });
    for (const message of messages) b.handle(message);
    assert.deepEqual(received, [{ id: 9, value: 'fallback' }]);
});

test('message and binary byte budgets reject before ownership transfer', async () => {
    const p = await pair(false, 4);
    try {
        assert.throws(() => p.a.send({ id: 1, text: 'é'.repeat(40000) }), /budget/);
        const bytes = new ArrayBuffer(5);
        assert.throws(() => p.a.send({ id: 2, bytes }, [bytes]), /budget/);
        assert.equal(bytes.byteLength, 5);
        assert.throws(() => p.a.send({ id: 3 }, [new Uint8Array(1)]), /ArrayBuffer/);
        assert.throws(() => p.b.handle({ etalonTransport: 'configure', binaryByteBudget: 8 }), /configuration/);
        await p.settle();
        assert.deepEqual(p.received, [[], []]);
    } finally { p.close(); }
});

test('wire corruption is rejected without invoking receiver', async () => {
    globalThis.crossOriginIsolated = false;
    const messages = [], received = [];
    const port = { postMessage(value) { messages.push(value); } };
    const a = new EtalonWorkerTransport(port, () => {});
    const b = new EtalonWorkerTransport(port, value => received.push(value));
    a.send({ id: 7, text: 'ok' });
    const original = messages[0].buffer;
    for (const offset of [0, 8, 12, 16]) {
        const corrupt = original.slice(0);
        new Uint8Array(corrupt)[offset] ^= 0xff;
        assert.throws(() => b.handle({ etalonTransport: 'message', buffer: corrupt }));
    }
    assert.throws(() => b.handle({ etalonTransport: 'message', buffer: new ArrayBuffer(0) }));
    assert.throws(() => b.handle({ etalonTransport: 'wake' }), /not attached/);
    assert.deepEqual(received, []);
});

// Node's decoder accepts SAB; Chromium's Web IDL BufferSource contract rejects it.
test('ring decoding supplies an owned ArrayBuffer to browser TextDecoder', async () => {
    const NativeDecoder = globalThis.TextDecoder;
    globalThis.TextDecoder = class extends NativeDecoder {
        decode(input, options) {
            assert.equal(input.buffer instanceof SharedArrayBuffer, false);
            return super.decode(input, options);
        }
    };
    try {
        const { EtalonWorkerTransport: Transport } = await import(
            'data:text/javascript;base64,' + Buffer.from(source + '\n// browser decoder contract').toString('base64')
        );
        globalThis.crossOriginIsolated = true;
        const messages = [], received = [];
        const port = { postMessage(value) { messages.push(value); } };
        const a = new Transport(port, () => {}, true);
        const b = new Transport(port, value => received.push(value));
        while (messages.length) b.handle(messages.shift());
        a.send({ id: 1, text: 'héllo from a shared ring' });
        while (messages.length) b.handle(messages.shift());
        assert.deepEqual(received, [{ id: 1, text: 'héllo from a shared ring' }]);
    } finally { globalThis.TextDecoder = NativeDecoder; }
});
