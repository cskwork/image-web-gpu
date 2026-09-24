import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  STORAGE_KEY,
  HISTORY_LIMIT,
  createSession,
  recordVerdict,
  focusRate,
  loadSession,
  saveSession,
  clearSession,
} from '../src/session-store.js';

function memoryStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
  };
}

test('recordVerdict counts every verdict in total and the matching bucket', () => {
  let s = createSession(1000);
  s = recordVerdict(s, 'focused', 'focused', 1);
  s = recordVerdict(s, 'distracted', 'distracted', 2);
  s = recordVerdict(s, 'unknown', '???', 3);
  assert.deepEqual(s.stats, { total: 3, focused: 1, distracted: 1, absent: 0 });
  assert.equal(s.history[0].status, 'unknown');
  assert.equal(s.history[2].status, 'focused');
});

test('recordVerdict does not mutate the previous session', () => {
  const s0 = createSession(1000);
  recordVerdict(s0, 'focused', 'x', 1);
  assert.equal(s0.stats.total, 0);
  assert.equal(s0.history.length, 0);
});

test('history keeps only the newest HISTORY_LIMIT entries', () => {
  let s = createSession(0);
  for (let i = 0; i < HISTORY_LIMIT + 5; i++) s = recordVerdict(s, 'absent', `n${i}`, i);
  assert.equal(s.history.length, HISTORY_LIMIT);
  assert.equal(s.history[0].text, `n${HISTORY_LIMIT + 4}`);
  assert.equal(s.stats.total, HISTORY_LIMIT + 5);
});

test('unrecognised status is stored as unknown', () => {
  const s = recordVerdict(createSession(0), 'sleepy', 'x', 1);
  assert.equal(s.history[0].status, 'unknown');
  assert.equal(s.stats.total, 1);
});

test('focusRate is null without analyses and rounds otherwise', () => {
  assert.equal(focusRate({ total: 0, focused: 0 }), null);
  assert.equal(focusRate({ total: 3, focused: 2 }), 67);
});

test('save/load round-trips a session', () => {
  const storage = memoryStorage();
  let s = recordVerdict(createSession(5), 'focused', 'focused', 10);
  assert.equal(saveSession(storage, s), true);
  assert.deepEqual(loadSession(storage), s);
});

test('loadSession rejects corrupt or malformed data', () => {
  const storage = memoryStorage();
  storage.setItem(STORAGE_KEY, '{not json');
  assert.equal(loadSession(storage), null);
  storage.setItem(STORAGE_KEY, JSON.stringify({ startedAt: 1, stats: { total: -1 }, history: [] }));
  assert.equal(loadSession(storage), null);
  assert.equal(loadSession(null), null);
});

test('saveSession reports failure when storage throws', () => {
  const throwing = { setItem() { throw new Error('QuotaExceeded'); } };
  assert.equal(saveSession(throwing, createSession(0)), false);
});

test('clearSession removes the stored session', () => {
  const storage = memoryStorage();
  saveSession(storage, createSession(0));
  clearSession(storage);
  assert.equal(loadSession(storage), null);
});
