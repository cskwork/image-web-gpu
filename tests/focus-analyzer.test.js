import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classifyFocus, FocusStatus } from '../src/focus-analyzer.js';

test('single-word model answers map directly', () => {
  assert.equal(classifyFocus('focused').status, FocusStatus.FOCUSED);
  assert.equal(classifyFocus('Distracted.').status, FocusStatus.DISTRACTED);
  assert.equal(classifyFocus('absent').status, FocusStatus.ABSENT);
});

test('empty response is unknown', () => {
  assert.equal(classifyFocus('').status, FocusStatus.UNKNOWN);
  assert.equal(classifyFocus(null).status, FocusStatus.UNKNOWN);
});

test('keyword fallback prefers absent, then distracted over focused', () => {
  assert.equal(classifyFocus('The chair is empty').status, FocusStatus.ABSENT);
  assert.equal(classifyFocus('person looking away, using a phone').status, FocusStatus.DISTRACTED);
  assert.equal(classifyFocus('person typing at the computer').status, FocusStatus.FOCUSED);
});
