/**
 * 세션 기록 저장소
 * 집계와 최근 분석 기록을 localStorage에 보관해 새로고침 후에도 세션이 이어지도록 한다.
 * 저장소(storage)는 주입받으므로 브라우저 밖(node:test)에서도 검증할 수 있다.
 */

export const STORAGE_KEY = 'focus-monitor:session:v1';
export const HISTORY_LIMIT = 20;
const TEXT_LIMIT = 280;
const STATUSES = ['focused', 'distracted', 'absent', 'unknown'];

export function createSession(now = Date.now()) {
  return {
    startedAt: now,
    stats: { total: 0, focused: 0, distracted: 0, absent: 0 },
    history: [],
  };
}

/**
 * 판정 1건을 반영한 새 세션을 반환한다 (원본은 변경하지 않음).
 * 기존 동작과 같이 total은 모든 판정에서 증가하고, 알 수 없는 판정은 개별 집계에 넣지 않는다.
 */
export function recordVerdict(session, status, text, at = Date.now()) {
  const safeStatus = STATUSES.includes(status) ? status : 'unknown';
  const stats = { ...session.stats, total: session.stats.total + 1 };
  if (safeStatus in stats && safeStatus !== 'total') stats[safeStatus] += 1;

  const entry = { status: safeStatus, text: String(text ?? '').slice(0, TEXT_LIMIT), at };
  const history = [entry, ...session.history].slice(0, HISTORY_LIMIT);
  return { ...session, stats, history };
}

/** 집중률(%) — 분석이 없으면 null */
export function focusRate(stats) {
  if (!stats || stats.total <= 0) return null;
  return Math.round((stats.focused / stats.total) * 100);
}

function isValidSession(value) {
  if (!value || typeof value !== 'object') return false;
  const { stats, history, startedAt } = value;
  if (typeof startedAt !== 'number' || !stats || !Array.isArray(history)) return false;
  return ['total', 'focused', 'distracted', 'absent'].every(
    (k) => Number.isInteger(stats[k]) && stats[k] >= 0,
  );
}

/** 저장된 세션을 읽는다. 없거나 손상되었으면 null */
export function loadSession(storage) {
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!isValidSession(parsed)) return null;
    return {
      startedAt: parsed.startedAt,
      stats: { ...parsed.stats },
      history: parsed.history
        .filter((h) => h && STATUSES.includes(h.status) && typeof h.at === 'number')
        .slice(0, HISTORY_LIMIT)
        .map((h) => ({ status: h.status, text: String(h.text ?? '').slice(0, TEXT_LIMIT), at: h.at })),
    };
  } catch {
    return null;
  }
}

/** 세션 저장. 저장소가 가득 찼거나 차단된 경우 false */
export function saveSession(storage, session) {
  try {
    storage?.setItem(STORAGE_KEY, JSON.stringify(session));
    return true;
  } catch {
    return false;
  }
}

export function clearSession(storage) {
  try {
    storage?.removeItem(STORAGE_KEY);
  } catch {
    /* 저장소 접근 불가: 무시 */
  }
}
