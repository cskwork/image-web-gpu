/**
 * 집중도 분석 모듈
 * VL 모델 출력을 파싱하여 근무 집중도를 판정한다.
 */

// 집중 상태 상수
export const FocusStatus = {
  FOCUSED: 'focused',
  DISTRACTED: 'distracted',
  ABSENT: 'absent',
  UNKNOWN: 'unknown',
};

// 한국어 키워드 기반 분류 규칙
const FOCUSED_KEYWORDS = [
  '집중', '화면', '모니터', '카메라', '보고 있', '바라보고',
  '작업', '업무', '일하', '타이핑', '키보드', '마우스',
  '컴퓨터', '노트북', '문서', '읽고', '쓰고',
  'focusing', 'looking at', 'working', 'screen', 'monitor', 'computer',
  'typing', 'writing', 'reading',
];

const DISTRACTED_KEYWORDS = [
  '딴짓', '다른 곳', '옆', '뒤', '전화', '휴대폰', '스마트폰',
  '졸고', '하품', '기지개', '멍', '산만',
  '안 보', '보지 않', '돌리', '돌아',
  'distracted', 'looking away', 'phone', 'not looking', 'yawning', 'sleeping',
  'turned away', 'side',
];

const ABSENT_KEYWORDS = [
  '없', '비어', '빈 자리', '아무도', '사람이 없',
  'empty', 'no one', 'nobody', 'absent', 'vacant',
];

/**
 * 모델 응답에서 집중 상태 판정
 * @param {string} description - 모델이 생성한 한국어 설명
 * @returns {{ status: string, confidence: number }}
 */
export function classifyFocus(description) {
  if (!description || !description.trim()) {
    return { status: FocusStatus.UNKNOWN, confidence: 0 };
  }

  const text = description.toLowerCase().trim();

  // 1차: 모델이 단일 단어로 답한 경우 (영어 프롬프트 최적화)
  if (/^focused\.?$/.test(text) || /^yes/.test(text)) {
    return { status: FocusStatus.FOCUSED, confidence: 1 };
  }
  if (/^distracted\.?$/.test(text)) {
    return { status: FocusStatus.DISTRACTED, confidence: 1 };
  }
  if (/^absent\.?$/.test(text) || /^no one/.test(text) || /^empty/.test(text)) {
    return { status: FocusStatus.ABSENT, confidence: 1 };
  }

  // 2차: 키워드 스코어링
  const absentScore = countMatches(text, ABSENT_KEYWORDS);
  if (absentScore >= 1) {
    return { status: FocusStatus.ABSENT, confidence: Math.min(absentScore / 2, 1) };
  }

  const distractedScore = countMatches(text, DISTRACTED_KEYWORDS);
  const focusedScore = countMatches(text, FOCUSED_KEYWORDS);

  if (distractedScore > focusedScore) {
    return { status: FocusStatus.DISTRACTED, confidence: Math.min(distractedScore / 3, 1) };
  }

  if (focusedScore > 0) {
    return { status: FocusStatus.FOCUSED, confidence: Math.min(focusedScore / 3, 1) };
  }

  return { status: FocusStatus.UNKNOWN, confidence: 0.3 };
}

function countMatches(text, keywords) {
  let count = 0;
  for (const kw of keywords) {
    if (text.includes(kw.toLowerCase())) {
      count++;
    }
  }
  return count;
}

/**
 * 집중 상태에 따른 UI 정보 반환
 */
export function getFocusDisplayInfo(status) {
  switch (status) {
    case FocusStatus.FOCUSED:
      return { label: '집중 중', color: '#22c55e', bgColor: 'rgba(34,197,94,0.15)', icon: 'check-circle' };
    case FocusStatus.DISTRACTED:
      return { label: '주의 산만', color: '#f59e0b', bgColor: 'rgba(245,158,11,0.15)', icon: 'alert-triangle' };
    case FocusStatus.ABSENT:
      return { label: '자리 비움', color: '#ef4444', bgColor: 'rgba(239,68,68,0.15)', icon: 'x-circle' };
    default:
      return { label: '분석 중...', color: '#94a3b8', bgColor: 'rgba(148,163,184,0.15)', icon: 'loader' };
  }
}
