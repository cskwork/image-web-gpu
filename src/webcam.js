/**
 * 웹캠 캡처 모듈
 * getUserMedia로 웹캠 스트림을 관리하고 프레임을 캡처한다.
 */

let stream = null;
let videoElement = null;

/**
 * 웹캠 시작
 * @param {HTMLVideoElement} video - 비디오 엘리먼트
 * @returns {Promise<MediaStream>}
 */
export async function startWebcam(video) {
  videoElement = video;

  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
  });

  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadeddata = () => {
      video.play();
      resolve();
    };
    if (video.readyState >= 2) {
      video.play();
      resolve();
    }
  });

  // 카메라 워밍업 대기 (첫 프레임 검은 화면 방지)
  await new Promise((r) => setTimeout(r, 500));

  return stream;
}

/**
 * 웹캠 중지
 */
export function stopWebcam() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  if (videoElement) {
    videoElement.srcObject = null;
    videoElement = null;
  }
}

/**
 * 현재 프레임을 data URL로 캡처
 * @param {number} resolution - 캡처 해상도 (정사각형)
 * @returns {string} JPEG data URL
 */
export function captureFrame(resolution = 384) {
  if (!videoElement) return null;

  const canvas = document.createElement('canvas');
  canvas.width = resolution;
  canvas.height = resolution;
  const ctx = canvas.getContext('2d');

  // 비디오 중앙을 정사각형으로 크롭
  const vw = videoElement.videoWidth;
  const vh = videoElement.videoHeight;
  const size = Math.min(vw, vh);
  const sx = (vw - size) / 2;
  const sy = (vh - size) / 2;

  ctx.drawImage(videoElement, sx, sy, size, size, 0, 0, resolution, resolution);

  return canvas.toDataURL('image/jpeg', 0.85);
}

/**
 * 웹캠 활성 상태 확인
 */
export function isActive() {
  return stream !== null && stream.active;
}
