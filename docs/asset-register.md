# Asset register

Every raster that ships with the site, and where it came from.

| File | Use | Source | Alt text |
|------|-----|--------|----------|
| `public/images/keyart-1536.webp`, `public/images/keyart-900.webp` | Landing key art (responsive `srcset`) | Generated 2026-09-24 with GPT Image 2 through the local Codex CLI (`gpt-image-2` skill). 1536x1024 PNG master, converted to WebP (q80 / q78) with ImageMagick. | "어두운 책상 위에 서 있는 검은 삼색 신호등. 맨 아래 청록색 불만 켜져 책상을 비추고, 옆에 노트북과 웹캠이 놓여 있다." |
| `public/images/lens-idle-720.webp` | Camera-off empty state inside the video frame | Generated 2026-09-24 with GPT Image 2 (Codex CLI). 1254x1254 master, resized to 720px WebP. | Decorative (`alt=""`, `aria-hidden`). The empty-state message is real text next to it. |
| `public/og-image.jpg` | Open Graph / social preview (1200x630) | Generated 2026-09-24 with GPT Image 2 (Codex CLI). Cropped and resized to 1200x630 JPEG. | Set in `og:image:alt`. |
| `public/favicon.svg` | Favicon (a three-lamp signal) | Hand-authored SVG. At runtime the app swaps in an inline SVG that lights the current state's lamp. | n/a |

Provenance: each WebP has a `.json` sidecar with its exact generation prompt, and the JPEG carries the prompt in a COM segment. Both were written by `impeccable embed-prompt`. The images show a generic signal head and desk. They contain no people, logos, or personal information.
