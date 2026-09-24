---
name: AI 근무 집중도 모니터
description: A night-street signal head that reads your focus from across the desk.
colors:
  night-ground: "#0b1017"
  night-ground-raised: "#101722"
  night-ground-well: "#172030"
  housing-black: "#0d0f12"
  housing-rim: "#252a31"
  road-white-ink: "#ece9e1"
  mist-ink: "#aab1bb"
  dusk-ink: "#8a929d"
  go-teal: "#1fd1ad"
  go-teal-ink: "#03211b"
  signal-amber: "#ffb020"
  stop-red: "#ff4d3f"
  plate-white: "#f3f1ea"
  plate-ink: "#121418"
  guide-blue: "#0a58b0"
  lane-yellow: "#ffd43b"
typography:
  display:
    fontFamily: "'Pretendard Variable', Pretendard, system-ui, sans-serif"
    fontSize: "clamp(2.5rem, 4.4vw, 4.25rem)"
    fontWeight: 800
    lineHeight: 1.08
    letterSpacing: "-0.035em"
  verdict:
    fontFamily: "'Pretendard Variable', Pretendard, system-ui, sans-serif"
    fontSize: "clamp(3rem, 4.6vw, 4.5rem)"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "-0.04em"
  title:
    fontFamily: "'Pretendard Variable', Pretendard, system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 800
    letterSpacing: "-0.02em"
  body:
    fontFamily: "'Pretendard Variable', Pretendard, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
  sign-numeral:
    fontFamily: "'Barlow Semi Condensed', 'Pretendard Variable', sans-serif"
    fontSize: "28px"
    fontWeight: 700
    fontFeature: "'tnum'"
  led:
    fontFamily: "Doto, ui-monospace, monospace"
    fontSize: "40px"
    fontWeight: 800
rounded:
  plate: "6px"
  control: "8px"
  panel: "10px"
  frame: "12px"
  housing: "14px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "28px"
components:
  button-go:
    backgroundColor: "{colors.go-teal}"
    textColor: "{colors.go-teal-ink}"
    rounded: "{rounded.control}"
    height: "52px"
    padding: "0 24px"
  button-go-active:
    backgroundColor: "{colors.stop-red}"
    textColor: "#ffffff"
  button-guide:
    backgroundColor: "{colors.guide-blue}"
    textColor: "#ffffff"
    rounded: "{rounded.panel}"
    height: "60px"
  button-plate:
    backgroundColor: "{colors.plate-white}"
    textColor: "{colors.plate-ink}"
    rounded: "{rounded.control}"
    height: "44px"
  select-plate:
    backgroundColor: "{colors.plate-white}"
    textColor: "{colors.plate-ink}"
    rounded: "{rounded.control}"
    height: "44px"
  tally-plate:
    backgroundColor: "{colors.plate-white}"
    textColor: "{colors.plate-ink}"
    rounded: "{rounded.panel}"
  guide-sign:
    backgroundColor: "{colors.guide-blue}"
    textColor: "#ffffff"
    rounded: "{rounded.frame}"
---

# Design System: AI 근무 집중도 모니터

## Overview

**Creative North Star: "The Signal at the Desk"**

The app is a traffic signal head standing beside your camera at night. Three lamps are always there. Exactly one is lit, and you can read it from across the room. Everything that is not the verdict is an auxiliary road sign bolted below it: white plates with a black inner rule for facts and counts, and a guide-sign blue board for setup. The ground is the ink-blue of a street after dark, taken from the key art.

The world is dark because of where it's used: someone at a desk in the evening, the app in a side window, glancing at it from the corner of their eye. Density is low and the scale is large. The verdict word and the lit lamp carry the screen, and the controls stay quiet until they're needed.

Motion follows how a signal behaves. A lamp comes on instantly with no fade-in. The lamp it replaced glows for one beat (about 520ms) and then goes dark. While a frame is being analyzed, a single dot blinks under the LED window. Nothing scrolls, loops, or floats.

**Key Characteristics:**
- Every state is always shown, with one of them lit (three lamps, twenty history cells).
- Sign plates carry the information. There are no generic cards.
- LED dot-matrix numerals appear only where a signal would show numbers.
- One dark scene, taken from the lighting of a desk at night.

## Colors

The palette has three signal lights, two kinds of road sign, and a night ground.

### Primary
- **Go Teal** (#1fd1ad): the "집중" lamp, the start button, the focus-rate meter and LED digits. It's a blue-green LED rather than a leafy green, because that's what real LED signals emit.
- **Signal Amber** (#ffb020): the "산만" lamp and warning notices.
- **Stop Red** (#ff4d3f): the "부재" lamp, the stop-monitoring state, and error notices.

### Secondary
- **Guide Blue** (#0a58b0): the Korean road guide-sign blue. Used for the primary landing action and the model-setup board, always with a white inner rule.
- **Plate White** (#f3f1ea) with **Plate Ink** (#121418): auxiliary-sign plates for feature facts, the tally, selects, and secondary buttons.

### Tertiary
- **Lane Yellow** (#ffd43b): the road-marking yellow. Used only for focus rings and text selection.

### Neutral
- **Night Ground** (#0b1017): the page field.
- **Night Ground Raised / Well** (#101722 / #172030): unlit meter segments and the scrollbar thumb.
- **Housing Black** (#0d0f12) with **Housing Rim** (#252a31): the signal head, the step row, and the LED window.
- **Road White Ink** (#ece9e1): primary text. **Mist Ink** (#aab1bb): secondary text. **Dusk Ink** (#8a929d): tertiary labels and timestamps. All three stay above 4.5:1 on Night Ground.

### Named Rules
**The Three Lights Rule.** Teal, amber, and red mean the three verdicts and nothing else. Never use them as decoration or for unrelated categories.
**The Label Beside the Light Rule.** A status color never appears without its word or pictogram (lamp names, tally labels, history status text).

## Typography

**Display / Body Font:** Pretendard Variable, self-hosted as a dynamic subset
**Sign Numerals:** Barlow Semi Condensed 600/700, which comes from highway-sign lettering
**LED Font:** Doto 800 (dot matrix)

**Character:** heavy Korean gothic for verdicts and headlines, condensed sign lettering for counts and times, and dot-matrix only where a real signal would show numbers.

### Hierarchy
- **Verdict** (800, clamp(3rem, 4.6vw, 4.5rem), 1): the current state word, colored by state.
- **Display** (800, clamp(2.5rem, 4.4vw, 4.25rem), 1.08, -0.035em): the landing headline only.
- **Title** (800, 19px): sign-board headings.
- **Body** (400, 15px, 1.6): descriptions and model output.
- **Label** (600–700, 12.5–15px): control labels, lamp names, section headings.
- **Sign numeral** (700, 22–28px, tabular): tally counts, timestamps, cache size.
- **LED** (800, 28–40px): focus rate in the readout and in the signal's countdown window.

### Named Rules
**The Keep-All Rule.** Korean text uses `word-break: keep-all` so lines break between words, not inside them.
**The LED Is Earned Rule.** Doto is used only for numbers a signal would display. Never use it for prose or labels.

## Layout

On desktop (1080px and up), the page is two columns. On the left, a stage holds the camera frame (16:10) with the vertical signal head beside it, the start controls under both, and the guide-sign setup board at the bottom. On the right is a sticky readout column (320–400px) with the verdict, focus rate, tally plate, step row, and history. The max width is 1520px, with 28px gutters and 32px between the columns.

Below 1080px the readout stacks under the stage. Below 760px the signal head turns horizontal (red, amber, green from left to right, like Korean horizontal signals), the lens shrinks from 84px to 64px, the camera frame goes to 4:3, and the primary buttons fill the full width. On mobile, the landing puts the key art in the top 42dvh and the copy below it.

Spacing rhythm: 8, 12, 16, and 28px. Headings get more space above than below.

## Elevation & Depth

Depth comes from physical objects, not from floating cards. Neutral drop shadows (`0 22px 40px -18px rgba(0,0,0,.9)`) sit under the signal head and the sign boards, the way a housing sits off a wall. The only colored light in the system comes from lit lamps: their glow is emitted light, and it appears only on the active lamp.

### Named Rules
**The Only Light Is a Lamp Rule.** Colored glow is allowed only on a lit signal lens. Buttons, boards, and text get neutral shadows or none.

## Shapes

Signs have gently rounded corners: 6px for plates, 8px for controls, 10px for panels, 12px for the camera frame and the setup board, and 14px for the signal housing. Signage is built from inner rules: white plates carry a 1.5–2px black line inset 3–4px from the edge, and guide-blue boards carry a 2px white line inset 4–6px. Lenses are full circles with a visor rim across the top arc.

## Components

### Signal Head (signature)
A housing with three visored lenses (부재, 산만, 집중 from top to bottom) and an LED countdown window showing the focus rate. An unlit lens shows a dim LED dot grid and a ghost pictogram. A lit lens shows bright LED dots, a dark pictogram, and emitted glow. States change instantly, and the previous lamp decays over 520ms (switched off under reduced motion). The whole head is `role="img"` with a live `aria-label`.

### Buttons
- **Go (primary):** Go Teal on dark teal ink, 52px tall, 8px radius. Switches to Stop Red with white text while monitoring. When disabled it becomes a housing-colored slab.
- **Guide:** Guide Blue with a white inner rule, 60px tall. Used for the landing entry and nowhere else.
- **Plate:** Plate White with a black inner rule. Used for secondary actions on the setup board.
- **Ghost:** transparent with a 1px hairline. Used for low-stakes actions (reset session, dismiss).
- **Focus:** a 3px Lane Yellow outline at 3px offset on every interactive element.

### Inputs / Selects
White plate selects, 44px tall, with a drawn chevron.

### Tally Plate
A white auxiliary sign that lists state and count the way a road sign lists destination and distance, with a lamp-colored key dot next to each label.

### Step Row
Twenty LED cells in a housing strip, oldest to newest from left to right. Empty cells stay dark, and the newest cell gets a white ring.

### Notices
Inline, never modal. Amber-tinted by default, red-tinted for errors. Each one names the problem and how to fix it.

## Do's and Don'ts

### Do:
- **Do** show all three lamps at all times and light only one.
- **Do** use white plates or the guide-blue board for any new grouping of controls or facts.
- **Do** self-host every font and image. The site runs cross-origin isolated (COOP/COEP).
- **Do** switch the signal to the horizontal arrangement on narrow screens.

### Don't:
- **Don't** add colored glow to anything except a lit lamp.
- **Don't** use eyebrow or kicker labels above headings.
- **Don't** animate lamps on with fades or tweens. Signals snap on.
- **Don't** add looping or scrolling motion. The analysis indicator is a stepped blink.
