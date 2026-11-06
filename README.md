# Unfold

Data structures, algorithms and transformers made of glowing particles, controlled by your hand through the webcam. Runs entirely in the browser.

Scenes: Stack, Queue, Linked List, Min-Heap, Bubble Sort, Binary Search, BST Insert, Breadth-First Search, Tokens → Embeddings, Self-Attention, Next-Token Prediction, Transformer Block.

## Run it

Needs Node.js 18+ and Chrome or Edge. A webcam is optional; everything also works with keyboard and mouse.

```
npm install
npm start
```

Open http://localhost:5173 and click **Start with camera** or **Continue without camera**.

## Real model scenes

**Real GPT-2 Attention** and **Prompt Injection** draw GPT-2 small's actual attention weights and next-token probabilities, not hand-picked numbers. Generate the data once (downloads GPT-2, ~550 MB):

```
pip install torch transformers
python tools/gpt2_attention.py
```

This writes `data/real/*.json`. Refresh the app. Edit the prompts in `PRESETS` at the top of the script to try your own.

The prompt injection scene hides "IGNORE ALL PREVIOUS INSTRUCTIONS…" inside an email and measures, layer by layer, how much of the final token's attention lands on that line. The first token is removed and the rest renormalized, because GPT-2 parks most of its attention there (an attention sink). GPT-2 isn't instruction-tuned, so this shows attention mechanics, not whether a modern assistant would obey.

## Controls

| Gesture | Keys | Does |
|---|---|---|
| 🫰 Snap | `Space` | Summon / dissolve the particles |
| ✊ Fist | `F` | Form the scene |
| ✊ Fist + slide left/right | `,` `.` | Scrub back and forth through the steps |
| 🤏 Pinch | `.` | Next step |
| | `Enter` / `Home` | Play / pause, restart |
| ✋ Open hand | `E`, wheel | Pull the Transformer Block apart |
| ☝ Point | click | Select a part and read what it does |
| 🔄 Twist hand | drag | Rotate |
| 🙌 Two hands | `+` `-` | Zoom |
| ✌ Peace | `V` `→` `←` | Next / previous scene |

`?n=100000` lowers the particle count on slower machines.

## How it works

Each scene runs the real algorithm and records a frame per step (`src/scenes/dsa.js`, `src/scenes/transformer.js`). Every element has a stable id and owns a fixed slice of particles across frames (`src/scenes/build.js`), so when two bars swap or a node moves, its own particles glide there. Elements that leave dissolve into background dust.

Hand tracking is MediaPipe Hand Landmarker, on-device. Particle physics runs on the GPU in hand-written WebGL2 with transform feedback.

## Credits

The gesture, particle and rendering engine is forked from [WonderSnap](https://github.com/AkbarSheikh-debug/wondersnap) by Akbar Sheikh (MIT). The scenes and stepping system are new.
