# OpenSkelly

Open-source controller for Home Depot's 12 ft. Skelly, built on the reverse-engineered
DecorPro Bluetooth protocol ([PROTOCOL.md](PROTOCOL.md)). Two front ends share
`protocol.py`:

- **Linux** (`openskelly`): the original controller and Live-mode audio bridge —
  lights, eyes, a PipeWire microphone or system audio through the jaw.
- **macOS** (`skelly_mac.py`): the same controls plus a **talking skeleton**: kids
  speak into a microphone, Whisper transcribes on-device, Claude answers in
  character, and an on-device cloned voice speaks through the jaw — in about two
  seconds, with a push-to-talk window or hands-free listening.

OpenSkelly is independent and is not affiliated with or endorsed by Home Depot
or the DecorPro application developers.

## Linux (`openskelly`)

### Requirements

Linux with BlueZ, PipeWire, WirePlumber, `bluetoothctl`, `gatttool`,
`pw-loopback`, `pactl`, and `wpctl`.

### Setup

Pair `12ft Skelly(Live)` in `bluetoothctl` using the PIN supplied with your
device or shown by DecorPro.

```bash
mkdir -p ~/.config/wireplumber/wireplumber.conf.d
cp config/90-openskelly.conf ~/.config/wireplumber/wireplumber.conf.d/
systemctl --user restart wireplumber
./openskelly configure
```

If discovery fails, use `configure --ble-mac AA:... --live-mac BB:...`.

### Usage

```bash
./openskelly start
./openskelly status
./openskelly stop
```

To route Spotify or other system audio to Skelly, use output mode. It selects
Skelly as the default output, moves an existing Spotify stream directly to it,
and automatically restores the Live handshake and route after a Bluetooth drop:

```bash
./openskelly output start
./openskelly output status
./openskelly output stop
```

Control the RGB chest light with a six-digit hex color or brightness from 0 to
100:

```bash
./openskelly lights color ff0000
./openskelly lights color 8000ff --brightness 50
./openskelly lights brightness 25
./openskelly lights off
```

The friendly channel name `chest` is the default. `--channel all` and numeric
channels `0` through `5` are also accepted for compatible hardware variants;
on the tested 12 ft. Skelly, only protocol channel `0` controlled a visible
light. Stop microphone Live mode before changing lights.

Select any of the 20 eye patterns built into Skelly's firmware:

```bash
./openskelly eyes set 1
./openskelly eyes set 20
```

Eye artwork cannot be uploaded through the known DecorPro Bluetooth protocol;
it only selects numbered firmware presets. Stop microphone Live mode before
changing the eyes.

To play any audio file with synchronized mouth animation:

```bash
./openskelly play speech.wav
```

OpenSkelly accepts any format supported by FFmpeg. It normalizes and converts
the input to 44.1 kHz stereo, adds a short lead-in so Bluetooth connection
sounds do not overlap playback, animates the mouth, and disconnects afterward.
This keeps OpenSkelly independent of any particular TTS engine; generate the
audio with your preferred voice system first. Stop microphone Live mode before
using `play`.

Disconnect `12ft Skelly(Live)` from phones first. Only one audio source can own
the classic-Bluetooth connection at a time.

The included WirePlumber override selects 44.1 kHz SBC and enables an AVRCP
compatibility player. The default 2x microphone gain matches DecorPro.

Runtime state and logs are stored under `.skelly-live/`.

Run the offline protocol and routing checks with:

```bash
python3 -m unittest discover -s tests -v
```


## macOS (`skelly_mac.py`)

### Requirements

An Apple Silicon Mac (the speech and voice models run on MLX; measured on an M2
Max, an M1 with 16 GB is fine), macOS 13+, Python 3.12, ffmpeg, and an Anthropic
API key for talk mode. Everything else — Whisper, the voice models, the voice
detector — runs locally and downloads on first use (about 2 GB).

```bash
brew install blueutil ffmpeg switchaudio-osx
python3 -m venv .venv && .venv/bin/pip install -r requirements-mac.txt
```

### Setup

Power the Skelly on, stand within Bluetooth range, and run:

```bash
.venv/bin/python skelly_mac.py configure
```

It finds the BLE control endpoint, switches on the Skelly's classic radio, pairs
`12ft Skelly(Live)` (pass `--pin 1234` if your unit asks; that is the PIN the
DecorPro app shows) and writes `~/.config/openskelly/mac.json`. macOS asks for
Bluetooth and Microphone permission for your terminal the first time.

Then make the voice:

```bash
.venv/bin/python skelly_mac.py voice design --seed 42
.venv/bin/python skelly_mac.py say "Who dares to wake me?"
```

`voice design` renders a reference clip with the Qwen3-TTS VoiceDesign model from
the description in `voice_instruct`; every later line is cloned from that clip, so
Skelly is the same skeleton all week. Different seeds give different skeletons —
audition a few and keep one.

### Talking to Skelly

```bash
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python skelly_mac.py talk
```

A window opens. **Hold SPACE and talk**; release, and Skelly answers in one short
sentence. The window shows what it heard and what he said, a mic level meter,
and four controls:

- **Hands-free** — no key: a turn starts when the Silero neural voice detector
  hears a human voice (cars, wind and the Skelly's own jingle do not count) and
  ends after `vad_silence` seconds of quiet. Short fragments and Whisper's
  silence hallucinations are ignored.
- **Calibrate** — press once with the mic in its final spot. It measures how loud
  the Skelly is at the mic, then has you talk from where a kid stands, and sets
  `barge_threshold` between the two. From then on a kid can interrupt him
  mid-sentence in hands-free mode (in push-to-talk, pressing SPACE interrupts).
- **Model** — Haiku (fastest, default), Sonnet, or Opus for wittier answers.
- **Clear memory** — he remembers the last 20 exchanges; this forgets them.

`F` goes fullscreen, `Esc` leaves it, Ctrl-C or `skelly_mac.py stop` ends the
session cleanly. Each turn prints a `timing:` line (speech-to-text, first
sentence from Claude, first audio) so you can see where the seconds go.

How a turn works: the clip goes to Whisper (small, on-device), the text goes to
Claude with the persona and backstory, and the reply streams back sentence by
sentence. Each sentence is cloned into the voice as it arrives and streamed
through the effects chain into the jaw, with the next sentence rendering while
the current one plays. If the first sound would take longer than about 1.2 s, a
short cached interjection ("Hmmmm.", "Ooooh.") covers the wait. If Claude's
safety classifier cuts a reply off (it happens on spooky banter), the turn is
retried once with a gentler instruction before he gives up on it.

### One connection per session

The Skelly plays a loud jingle every time its audio link connects or drops, so
the Mac holds one connection for the whole session. `talk` and `serve` keep it
up (and reconnect if it fails), and `say`, `play`, `lights` and `eyes` forward
to the running process over a local socket instead of opening their own link.
Without a server running they connect on their own. While connected, the Mac's
own sound stays on its speakers; only Skelly commands reach the jaw.

```bash
.venv/bin/python skelly_mac.py serve &
.venv/bin/python skelly_mac.py say "Happy Halloween!"
.venv/bin/python skelly_mac.py play scream.mp3          # any ffmpeg format, loudness-normalized
.venv/bin/python skelly_mac.py lights color ff2200 --brightness 60
.venv/bin/python skelly_mac.py eyes 7
.venv/bin/python skelly_mac.py stop
```

### Configuration (`~/.config/openskelly/mac.json`)

Only keys you set are stored; everything else uses the defaults in
`skelly_mac.py`.

| Key | Default | What it does |
|---|---|---|
| `persona` | built in | The system prompt: who Skelly is, his backstory, one-sentence replies, kid-safe rules. Replace it to change his character. |
| `model` | `claude-haiku-5-5` | Chat model (also switchable in the window). |
| `tts` | `clone` | Voice engine: `clone` (reference clip), `qwen` (VoiceDesign, a new voice every line), `kokoro` (fast neural, `voice` names it), `say` (macOS voice named by `voice`). |
| `voice_instruct`, `voice_seed` | built in | What `voice design` renders: the description and the seed. |
| `voice_fx` | pitch-down, light echo, gain, limiter | ffmpeg filter chain before the jaw. Stream-safe filters only (no `loudnorm`); the jaw opens on level, so avoid anything that fills the gaps between words. |
| `jaw_ratio` | `0` | Optional syllable gate that ducks the dips between words so the jaw closes more; `0.3`–`0.5` to try it. |
| `filler` | `true` | Cached interjections while a slow reply starts. |
| `mic_device` | system default | Microphone by name or substring, e.g. `"Yeti"`. Falls back to the default mic with a warning if absent. |
| `vad`, `vad_prob` | `silero`, `0.5` | Hands-free speech detection. `energy` uses the plain level gate on `vad_threshold` instead. |
| `vad_silence` | `1.2` | Seconds of quiet that end a hands-free turn. |
| `barge_threshold` | set by Calibrate | Level a voice must exceed while Skelly talks to interrupt him. |
| `whisper_model` | `mlx-community/whisper-small-mlx` | Speech-to-text model. |

`talk --listen-test` prints the mic level and voice probability for five seconds
so you can see what the detector makes of your street. `talk --vad` starts
hands-free without the window. `jaw test` plays a level staircase and burst
trains so you can watch what the jaw responds to (on the tested unit: nearly any
sound opens it, and it needs about a quarter second per open/close).

### Deploying outdoors

A tested setup: the Mac indoors, the Skelly in the yard, and a near-field
microphone at the fence. A directional USB mic (the Yeti in cardioid, pointed at
the kids, Skelly behind it) works on a dry night; for a week outside use an
outdoor security microphone (IP66, vandal-rated, 5 m pickup, line-level out) into
a USB audio adapter at the Mac, with its 12 V supply indoors on the same cable
run. Hands-free with Silero handles the street; a button that types SPACE (an
arcade button on a Pico, or a USB HID button) gives passers-by push-to-talk.

### Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Offline: protocol frames, the voice detector and segmenter, effects levels and
jaw gaps, streaming playback, interrupts, refusal handling, config saving.
