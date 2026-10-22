# OpenSkelly

Open-source Linux controller and Live-mode audio bridge for Home Depot's 12 ft.
Skelly. It reproduces the DecorPro Live sequence and routes a PipeWire
microphone to `12ft Skelly(Live)` with speech-driven mouth movement.

## Requirements

Linux with BlueZ, PipeWire, WirePlumber, `bluetoothctl`, `gatttool`,
`pw-loopback`, `pactl`, and `wpctl`.

## Setup

Pair `12ft Skelly(Live)` in `bluetoothctl` using the PIN supplied with your
device or shown by DecorPro.

```bash
mkdir -p ~/.config/wireplumber/wireplumber.conf.d
cp config/90-openskelly.conf ~/.config/wireplumber/wireplumber.conf.d/
systemctl --user restart wireplumber
./openskelly configure
```

If discovery fails, use `configure --ble-mac AA:... --live-mac BB:...`.

## Usage

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
python3 -m unittest discover -s tests -v   # or .venv/bin/python on macOS
```

See [PROTOCOL.md](PROTOCOL.md) for reverse-engineered protocol details.

OpenSkelly is independent and is not affiliated with or endorsed by Home Depot
or the DecorPro application developers.

## macOS (`skelly_mac.py`)

The same protocol over CoreBluetooth (bleak) and A2DP, plus a two-way talk mode:
the Mac microphone → Whisper (on-device, Apple Silicon) → Claude → macOS `say`
through Skelly's jaw.

```bash
brew install blueutil ffmpeg
python3 -m venv .venv && .venv/bin/pip install -r requirements-mac.txt
export ANTHROPIC_API_KEY=...        # talk mode only
.venv/bin/python skelly_mac.py configure        # Skelly powered on and in range
.venv/bin/python skelly_mac.py say "Happy Halloween!"
.venv/bin/python skelly_mac.py talk
```

`configure` finds the BLE endpoint, switches on Skelly's classic radio, pairs
`12ft Skelly(Live)` (pass `--pin` if your unit asks for one) and writes
`~/.config/openskelly/mac.json`. `lights`, `eyes` and `play` take the same
arguments as the Linux CLI. Before the first `talk`, run
`skelly_mac.py talk --listen-test` and set `vad_threshold` in `mac.json` just above
the room's quiet level. macOS will ask for Bluetooth and Microphone permission for
your terminal on first use.

The voice is picked by `tts` in `mac.json`: `qwen` (default; Qwen3-TTS VoiceDesign,
on-device, the voice is whatever `voice_instruct` describes), `kokoro` (fast
neural voice named by `voice`, e.g. `am_michael`) or `say` (macOS voice named by
`voice`). `voice_fx` is an ffmpeg filter chain applied before the jaw; prepend
`asetrate=44100*0.8,aresample=44100,` to pitch a voice down. Models download on
first use (Qwen is about 1 GB).
