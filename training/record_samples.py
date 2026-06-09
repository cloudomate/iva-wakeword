#!/usr/bin/env python3
"""Record wake-word training samples on the device, through the SAME reSpeaker
FL channel the daemon uses. A robust replacement for record_wakeword.py.

Fixes the problems that bit us with the old recorder:
  * Auto-calibrates the room noise floor -> adaptive trigger. No more guessing
    --rms, and no more capturing 40 clips of ambient *before* you start talking.
  * Per-clip "speak now" prompt + sustained-onset gating (~120 ms above
    threshold) so a transient noise blip doesn't start a capture.
  * Rejects clips that are too short or too quiet (noise/echo) and re-prompts,
    so the saved count reflects real utterances.
  * Live feedback (duration + peak level) per clip; persistent output dir.

Usage (stop the daemon first to free the mic):
  systemctl --user stop hermes-voice            # XDG_RUNTIME_DIR=/run/user/$(id -u)
  python record_samples.py --phrase "okay iva" --count 25 --out ~/ww/wife_pos
  # negatives: label it anything, then say the near-miss phrases / normal talk
  python record_samples.py --phrase "near-miss negatives" --count 40 --out ~/ww/neg
Then on the Mac:  scp 'iva:~/ww/wife_pos/*.wav' ~/coderepo/micro-wake-word/<dir>/

Env: WAKE_CH (0=FL, 1=FR), IVA_AUDIO_PRESET (for parity; channel count auto-tried).
"""
import os, time, wave, argparse
import numpy as np
import sounddevice as sd

RATE = 16000
BLOCK = 320  # 20 ms frames
WAKE_CH = int(os.environ.get("WAKE_CH", "0"))


def open_input():
    """Open the capture stream; try 6ch (XVF3800) then mono, like the daemon."""
    last = None
    for ch in (6, 1):
        try:
            s = sd.RawInputStream(samplerate=RATE, channels=ch, dtype="int16", blocksize=BLOCK)
            s.start()
            return s, ch
        except Exception as e:
            last = e
    raise RuntimeError(f"no input stream: {last}")


def read_fl(s, ch):
    """Read one frame and extract the FL (WAKE_CH) lane from interleaved audio."""
    raw, _ = s.read(BLOCK)
    a = np.frombuffer(raw, np.int16)
    if ch > 1:
        a = a[WAKE_CH::ch]
    return a


def rms(a):
    return float(np.sqrt(np.mean(a.astype(np.float32) ** 2))) if a.size else 0.0


def calibrate(s, ch, secs=1.5):
    print(f"Calibrating room noise ({secs:.1f}s) — stay quiet...", flush=True)
    t0 = time.monotonic(); vals = []
    while time.monotonic() - t0 < secs:
        vals.append(rms(read_fl(s, ch)))
    floor = float(np.median(vals)) if vals else 0.0
    thr = max(floor * 4.0, 700.0)
    print(f"  noise floor ~{floor:.0f}  ->  trigger threshold {thr:.0f}", flush=True)
    return thr


def capture_one(s, ch, thr, maxlen, silence_s, pad):
    """Wait for a sustained speech onset, then capture until silence/maxlen."""
    pre = int(pad * RATE); ring = []
    onset = 0; need = 6  # ~120 ms (6 x 20 ms) sustained above threshold
    while True:
        a = read_fl(s, ch); ring.append(a.copy())
        tot = sum(x.size for x in ring)
        while tot > pre and len(ring) > 1:
            tot -= ring.pop(0).size
        if rms(a) >= thr:
            onset += 1
            if onset >= need:
                break
        else:
            onset = 0
    frames = list(ring); t0 = time.monotonic(); last = t0; peak = 0.0
    while True:
        a = read_fl(s, ch); frames.append(a.copy()); r = rms(a); peak = max(peak, r)
        now = time.monotonic()
        if r >= thr:
            last = now
        if (now - last) >= silence_s or (now - t0) >= maxlen:
            break
    clip = np.concatenate(frames)[: int(maxlen * RATE)]
    return clip, clip.size / RATE, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", default="okay iva")
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--out", default=os.path.expanduser("~/wakeword_recordings"))
    ap.add_argument("--rms", type=float, default=0.0, help="override trigger (0 = auto-calibrate)")
    ap.add_argument("--maxlen", type=float, default=2.0)
    ap.add_argument("--silence", type=float, default=0.6, help="silence (s) that ends an utterance")
    ap.add_argument("--min-dur", type=float, default=0.35, help="reject clips shorter than this")
    ap.add_argument("--pad", type=float, default=0.25, help="pre-roll seconds kept before onset")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s, ch = open_input()
    print(f"input: {ch}ch, FL=ch{WAKE_CH}  ->  {a.out}", flush=True)
    n = 0
    try:
        thr = a.rms if a.rms > 0 else calibrate(s, ch)
        print(f"\n>>> Say '{a.phrase}' {a.count} times. Wait for 'speak now', say it once, pause ~1s.\n", flush=True)
        while n < a.count:
            print(f"[{n + 1}/{a.count}] speak now...", flush=True)
            clip, dur, peak = capture_one(s, ch, thr, a.maxlen, a.silence, a.pad)
            if dur < a.min_dur or peak < thr * 1.3:
                print(f"    skipped (dur={dur:.2f}s peak={peak:.0f}) — too short/quiet; retry", flush=True)
                continue
            n += 1
            p = os.path.join(a.out, f"{n:03d}.wav")
            w = wave.open(p, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
            w.writeframes(clip.tobytes()); w.close()
            print(f"    [{n}/{a.count}] saved {os.path.basename(p)}  dur={dur:.2f}s peak={peak:.0f}", flush=True)
            time.sleep(0.25)
    finally:
        try:
            s.stop(); s.close()
        except Exception:
            pass
    print(f"\nDone: {n} clips in {a.out}", flush=True)


if __name__ == "__main__":
    main()
