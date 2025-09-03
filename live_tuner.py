# live_tuner.py — ESP32-like live tuner using your computer mic (terminal readout)
# Requires: sounddevice, numpy  (pip install sounddevice numpy)
import numpy as np, sounddevice as sd, math, sys, time, queue, argparse

# ---------- Config ----------
FMIN, FMAX = 70.0, 1200.0        # pitch search range (Hz)
FS = 48000                       # target sample rate (we'll ask for this)
WIN = 4096                       # analysis window (samples) ~85 ms @ 48 kHz
HOP = 2048                       # hop size (prints ~FS/HOP updates/sec)
CLIP_FRAC = 0.6                  # center-clipping threshold fraction of peak
AMP_GATE = 0.02                  # min RMS to attempt pitch (0..1 after norm)
SMOOTH_ALPHA = 0.35              # EMA smoothing for f0 (0..1), higher=snappier
PRINT_HZ = 10                    # max prints per second (rate-limit terminal)

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def midi_from_freq(f):
    return 69 + 12.0*math.log2(f/440.0)

def freq_from_midi(m):
    return 440.0 * (2.0**((m-69)/12.0))

def nearest_note(f):
    if not (f and f>0): return None
    m = round(midi_from_freq(f))
    name = NOTE_NAMES[int(m)%12]; octv = int(m//12) - 1
    fref = freq_from_midi(m)
    cents = 1200.0*math.log2(f/fref)
    return name, octv, fref, cents

def parabolic_interp(y, i):
    # refine peak location using a parabola fit around i
    if i <= 0 or i >= len(y)-1: return float(i), y[i]
    xv = 0.5*(y[i-1]-y[i+1])/(y[i-1]-2*y[i]+y[i+1]) + i
    yv = y[i] - 0.25*(y[i-1]-y[i+1])*(xv-i)
    return xv, yv

def estimate_f0_autocorr(x, fs, fmin=FMIN, fmax=FMAX, clip_frac=CLIP_FRAC):
    # x: 1D float np.array
    if np.max(np.abs(x)) < 1e-9:
        return 0.0
    # remove DC and normalize
    x = x - np.mean(x)
    mx = np.max(np.abs(x))
    if mx <= 0: return 0.0
    x = x / mx
    # center clipping
    T = clip_frac
    y = np.where(x >  T, x - T, np.where(x < -T, x + T, 0.0))
    # find ACF via FFT
    n = 1 << int(np.ceil(np.log2(2*len(y)-1)))
    Y = np.fft.rfft(y, n=n)
    ac = np.fft.irfft(np.abs(Y)**2, n=n)[:len(y)]
    # search lags
    lag_min = int(fs / fmax)
    lag_max = int(fs / fmin)
    if lag_min < 1: lag_min = 1
    lag_max = min(lag_max, len(ac)-1)
    if lag_min >= lag_max:
        return 0.0
    # peak pick
    idx = lag_min + np.argmax(ac[lag_min:lag_max])
    xv, _ = parabolic_interp(ac, idx)
    if xv <= 0: return 0.0
    f0 = fs / xv
    if f0 < fmin or f0 > fmax: return 0.0
    return float(f0)

def needle(cents, width=25, span=50):
    # ASCII needle bar centered at 0 cents; span = +/- span
    c = max(-span, min(span, cents))
    mid = width//2
    pos = int(round((c/span)*(mid)))
    # build |-----|-----|--^--|-----|-----|
    s = ["-"]*width
    center = mid
    marks = [0, mid//2, center, center + (mid//2)]
    for m in [0, center, width-1]: pass
    # put center bar
    s[center] = "|"
    # place caret
    caret = center + pos
    caret = max(0, min(width-1, caret))
    s[caret] = "^"
    return "|" + "".join(s) + "|"

def list_devices():
    print(sd.query_devices())

def run(device=None):
    q = queue.Queue()
    last_print = 0.0
    fs = FS
    # build stream
    try:
        sd.default.samplerate = FS
    except Exception:
        pass

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # Open input stream mono
    with sd.InputStream(channels=1, samplerate=fs, blocksize=HOP, device=device, callback=callback):
        # If host adjusted samplerate, capture it
        fs = sd.default.samplerate or FS
        # rolling buffer
        buf = np.zeros(0, dtype=np.float32)
        f0_smooth = 0.0
        print("🎤 Live tuner started. Press Ctrl+C to quit.")
        while True:
            block = q.get()
            block = block.reshape(-1)  # mono
            # simple amplitude gate to avoid printing silence
            if np.sqrt(np.mean(block**2)) < 1e-5:
                # still collect to keep timing
                pass
            buf = np.concatenate([buf, block])
            while len(buf) >= (WIN + 0):
                win = buf[-WIN:]  # last WIN samples
                buf = buf[HOP:]   # slide by HOP
                # amplitude/RMS gate (after norm)
                rms = np.sqrt(np.mean(win**2))
                if rms < AMP_GATE:
                    # low signal; print idle less often
                    now = time.time()
                    if now - last_print > 1.0/PRINT_HZ:
                        sys.stdout.write("\r(listening...)                              ")
                        sys.stdout.flush()
                        last_print = now
                    continue
                f0 = estimate_f0_autocorr(win.astype(np.float64), fs)
                if f0 <= 0:
                    now = time.time()
                    if now - last_print > 1.0/PRINT_HZ:
                        sys.stdout.write("\r(no pitch)                                 ")
                        sys.stdout.flush()
                        last_print = now
                    continue
                # smooth f0
                if f0_smooth == 0.0:
                    f0_smooth = f0
                else:
                    f0_smooth = SMOOTH_ALPHA*f0 + (1.0-SMOOTH_ALPHA)*f0_smooth

                note = nearest_note(f0_smooth)
                now = time.time()
                if note and (now - last_print) > (1.0/PRINT_HZ):
                    name, octv, fref, cents = note
                    direction = "⬇️  tune DOWN" if cents > 0 else ("⬆️  tune UP" if cents < 0 else "✅ in tune")
                    bar = needle(cents, width=27, span=50)
                    msg = f"{name}{octv:1d}  f0={f0_smooth:6.1f} Hz   {cents:+5.1f} cents  {direction}   {bar}"
                    sys.stdout.write("\r" + msg + " " * 4)
                    sys.stdout.flush()
                    last_print = now

def main():
    ap = argparse.ArgumentParser(description="Live chromatic tuner (mic → terminal)")
    ap.add_argument("--device", type=int, default=None, help="Input device index (see --list-devices)")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = ap.parse_args()
    if args.list_devices:
        list_devices()
        return
    try:
        run(device=args.device)
    except KeyboardInterrupt:
        print("\nBye!")
    except Exception as e:
        print("\nError:", e)
        print("Tip: try `python3 live_tuner.py --list-devices` and select a valid input index.")

if __name__ == "__main__":
    main()
