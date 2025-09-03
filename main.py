# ===== Guitarrón note analysis: FFT, pitch, harmonics, timbre =====
# How to use:
# - Put your path in WAV_PATH below (your Mac path is already set).
# - Run. It prints a summary, saves a CSV of features, and shows 3 plots.

import os, math, re, numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, get_window, find_peaks, hilbert

# ---------- config ----------
WAV_PATH = r"/Users/angelramirez/Desktop/Desktop- Angels Mac-Mini/physics4bl/GuitarronA.wav"
BANDPASS = (50.0, 4000.0)   # Hz, keep instrument band; adjust if needed
FMIN, FMAX = 40.0, 1500.0   # Hz, search range for f0
ROLL_PERCENT = 0.85         # spectral roll-off (85% energy)

ENV_SMOOTH_SEC = 0.01       # RMS/Env smoothing for envelope (10 ms)

# ---------- helpers ----------
def butter_bandpass(x, fs, lo, hi, order=4):
    ny = 0.5*fs
    lo = max(1.0, lo)
    hi = min(ny-50.0, hi)
    b, a = butter(order, [lo/ny, hi/ny], btype="band")
    return filtfilt(b, a, x)

def parabolic_interp(y, i):
    if i<=0 or i>=len(y)-1:
        return float(i), y[i]
    xv = 0.5*(y[i-1]-y[i+1])/(y[i-1]-2*y[i]+y[i+1]) + i
    yv = y[i] - 0.25*(y[i-1]-y[i+1])*(xv-i)
    return xv, yv

def autocorr_f0(seg, fs, fmin=FMIN, fmax=FMAX):
    seg = seg - np.mean(seg)
    if np.max(np.abs(seg)) < 1e-8:
        return np.nan
    seg = seg/np.max(np.abs(seg))
    w = get_window("hann", len(seg))
    x = seg*w
    n = 1<<int(np.ceil(np.log2(2*len(x)-1)))
    X = np.fft.rfft(x, n=n)
    ac = np.fft.irfft(np.abs(X)**2, n=n)[:len(x)]
    lag_min = int(fs/fmax)
    lag_max = int(fs/fmin)
    lag_max = min(lag_max, len(ac)-1)
    if lag_min>=lag_max: return np.nan
    idx = lag_min + np.argmax(ac[lag_min:lag_max])
    xv, _ = parabolic_interp(ac, idx)
    if xv <= 0: return np.nan
    f0 = fs/xv
    return f0 if (f0>=fmin and f0<=fmax) else np.nan

def nearest_note(f):
    # A4 = 440 Hz reference
    if not np.isfinite(f) or f<=0: return None
    midi = round(69 + 12*np.log2(f/440.0))
    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    name = note_names[int(midi)%12]
    octave = int(midi//12) - 1
    f_ref = 440.0 * (2.0**((midi-69)/12))
    cents = 1200*np.log2(f/f_ref)
    return dict(midi=int(midi), name=name, octave=octave, f_ref=f_ref, cents=cents)

def spectral_centroid(freqs, mags):
    p = np.sum(mags)
    return np.sum(freqs*mags)/p if p>0 else np.nan

def spectral_rolloff(freqs, mags, pct=0.85):
    cumsum = np.cumsum(mags)
    target = pct * cumsum[-1]
    k = np.searchsorted(cumsum, target)
    return freqs[min(k, len(freqs)-1)]

def window_at_peak(x, fs, win_sec=0.6):
    # grab a chunk around global peak to avoid silence/handling noise
    N = len(x); peak = np.argmax(np.abs(x))
    half = int(0.5*win_sec*fs)
    a = max(0, peak-half); b = min(N, peak+half)
    return x[a:b], a, b

def envelope_rms(x, fs, smooth_sec=0.01):
    # RMS envelope using moving window
    wlen = max(1, int(smooth_sec*fs))
    pad = wlen//2
    x2 = np.pad(x**2, (pad,pad), mode="edge")
    c = np.convolve(x2, np.ones(wlen)/wlen, mode="same")[pad:-pad]
    return np.sqrt(np.maximum(c,1e-12))

# ---------- load ----------
fs, data = wavfile.read(WAV_PATH)
data = data.astype(np.float32)
if data.ndim==2:
    data = data.mean(axis=1)
# normalize if integer-like
mx = np.max(np.abs(data))
if mx>1.1: data /= mx

# band-limit to instrument band
x = butter_bandpass(data, fs, *BANDPASS)

# take a focused segment around the loudest part
seg, i0, i1 = window_at_peak(x, fs, win_sec=1.0)

# ---------- pitch (f0) ----------
# skip a brief attack; analyze ~250 ms starting 120 ms after peak
offset = int(0.12*fs)
wlen = int(0.25*fs)
start = min(max(0, offset), max(0, len(seg)-wlen))
f0 = autocorr_f0(seg[start:start+wlen], fs)

note_info = nearest_note(f0) if np.isfinite(f0) else None

# ---------- FFT & harmonics ----------
win = get_window("hann", len(seg))
NFFT = 1<<int(np.ceil(np.log2(len(seg))))
S = np.fft.rfft(seg*win, n=NFFT)
freqs = np.fft.rfftfreq(NFFT, 1/fs)
mag = np.abs(S)

# peak picking in spectrum (avoid DC)
peaks, _ = find_peaks(mag, height=np.max(mag)*0.03, distance=5)
# limit to < 4 kHz for readability
mask = freqs[peaks] <= 4000.0
peaks = peaks[mask]

# assign harmonics near n*f0
harmonics = []
if note_info:
    for p in peaks:
        fpk = freqs[p]
        n_est = int(round(fpk / f0))
        if n_est>=1 and n_est<=20:
            harmonics.append((n_est, fpk, mag[p]))
    # keep first unique 10 harmonics by n
    # choose the strongest per n if duplicates
    byn = {}
    for n, fpk, amp in harmonics:
        if (n not in byn) or (amp > byn[n][1]):
            byn[n] = (fpk, amp)
    harmonics = sorted([(n,)+byn[n] for n in byn.keys()])[:10]
else:
    # fall back: top 10 peaks
    idx = np.argsort(mag[peaks])[::-1][:10]
    harmonics = [(-1, freqs[peaks[i]], mag[peaks[i]]) for i in idx]
    harmonics.sort(key=lambda t: t[1])

# ---------- timbre features ----------
# spectral
band = (freqs>=20.0) & (freqs<=4000.0)
cent = spectral_centroid(freqs[band], mag[band])
roll = spectral_rolloff(freqs[band], mag[band], pct=ROLL_PERCENT)

# odd/even energy ratio (requires f0)
odd_even_ratio = np.nan
inharm_rms_ppm = np.nan
if note_info and harmonics:
    amps = np.array([h[2] for h in harmonics])
    ns = np.array([h[0] for h in harmonics], dtype=float)
    odds = amps[(ns%2==1)]
    evens = amps[(ns%2==0)]
    if evens.size>0:
        odd_even_ratio = float(np.sum(odds)/np.sum(evens))
    # inharmonicity: deviation of f_n from n*f0 (ppm rms)
    f_meas = np.array([h[1] for h in harmonics])
    f_ideal = ns * f0
    inh_ppm = 1e6*(f_meas - f_ideal)/np.maximum(f_ideal,1e-9)
    inharm_rms_ppm = float(np.sqrt(np.mean(inh_ppm**2)))

# envelope + attack/decay
env = envelope_rms(seg, fs, smooth_sec=ENV_SMOOTH_SEC)
tseg = np.arange(len(seg))/fs
# attack 10%->90% time
pmax = np.argmax(env)
pre = env[:pmax+1]
if pre.size>5:
    lo = 0.1*np.max(pre); hi = 0.9*np.max(pre)
    try:
        t10 = tseg[np.where(pre>=lo)[0][0]]
        t90 = tseg[np.where(pre>=hi)[0][0]]
        attack_ms = 1000*(t90 - t10)
    except Exception:
        attack_ms = np.nan
else:
    attack_ms = np.nan
# decay: fit exp after peak for ~300 ms (log-linear)
post = env[pmax:pmax+int(0.35*fs)]
tpost = tseg[pmax:pmax+int(0.35*fs)]
decay_tau_ms = np.nan
if len(post)>10 and np.all(post>0):
    y = np.log(post/post[0] + 1e-12)
    A = np.vstack([tpost - tpost[0], np.ones_like(tpost)]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]  # y ~ k*t + b
    if k<0:
        tau = -1.0/k
        decay_tau_ms = 1000*tau

# ---------- print summary ----------
print("\n=== Analysis Summary ===")
print(f"File: {WAV_PATH}")
print(f"Sample rate: {fs} Hz, duration: {len(data)/fs:.2f} s")
print(f"Estimated f0: {f0:.2f} Hz" if np.isfinite(f0) else "Estimated f0: n/a")
if note_info:
    print(f"Nearest note: {note_info['name']}{note_info['octave']} "
          f"(ref {note_info['f_ref']:.2f} Hz), cents error: {note_info['cents']:+.1f}")
print(f"Spectral centroid: {cent:.1f} Hz, roll-off ({int(ROLL_PERCENT*100)}%): {roll:.1f} Hz")
print(f"Odd/Even harmonic energy ratio: {odd_even_ratio:.3f}")
print(f"Inharmonicity RMS: {inharm_rms_ppm:.1f} ppm")
print(f"Attack (10–90%): {attack_ms:.1f} ms, Decay tau: {decay_tau_ms:.1f} ms")

# harmonic table
if harmonics:
    print("\nHarmonics (n, freq Hz, rel dB):")
    h1_amp = harmonics[0][2] if harmonics[0][0]==1 else max([h[2] for h in harmonics])
    for n, fpk, amp in harmonics:
        rel_db = 20*np.log10(amp / h1_amp + 1e-12)
        label = f"n={n}" if n!=-1 else "peak"
        print(f"  {label:>5s}: {fpk:7.2f} Hz   {rel_db:6.1f} dB")

# ---------- plots (1 figure each) ----------
# 1) magnitude spectrum (0-4 kHz), annotate harmonics
plt.figure(figsize=(8,4))
fmask = (freqs>=0) & (freqs<=4000)
plt.plot(freqs[fmask], mag[fmask])
if note_info and harmonics:
    for n, fpk, amp in harmonics:
        if fpk<=4000:
            plt.axvline(fpk, linestyle=':', linewidth=1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum (Hann + zero-padding)")
plt.tight_layout()
plt.show()

# 2) envelope with attack/decay markers
plt.figure(figsize=(8,3.5))
plt.plot(tseg, env)
plt.xlabel("Time (s)")
plt.ylabel("RMS Envelope")
plt.title("Envelope (RMS, smoothed)")
plt.tight_layout()
plt.show()

# 3) harmonic bar chart (relative to H1)
if harmonics:
    plt.figure(figsize=(7,3.5))
    ns = [h[0] for h in harmonics]
    rel_db = [20*np.log10(h[2]/h1_amp + 1e-12) for h in harmonics]
    xs = np.arange(len(ns))
    plt.bar(xs, rel_db)
    plt.xticks(xs, [str(n) for n in ns])
    plt.xlabel("Harmonic number (n)")
    plt.ylabel("Level vs H1 (dB)")
    plt.title("Harmonic Levels (relative to fundamental)")
    plt.tight_layout()
    plt.show()

# ---------- save features to CSV ----------
import pandas as pd
rows = [{
    "file": WAV_PATH,
    "fs": fs,
    "duration_s": len(data)/fs,
    "f0_Hz": f0,
    "note": None if not note_info else f"{note_info['name']}{note_info['octave']}",
    "note_ref_Hz": None if not note_info else note_info['f_ref'],
    "cents": None if not note_info else note_info['cents'],
    "centroid_Hz": cent,
    "rolloff_Hz": roll,
    "odd_even_ratio": odd_even_ratio,
    "inharm_rms_ppm": inharm_rms_ppm,
    "attack_ms": attack_ms,
    "decay_tau_ms": decay_tau_ms
}]
# harmonic columns Hn_freq / Hn_rel_dB
if harmonics:
    for n, fpk, amp in harmonics:
        rows[0][f"H{n}_freq_Hz"] = fpk
        rows[0][f"H{n}_rel_dB"]  = 20*np.log10(amp/(h1_amp+1e-12)+1e-12)

df = pd.DataFrame(rows)
out_csv = "guitarron_analysis.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved features to: {os.path.abspath(out_csv)}")
