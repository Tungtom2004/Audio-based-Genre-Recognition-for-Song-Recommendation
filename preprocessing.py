import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data/genres_original"
SAMPLE_RATE = 22050
DURATION = 30
NUM_SAMPLES = SAMPLE_RATE * DURATION

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎧 Using device: {device}")

# ===============================
# 🎛️ Khởi tạo các transform
# ===============================
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
).to(device)

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=40,
    melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64}
).to(device)


# ===============================
# ⚙️ HÀM TRÍCH XUẤT ĐẶC TRƯNG
# ===============================
def extract_features(file_path):
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.to(device)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    # Pad hoặc cắt
    if waveform.shape[1] > NUM_SAMPLES:
        waveform = waveform[:, :NUM_SAMPLES]
    elif waveform.shape[1] < NUM_SAMPLES:
        pad_len = NUM_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    # --- MFCC ---
    mfcc = mfcc_transform(waveform)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    mfcc_mean = mfcc.mean(dim=2).squeeze()

    # --- Mel Spectrogram ---
    mel = mel_spec(waveform)
    mel_db = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0)
    mel_mean = mel_db.mean(dim=2).squeeze()

    # --- Spectral Centroid ---
    window = torch.hann_window(1024).to(device)
    centroid = torchaudio.functional.spectral_centroid(
        waveform,
        SAMPLE_RATE,
        window=window,
        win_length=1024,
        n_fft=1024,
        hop_length=512,
        pad=0
    )
    centroid_mean = centroid.mean(dim=1)

    # --- Zero Crossing Rate ---
    sign_changes = torch.diff(torch.sign(waveform))
    zero_crossings = (sign_changes != 0).float().sum()
    zcr_mean = torch.tensor([zero_crossings / waveform.numel()])


    # --- Custom: Spectral Rolloff ---
    # rolloff_freq = tần số mà năng lượng tích lũy đạt 85% tổng
    spec = torch.abs(torch.fft.rfft(waveform))
    power = spec ** 2
    total_energy = torch.sum(power, dim=1, keepdim=True)
    cum_energy = torch.cumsum(power, dim=1)
    cutoff = 0.85 * total_energy
    rolloff_bin = (cum_energy < cutoff).sum(dim=1)
    rolloff_freq = (rolloff_bin.float() / spec.shape[1]) * (SAMPLE_RATE / 2)

    # Gộp tất cả đặc trưng lại
    features = torch.cat([
        mfcc_mean,
        mel_mean,
        centroid_mean,
        zcr_mean,
        rolloff_freq
    ]).cpu().numpy()

    return features


# ===============================
# 📦 XỬ LÝ TOÀN BỘ DỮ LIỆU
# ===============================
data = []
labels = []
genres = sorted(os.listdir(DATA_DIR))

for label, genre in enumerate(genres):
    genre_path = os.path.join(DATA_DIR, genre)
    if not os.path.isdir(genre_path):
        continue

    for file_name in tqdm(os.listdir(genre_path), desc=f"⏳ {genre}"):
        if file_name.endswith((".au", ".wav")):
            path = os.path.join(genre_path, file_name)
            feats = extract_features(path)
            data.append(feats)
            labels.append(label)

# ===============================
# 💾 LƯU RA CSV
# ===============================
df = pd.DataFrame(data)
df["label"] = labels
df.to_csv("music_features.csv", index=False)
print("💾 Saved torchaudio features → music_features.csv")
