import torch
from torch.utils.data import DataLoader
from pydub import AudioSegment
import os

from .dataset import SERDataset
from .models import CRNN
from . import config, features
from .train import collate_fn

def compress_audio(input_path, output_path, bitrate="64k"):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="mp3", bitrate=bitrate)

def evaluate_with_compression(model, dataset, bitrate="64k"):
    correct = 0
    total = 0

    for idx in range(len(dataset)):
        item = dataset.get_audio_metadata(idx)
        original_path = item["path"]

        compressed_path = original_path.replace(".wav", f"_{bitrate}.mp3")
        compress_audio(original_path, compressed_path, bitrate)

        # Extract features after compression
        mel = features.extract_features_from_path(compressed_path)
        X = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float()  # (1, 1, mel, T)
        y = dataset[idx][1]

        with torch.no_grad():
            logits = model(X)
            pred = torch.argmax(logits, dim=1).item()
            if pred == y.item():
                correct += 1

        total += 1
        os.remove(compressed_path)

    return correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = SERDataset()
    model = CRNN()
    model.load_state_dict(torch.load("ser_epoch20.pth", map_location=device))
    model.to(device)
    model.eval()

    for br in ["128k", "64k", "32k"]:
        acc = evaluate_with_compression(model, dataset, bitrate=br)
        print(f"Compression {br}: accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()
