import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pydub import AudioSegment
import os

from src.dataset import SERDataset
from src.models import CRNN, SERTransformer
from src import config, features
from src.train import collate_fn


def compress_audio(input_path, output_path, bitrate="64k"):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="mp3", bitrate=bitrate)


def evaluate_model(model, loader, device):
    """
    Compute Unweighted Accuracy (UA) and Macro-F1 on the given DataLoader
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # Compute UA (per-class accuracy, averaged)
    per_class_acc = []
    for c in range(config.NUM_CLASSES):
        idx = [i for i, label in enumerate(all_labels) if label == c]
        if len(idx) == 0:
            continue
        correct = sum([all_preds[i] == all_labels[i] for i in idx])
        per_class_acc.append(correct / len(idx))
    UA = sum(per_class_acc) / len(per_class_acc)

    # Compute Macro-F1
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return UA, macro_f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = SERDataset()
    val_size = int(config.VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    _, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Select model
    if config.MODEL_TYPE == "crnn":
        model = CRNN()
        checkpoint = "ser_crnn_best.pth"
    else:
        model = SERTransformer(
            embed_dim=config.TRANSFORMER_EMBED_DIM,
            num_heads=config.TRANSFORMER_NUM_HEADS,
            ff_dim=config.TRANSFORMER_FF_DIM,
            num_layers=config.TRANSFORMER_NUM_LAYERS
        )
        checkpoint = "ser_transformer_best.pth"

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    UA, macro_f1 = evaluate_model(model, val_loader, device)
    print(f"Validation UA: {UA:.4f} | Macro-F1: {macro_f1:.4f}")

    # Optional: Evaluate compression
    for br in ["128k", "64k", "32k"]:
        correct = 0
        total = 0
        for idx in range(len(val_ds)):
            row = val_ds[idx][0]
            y = val_ds[idx][1].item()
            audio_path = dataset.df.iloc[val_ds.indices[idx]]["path"]
            compressed_path = audio_path.replace(".wav", f"_{br}.mp3")
            compress_audio(audio_path, compressed_path, bitrate=br)

            mel = features.extract_features_from_path(compressed_path)
            X = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = model(X)
                pred = torch.argmax(logits, dim=1).item()
                if pred == y:
                    correct += 1
            total += 1
            os.remove(compressed_path)
        print(f"Compression {br}: accuracy = {correct/total:.4f}")


if __name__ == "__main__":
    main()
