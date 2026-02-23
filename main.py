import argparse
from src.train import train_model
from src.predict import predict_folder

def main():
    parser = argparse.ArgumentParser(
        description="MAEC multimodal model (audio + text)"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Run mode: train or predict"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data/MAEC_Dataset",
        help="Path to MAEC dataset"
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Folder path for prediction"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.data_root)

    elif args.mode == "predict":
        if not args.folder:
            raise ValueError("You must provide --folder for prediction mode")
        pred, conf = predict_folder(args.folder)
        print(f"Prediction: {pred} | Confidence: {conf:.3f}")

if __name__ == "__main__":
    main()
