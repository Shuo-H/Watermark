import numpy as np
import soundfile as sf
import librosa
import torch
import wavmark
import os
import argparse
from tqdm import tqdm

def main(wav_path, watermark=None):

    output_dir = "wavmark_output"
    os.makedirs(output_dir, exist_ok=True)

    # Check if the wav_path is a directory or file path
    if os.path.isdir(wav_path):
        wav_paths = [os.path.join(wav_path, f) for f in os.listdir(wav_path) if f.endswith('.wav')]
    elif os.path.isfile(wav_path):
        wav_paths = [wav_path]
    else:
        raise ValueError("Invalid path provided. Must be a file or directory.")

    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = wavmark.load_model().to(device)

    # Create or parse 16-bit watermark
    if watermark is None:
        watermark = np.random.choice([0, 1], size=16)
    else:
        # Convert watermark string to a numpy array of binary values
        if len(watermark) != 16:
            raise ValueError("Watermark must be a binary string of exactly 16 bits.")
        watermark = np.array([int(bit) for bit in watermark], dtype=int)
    
    for wav_path in tqdm(wav_paths):
        # Read host audio
        signal, sample_rate = librosa.load(wav_path, sr=16000)

        file_name = os.path.basename(wav_path).replace(".wav", "")
        with torch.no_grad():
            # Encode watermark
            watermarked_signal, _ = wavmark.encode_watermark(model, signal, watermark, show_progress=False)
            # Decode watermark
            decoded_watermark, _ = wavmark.decode_watermark(model, watermarked_signal, show_progress=False)

        # Save watermarked audio
        output_file = f"{output_dir}/{file_name}_{''.join(map(str, watermark))}.wav"
        sf.write(output_file, watermarked_signal, 16000)

        # Calculate BER
        BER = (watermark != decoded_watermark).mean() * 100
        print(f"File: {file_name}")
        print(f"Given   Watermark is: {watermark}")
        print(f"Decoded Watermark is: {decoded_watermark}")
        print(f"BER: {BER:.2f}%")
        print(f"\n")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Watermark for single-channel 16kHz audio files")
    parser.add_argument("wav_path", type=str, help="Path to the wav file for watermarking")
    parser.add_argument("--watermark", type=str, default=None, help="Binary string of 16 bits to watermark the audio")
    args = parser.parse_args()

    # Run main with parsed arguments
    main(args.wav_path, args.watermark)
