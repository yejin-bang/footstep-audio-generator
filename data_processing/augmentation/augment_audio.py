"""
Audio Augmentation for Training Data
Generates augmented versions of audio files with configurable intensity levels.
"""

import os
import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm
import torchaudio.transforms as T
import argparse


class AudioAugmenter:
    """
    Audio augmentation with configurable intensity.
    Creates multiple augmented versions of each input audio file.
    """
    
    def __init__(self, sample_rate=44100, intensity='subtle'):
        """
        Args:
            sample_rate: Target sample rate for audio processing
            intensity: Augmentation strength - 'subtle' or 'aggressive'
        """
        self.sample_rate = sample_rate
        self.intensity = intensity
        
        # Set augmentation parameters based on intensity
        if intensity == 'aggressive':
            self.tempo_factor = 1.06  # 6% variation
            self.pitch_semitones = 1.0
            self.eq_tilt_db = 2.2
            self.gain_db_v1 = 1.2
            self.gain_db_v2 = -1.0
            self.saturation_amount = 0.030
        elif intensity == 'subtle':
            self.tempo_factor = 1.02  # 2% variation
            self.pitch_semitones = 0.3
            self.eq_tilt_db = 1.0
            self.gain_db_v1 = 0.5
            self.gain_db_v2 = -0.5
            self.saturation_amount = 0.015
        else:
            raise ValueError(f"Unknown intensity: {intensity}. Use 'subtle' or 'aggressive'")
        
    def time_stretch(self, waveform, rate):
        """Apply time stretching to audio."""
        effects = [
            ["tempo", str(rate)],
            ["rate", str(self.sample_rate)]
        ]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        return augmented
    
    def pitch_shift(self, waveform, n_steps):
        """Apply pitch shifting to audio."""
        effects = [
            ["pitch", str(n_steps * 100)],
            ["rate", str(self.sample_rate)]
        ]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        return augmented
    
    def subtle_eq(self, waveform, tilt_db=1.0):
        """Apply equalization tilt for tonal variation."""
        effects = [
            ["equalizer", "100", "0.5q", str(-tilt_db * 0.3)],
            ["equalizer", "500", "0.5q", str(-tilt_db * 0.5)],
            ["equalizer", "2000", "0.5q", str(tilt_db * 0.3)],
            ["equalizer", "8000", "0.5q", str(tilt_db * 0.7)],
            ["rate", str(self.sample_rate)]
        ]
        augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects
        )
        return augmented
    
    def adjust_level(self, waveform, gain_db=0.5):
        """Apply gain adjustment."""
        gain_linear = 10 ** (gain_db / 20)
        return waveform * gain_linear
    
    def micro_saturation(self, waveform, amount=0.015):
        """Apply harmonic saturation for analog warmth."""
        saturated = torch.tanh(waveform * (1 + amount)) / (1 + amount * 0.5)
        return saturated
    
    def augment_v1(self, waveform):
        """
        Augmentation variant 1: Faster, brighter, louder.
        Simulates lighter/quicker characteristics.
        """
        aug = self.time_stretch(waveform, rate=self.tempo_factor)
        aug = self.pitch_shift(aug, n_steps=self.pitch_semitones)
        aug = self.subtle_eq(aug, tilt_db=self.eq_tilt_db)
        aug = self.adjust_level(aug, gain_db=self.gain_db_v1)
        return aug
    
    def augment_v2(self, waveform):
        """
        Augmentation variant 2: Slower, warmer, softer.
        Simulates heavier/deliberate characteristics.
        """
        aug = self.time_stretch(waveform, rate=2 - self.tempo_factor)
        aug = self.pitch_shift(aug, n_steps=-self.pitch_semitones)
        aug = self.subtle_eq(aug, tilt_db=-self.eq_tilt_db)
        aug = self.adjust_level(aug, gain_db=self.gain_db_v2)
        aug = self.micro_saturation(aug, amount=self.saturation_amount)
        return aug
    
    def ensure_length(self, waveform, target_length=265000):
        """Pad or trim audio to exact length."""
        if waveform.shape[1] < target_length:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        return waveform
    
    def process_folder(self, input_dir, output_dir, target_length=265000):
        """
        Process all audio files in input directory and create augmented versions.
        
        Args:
            input_dir: Directory containing source audio files
            output_dir: Directory where augmented files will be saved
            target_length: Target length in samples (default: 265000 for 6 seconds at 44.1kHz)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.aiff']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(Path(input_dir).glob(f'*{ext}'))
        
        audio_files = sorted(audio_files)
        
        if len(audio_files) == 0:
            print(f"❌ ERROR: No audio files found in {input_dir}")
            print(f"   Supported formats: {audio_extensions}")
            return
        
        print(f"🎵 Audio Augmentation Pipeline")
        print(f"=" * 60)
        print(f"Intensity:     {self.intensity}")
        print(f"Input folder:  {input_dir}")
        print(f"Output folder: {output_dir}")
        print(f"Found {len(audio_files)} audio files")
        print(f"Will create {len(audio_files) * 3} files (original + 2 augmented)")
        print(f"=" * 60)
        
        processed_count = 0
        
        for audio_path in tqdm(audio_files, desc="Processing"):
            try:
                # Load audio
                waveform, sr = torchaudio.load(str(audio_path))
                
                # Resample if needed
                if sr != self.sample_rate:
                    resampler = T.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                
                # Ensure stereo
                if waveform.shape[0] == 1:
                    waveform = waveform.repeat(2, 1)
                elif waveform.shape[0] > 2:
                    waveform = waveform[:2, :]
                
                # Ensure target length
                waveform = self.ensure_length(waveform, target_length=target_length)
                
                # Get base filename
                base_filename = audio_path.stem
                
                # Save original version
                orig_filename = f"{base_filename}_orig.wav"
                orig_path = os.path.join(output_dir, orig_filename)
                torchaudio.save(orig_path, waveform, self.sample_rate)
                
                # Save augmented version 1
                aug1 = self.augment_v1(waveform)
                aug1 = self.ensure_length(aug1, target_length=target_length)
                aug1_filename = f"{base_filename}_aug1.wav"
                aug1_path = os.path.join(output_dir, aug1_filename)
                torchaudio.save(aug1_path, aug1, self.sample_rate)
                
                # Save augmented version 2
                aug2 = self.augment_v2(waveform)
                aug2 = self.ensure_length(aug2, target_length=target_length)
                aug2_filename = f"{base_filename}_aug2.wav"
                aug2_path = os.path.join(output_dir, aug2_filename)
                torchaudio.save(aug2_path, aug2, self.sample_rate)
                
                processed_count += 1
                
            except Exception as e:
                print(f"\n⚠️  Error processing {audio_path.name}: {e}")
                continue
        
        print(f"\n✅ Processing complete!")
        print(f"  Input files:            {len(audio_files)}")
        print(f"  Successfully processed: {processed_count}")
        print(f"  Output files:           {processed_count * 3}")
        print(f"  Saved to:               {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Audio augmentation for training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggressive augmentation (larger variations)
  python augment_audio.py --input ./audio --output ./augmented --intensity aggressive
  
  # Subtle augmentation (smaller variations)
  python augment_audio.py --input ./audio --output ./augmented --intensity subtle
  
  # Custom sample rate
  python augment_audio.py --input ./audio --output ./aug --sample_rate 48000
  
  # Custom target length (in samples)
  python augment_audio.py --input ./audio --output ./aug --target_length 220500
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing audio files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for augmented files'
    )
    parser.add_argument(
        '--intensity',
        type=str,
        default='aggressive',
        choices=['subtle', 'aggressive'],
        help='Augmentation intensity level (default: aggressive)'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=44100,
        help='Target sample rate (default: 44100)'
    )
    parser.add_argument(
        '--target_length',
        type=int,
        default=265000,
        help='Target length in samples (default: 265000 = 6 seconds at 44.1kHz)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input):
        print(f"❌ ERROR: Input directory does not exist: {args.input}")
        return
    
    # Create augmenter and process
    augmenter = AudioAugmenter(
        sample_rate=args.sample_rate,
        intensity=args.intensity
    )
    augmenter.process_folder(
        args.input,
        args.output,
        target_length=args.target_length
    )
    
    print(f"\n{'=' * 60}")
    print("✅ Augmentation complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()