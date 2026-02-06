import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class PeakWidthFootstepChopper:
    def __init__(self, 
                 target_chunk_size: float = 5.0, 
                 fade_duration: float = 0.05,
                 first_clip_buffer_samples: int = 6500):
        """
        Initialize peak width-based chopper
        
        Args:
            target_chunk_size: Target 5-second chunks
            fade_duration: Fade in/out duration in seconds
            first_clip_buffer_samples: Silent buffer before first footstep (default 6500 ≈ 147ms)
        """
        self.target_chunk_size = target_chunk_size
        self.fade_duration = fade_duration
        self.sample_rate = 44100
        self.first_clip_buffer_samples = first_clip_buffer_samples
        
        # Peak detection parameters
        self.min_peak_distance = 0.3
        self.peak_prominence_factor = 0.3
        self.peak_height_factor = 0.5
        self.peak_width_rel_height = 0.8  # CHANGED: 20% for wider boundaries
        
        # Quality control
        self.min_clip_length = 1.5
        self.min_quiet_region = 0.3
        self.long_silence_threshold = 2.0  # NEW: Trim silences longer than this
        
        print(f"Peak Width Footstep Chopper initialized")
        print(f"   Target chunk size: {target_chunk_size}s")
        print(f"   Peak width detection: {self.peak_width_rel_height*100:.0f}% of peak height")
        print(f"   First clip buffer: {first_clip_buffer_samples} samples ({first_clip_buffer_samples/self.sample_rate:.3f}s)")
        print(f"   Long silence threshold: {self.long_silence_threshold}s")
        print(f"   Sample rate: {self.sample_rate}Hz")
    
    def find_first_sound_adaptive(self, audio: np.ndarray, sr: int) -> float:
        """ Find first sound using adaptive threshold"""
        if len(audio) == 0:
            return 0.0
        
        # Calculate RMS
        frame_length = 1024
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        if len(rms) == 0:
            return 0.0
        
        # Adaptive threshold: use statistics of non-zero RMS values
        non_zero_rms = rms[rms > 0]
        if len(non_zero_rms) == 0:
            return 0.0
        
        # Use 20th percentile * 2 as threshold
        noise_floor = np.percentile(non_zero_rms, 20)
        adaptive_threshold = max(noise_floor * 2.0, 0.001)
        
        # Find first non-silent frame
        non_silent_frames = np.where(rms > adaptive_threshold)[0]
        
        if len(non_silent_frames) == 0:
            return 0.0
        
        first_sound_frame = non_silent_frames[0]
        first_sound_time = librosa.frames_to_time(first_sound_frame, sr=sr, hop_length=hop_length)
        
        return first_sound_time
    
    def analyze_peaks_and_widths(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Find peaks and their widths using scipy.signal.peak_widths
        
        Returns:
            Dictionary with peak information and quiet regions
        """
        # Calculate RMS envelope
        frame_length = 1024
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        if len(rms) < 10:
            return {
                'peaks': [],
                'peak_times': [],
                'peak_starts': [],
                'peak_ends': [],
                'quiet_regions': [(0, len(audio)/sr)],
                'rms': rms,
                'frame_times': frame_times
            }
        
        # Adaptive peak detection parameters
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        peak_height = rms_mean + rms_std * self.peak_height_factor
        peak_prominence = rms_std * self.peak_prominence_factor
        min_distance_frames = int(self.min_peak_distance * sr / hop_length)
        
        # Find peaks (footstep events)
        peaks, peak_properties = signal.find_peaks(
            rms,
            height=peak_height,
            prominence=peak_prominence,
            distance=min_distance_frames
        )
        
        if len(peaks) == 0:
            # No peaks found, entire audio is quiet
            return {
                'peaks': [],
                'peak_times': [],
                'peak_starts': [],
                'peak_ends': [],
                'quiet_regions': [(0, len(audio)/sr)],
                'rms': rms,
                'frame_times': frame_times
            }
        
        # Calculate peak widths (where each footstep starts and ends)
        try:
            widths, width_heights, left_ips, right_ips = signal.peak_widths(
                rms, 
                peaks, 
                rel_height=self.peak_width_rel_height
            )
            
            # Convert to time domain
            peak_times = frame_times[peaks]
            peak_start_times = []
            peak_end_times = []
            
            for i in range(len(peaks)):
                # Ensure indices are within bounds
                left_idx = max(0, min(len(frame_times)-1, int(left_ips[i])))
                right_idx = max(0, min(len(frame_times)-1, int(right_ips[i])))
                
                peak_start_times.append(frame_times[left_idx])
                peak_end_times.append(frame_times[right_idx])
            
            peak_start_times = np.array(peak_start_times)
            peak_end_times = np.array(peak_end_times)
            
        except Exception as e:
            print(f"      Warning: Peak width calculation failed: {e}")
            # Fallback: use peak locations with small window
            peak_times = frame_times[peaks]
            window = 0.1  # 100ms window around each peak
            peak_start_times = np.maximum(0, peak_times - window)
            peak_end_times = np.minimum(len(audio)/sr, peak_times + window)
        
        # Find quiet regions between peak widths
        quiet_regions = self._find_quiet_regions(peak_start_times, peak_end_times, peak_times, len(audio)/sr)
        
        return {
            'peaks': peaks,
            'peak_times': peak_times,
            'peak_starts': peak_start_times,
            'peak_ends': peak_end_times,
            'quiet_regions': quiet_regions,
            'rms': rms,
            'frame_times': frame_times,
            'analysis_params': {
                'peak_height': peak_height,
                'peak_prominence': peak_prominence,
                'rms_mean': rms_mean,
                'rms_std': rms_std
            }
        }
    def calculate_adaptive_quiet_threshold(self, peak_times):
        if len(peak_times) < 2:
            return 0.1
        
        intervals = np.diff(peak_times)
        avg_interval = np.mean(intervals)
        
        # Use 20% of average interval as minimum quiet region
        return max(0.05, avg_interval * 0.2)
    
    def _find_quiet_regions(self, peak_starts: np.ndarray, peak_ends: np.ndarray, 
                       peak_times: np.ndarray, audio_duration: float) -> List[Tuple[float, float]]:
        quiet_regions = []
        
        if len(peak_starts) == 0:
            return [(0, audio_duration)]
        
        # Calculate adaptive threshold ONCE at the start
        adaptive_threshold = self.calculate_adaptive_quiet_threshold(peak_times)
        
        # Sort peaks by start time
        sorted_indices = np.argsort(peak_starts)
        sorted_starts = peak_starts[sorted_indices]
        sorted_ends = peak_ends[sorted_indices]
        
        # Before first peak
        if sorted_starts[0] > adaptive_threshold:  # USE ADAPTIVE HERE
            quiet_regions.append((0, sorted_starts[0]))
        
        # Between peaks
        for i in range(len(sorted_starts) - 1):
            quiet_start = sorted_ends[i]
            quiet_end = sorted_starts[i + 1]
            
            if quiet_end - quiet_start >= adaptive_threshold:  # USE ADAPTIVE HERE
                quiet_regions.append((quiet_start, quiet_end))
        
        # After last peak - THIS IS THE KEY FIX
        if sorted_ends[-1] < audio_duration - adaptive_threshold:  # USE ADAPTIVE HERE
            quiet_regions.append((sorted_ends[-1], audio_duration))
        
        return quiet_regions
    
    def trim_long_silences(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """
        Trim long silences (>2s) down to average quiet region length
        
        Returns:
            (trimmed_audio, trimming_stats)
        """
        print(f"   Analyzing silences for trimming...")
        
        # Analyze full audio for peaks and quiet regions
        peak_analysis = self.analyze_peaks_and_widths(audio, sr)
        quiet_regions = peak_analysis['quiet_regions']
        
        if not quiet_regions:
            return audio, {'silences_trimmed': 0, 'time_saved': 0}
        
        # Calculate durations
        quiet_durations = [(end - start) for start, end in quiet_regions]
        
        # Separate normal and long silences
        normal_durations = [d for d in quiet_durations if d < self.long_silence_threshold]
        long_silences = [(start, end) for start, end in quiet_regions 
                        if (end - start) >= self.long_silence_threshold]
        
        if not long_silences:
            print(f"   No long silences found (all < {self.long_silence_threshold}s)")
            return audio, {'silences_trimmed': 0, 'time_saved': 0}
        
        # Calculate target duration for long silences
        if normal_durations:
            avg_quiet = np.mean(normal_durations)
        else:
            avg_quiet = 0.5  # Fallback if no normal quiet regions
        
        print(f"   Found {len(long_silences)} long silences (>{self.long_silence_threshold}s)")
        print(f"   Average normal quiet: {avg_quiet:.2f}s, will trim long silences to this")
        
        # Build new audio by keeping segments and trimming long silences
        segments = []
        current_pos = 0
        total_removed = 0
        
        for quiet_start, quiet_end in quiet_regions:
            duration = quiet_end - quiet_start
            
            if duration >= self.long_silence_threshold:
                # Keep audio up to start of silence
                start_sample = int(quiet_start * sr)
                segments.append(audio[current_pos:start_sample])
                
                # Add trimmed silence (avg_quiet duration)
                silence_samples = int(avg_quiet * sr)
                segments.append(np.zeros(silence_samples))
                
                # Skip the rest of the long silence
                end_sample = int(quiet_end * sr)
                removed = duration - avg_quiet
                total_removed += removed
                
                current_pos = end_sample
            else:
                # Normal duration - keep as is
                # This will be added in the next iteration or at the end
                pass
        
        # Add remaining audio
        segments.append(audio[current_pos:])
        
        # Concatenate all segments
        trimmed_audio = np.concatenate(segments)
        
        trimming_stats = {
            'silences_trimmed': len(long_silences),
            'time_saved': total_removed,
            'original_duration': len(audio) / sr,
            'trimmed_duration': len(trimmed_audio) / sr,
            'target_quiet_duration': avg_quiet
        }
        
        print(f"   Trimmed {len(long_silences)} long silences")
        print(f"   Time saved: {total_removed:.2f}s ({len(audio)/sr:.1f}s → {len(trimmed_audio)/sr:.1f}s)")
        
        return trimmed_audio, trimming_stats
    
    def is_cutting_through_footstep(self, target_time: float, peak_starts: np.ndarray, 
                                   peak_ends: np.ndarray, safety_margin: float = 0.1) -> Tuple[bool, str]:
        """
        Check if target cut time falls within any footstep peak width
        
        Returns:
            (is_cutting_through_peak, reason)
        """
        for i, (start, end) in enumerate(zip(peak_starts, peak_ends)):
            if start <= target_time <= end:
                return True, f"cutting_through_peak_{i}"
            
            if end <= target_time <= end + safety_margin:
                return True, f"too_close_after_peak_{i}"
        return False, "safe_quiet_region"
    
    def find_best_cut_point(self, target_time: float, quiet_regions: List[Tuple[float, float]], 
                           audio_duration: float) -> Tuple[float, str]:
        """Always cut at exact middle of quiet region"""
        # First, check if target time is already in a quiet region
        for quiet_start, quiet_end in quiet_regions:
            if quiet_start <= target_time <= quiet_end:
                return target_time, "target_in_quiet_region"
        
        # Find the nearest quiet region after target time
        future_quiet_regions = [(start, end) for start, end in quiet_regions if start > target_time]
        
        if future_quiet_regions:
            # Use the EXACT MIDDLE of the next quiet region
            next_quiet_start, next_quiet_end = future_quiet_regions[0]
            cut_point = (next_quiet_start + next_quiet_end) / 2 
            return cut_point, "extended_to_next_quiet_region"
        
        # No future quiet regions, use end of audio
        return audio_duration, "use_remaining_audio"
    
    def iterative_chop_with_peak_widths(self, audio: np.ndarray, sr: int, filename: str) -> List[Dict]:
        """First clip gets buffer, rest use middle cuts"""
        clips = []
        remaining_audio = audio.copy()
        global_offset = 0
        clip_index = 0
        
        print(f"   Starting peak width iterative processing...")
        
        while len(remaining_audio) > self.min_clip_length * sr:
            clip_index += 1
            
            # Step 1: Clean/trim leading silence (only for first clip)
            if clip_index == 1:
                first_sound_time = self.find_first_sound_adaptive(remaining_audio, sr)
                
                if first_sound_time > 0:
                    # First clip - keep 6500 samples before first sound
                    buffer_samples = self.first_clip_buffer_samples
                    first_sound_sample = int(first_sound_time * sr)
                    
                    if first_sound_sample > buffer_samples:
                        trim_samples = first_sound_sample - buffer_samples
                        remaining_audio = remaining_audio[trim_samples:]
                        global_offset += trim_samples / sr
                        print(f"      Clip {clip_index}: First clip - kept {buffer_samples} samples ({buffer_samples/sr:.3f}s) before first sound")
                    else:
                        print(f"      Clip {clip_index}: First clip - keeping all {first_sound_sample} samples before first sound")
            else:
                # Subsequent clips - NO trimming, keep natural spacing from middle-cut
                print(f"      Clip {clip_index}: Keeping natural spacing from previous cut")
            
            remaining_duration = len(remaining_audio) / sr
            if remaining_duration < self.min_clip_length:
                print(f"      Clip {clip_index}: Remaining audio too short ({remaining_duration:.2f}s)")
                break
            
            # Step 2: Analyze peaks and widths in remaining audio
            peak_analysis = self.analyze_peaks_and_widths(remaining_audio, sr)
            
            print(f"      Clip {clip_index}: Found {len(peak_analysis['peaks'])} peaks, "
                  f"{len(peak_analysis['quiet_regions'])} quiet regions")
            
            # Step 3: Check if 5-second mark cuts through a footstep
            target_cut_time = min(self.target_chunk_size, remaining_duration)
            
            is_cutting_peak, reason = self.is_cutting_through_footstep(
                target_cut_time, 
                peak_analysis['peak_starts'], 
                peak_analysis['peak_ends']
            )
            
            if is_cutting_peak:
                # Find next safe cut point (middle of quiet region)
                actual_cut_time, cut_reason = self.find_best_cut_point(
                    target_cut_time,
                    peak_analysis['quiet_regions'],
                    remaining_duration
                )
                print(f"      Clip {clip_index}: 5s cuts through footstep, extending to {actual_cut_time:.2f}s")
            else:
                actual_cut_time = target_cut_time
                cut_reason = "safe_cut_at_target"
                print(f"      Clip {clip_index}: Safe to cut at {actual_cut_time:.2f}s")
            
            # Extract the clip
            cut_sample = int(actual_cut_time * sr)
            clip_audio = remaining_audio[:cut_sample]
            
            # Create clip metadata
            clip_info = {
                'clip_index': clip_index,
                'start_time_in_original': global_offset,
                'end_time_in_original': global_offset + actual_cut_time,
                'duration': actual_cut_time,
                'audio_data': clip_audio,
                'source_filename': filename,
                'is_final_clip': actual_cut_time == remaining_duration,
                'method': 'peak_width_analysis',
                'cut_reason': cut_reason,
                'peaks_in_clip': len(peak_analysis['peaks']),
                'quiet_regions_in_clip': len(peak_analysis['quiet_regions']),
                'target_was_safe': not is_cutting_peak
            }
            
            clips.append(clip_info)
            
            # Move to next iteration
            remaining_audio = remaining_audio[cut_sample:]
            global_offset += actual_cut_time
        
        print(f"   Peak width analysis complete: {len(clips)} clips created")
        return clips
    
    def ensure_stereo(self, audio: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Convert mono to dual-mono stereo and ensure correct shape for soundfile
        
        librosa loads as: (samples,) for mono, (channels, samples) for stereo
        soundfile expects: (samples, channels) for writing
        """
        if audio.ndim == 1:
            # Mono - duplicate to stereo and transpose
            stereo_audio = np.stack([audio, audio], axis=0)  # Shape: (2, samples)
            stereo_audio = stereo_audio.T  # Transpose to (samples, 2)
            return stereo_audio, 'mono_converted_to_stereo'
        
        elif audio.ndim == 2 and audio.shape[0] == 1:
            # Single channel with extra dimension - expand and transpose
            stereo_audio = np.repeat(audio, 2, axis=0)  # Shape: (2, samples)
            stereo_audio = stereo_audio.T  # Transpose to (samples, 2)
            return stereo_audio, 'mono_converted_to_stereo'
        
        elif audio.ndim == 2 and audio.shape[0] == 2:
            # Already stereo in librosa format (2, samples) - just transpose
            stereo_audio = audio.T  # Transpose to (samples, 2)
            return stereo_audio, 'already_stereo_transposed'
        
        elif audio.ndim == 2 and audio.shape[1] == 2:
            # Already in soundfile format (samples, 2) - no change needed
            return audio, 'already_stereo_correct_shape'
        
        else:
            # Unexpected shape - raise error with helpful message
            raise ValueError(
                f"Unexpected audio shape: {audio.shape}. "
                f"Expected (samples,), (2, samples), or (samples, 2)"
            )
        
    def normalize_lufs(self, audio: np.ndarray, sr: int, target_lufs: float = -16.0) -> Tuple[np.ndarray, Dict]:
        """Normalize audio to target LUFS using pyloudnorm"""
        try:
            import pyloudnorm as pyln
            
            # Measure loudness
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            
            # Skip if already close to target (within 1 LU)
            if abs(loudness - target_lufs) < 1.0:
                print(f"      Already at target LUFS ({loudness:.1f}), skipping normalization")
                return audio, {'normalized': False, 'original_lufs': loudness, 'target_lufs': target_lufs}
            
            # Normalize to target
            normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
            
            # Prevent clipping
            peak = np.max(np.abs(normalized_audio))
            if peak > 0.99:
                normalized_audio = normalized_audio * (0.99 / peak)
            
            return normalized_audio, {
                'normalized': True,
                'original_lufs': loudness,
                'target_lufs': target_lufs,
                'applied_gain_db': target_lufs - loudness
            }
            
        except ImportError:
            print(f"      Warning: pyloudnorm not installed, using -1dB peak normalization")
            return self.normalize_to_minus_1db(audio), {'normalized': False, 'method': 'peak'}
                
    
    def apply_fades(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply fade in and fade out to prevent clicks
        Works with both mono (samples,) and stereo (samples, 2)
        """
        fade_samples = int(self.fade_duration * sr)
        
        # Handle edge case where fade is too long
        if fade_samples >= len(audio) // 2:
            return audio
        
        # Create fade curves
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio_faded = audio.copy()
        
        # Handle stereo vs mono
        if audio.ndim == 2:  # Stereo: (samples, 2)
            # Apply fade to both channels by reshaping fade curve
            # fade_in becomes (fade_samples, 1) so it broadcasts across channels
            audio_faded[:fade_samples] *= fade_in[:, np.newaxis]
            audio_faded[-fade_samples:] *= fade_out[:, np.newaxis]
        else:  # Mono: (samples,)
            audio_faded[:fade_samples] *= fade_in
            audio_faded[-fade_samples:] *= fade_out
        
        return audio_faded
    
    def visualize_peak_analysis(self, audio: np.ndarray, sr: int, peak_analysis: Dict, 
                           clip_info: Dict, output_dir: str):
        """Create visualization of peak width analysis for debugging"""

        source_name = Path(clip_info['source_filename']).stem
        source_name_clean = source_name.replace(' ', '_').replace(',', '').replace('/', '_')
        viz_filename = f"debug_{source_name_clean}_clip_{clip_info['clip_index']:03d}.png"
        
        try: 
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot waveform
            time_axis = np.linspace(0, len(audio)/sr, len(audio))
            ax1.plot(time_axis, audio, alpha=0.6, color='blue', linewidth=0.5)
            ax1.set_title(f"Audio Waveform - {clip_info['source_filename']}")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True, alpha=0.3)
            
            # Plot RMS with peak analysis
            ax2.plot(peak_analysis['frame_times'], peak_analysis['rms'], 
                    color='red', linewidth=2, label='RMS Envelope')
            
            # Mark peaks
            if len(peak_analysis['peaks']) > 0:
                peak_times = peak_analysis['peak_times']
                peak_rms_values = peak_analysis['rms'][peak_analysis['peaks']]
                ax2.scatter(peak_times, peak_rms_values, color='red', s=100, 
                        marker='o', label='Detected Peaks', zorder=5)
                
                # Mark peak widths
                for i, (start, end) in enumerate(zip(peak_analysis['peak_starts'], 
                                                peak_analysis['peak_ends'])):
                    ax2.axvspan(start, end, alpha=0.3, color='orange', 
                            label='Peak Width (20%)' if i == 0 else '')
            
            # Mark quiet regions
            for i, (start, end) in enumerate(peak_analysis['quiet_regions']):
                ax2.axvspan(start, end, alpha=0.2, color='green', 
                        label='Quiet Region' if i == 0 else '')
            
            # Mark cut points
            ax2.axvline(x=self.target_chunk_size, color='blue', linestyle='--', 
                    label='Target Cut (5s)')
            ax2.axvline(x=clip_info['duration'], color='purple', linestyle='-', 
                    linewidth=3, label='Actual Cut')
            
            ax2.set_title("Peak Width Analysis (20% height)")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("RMS Energy")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = output_path / viz_filename
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"      Debug visualization saved: {viz_path.name}")
            
        except Exception as e:
            print(f"      Warning: Could not create visualization for {viz_filename}: {e}")

    
    def process_single_file(self, input_path: str, surface: str, footwear: str, 
                           visualize: bool = False, output_dir: str = None) -> List[Dict]:
        """Process a single audio file using peak width analysis"""

        input_path = Path(input_path)
        source_name = input_path.stem
        
        print(f"Processing: {input_path.name}")
        
        try:
            # Load audio
            audio, sr = librosa.load(str(input_path), sr=self.sample_rate, mono=True)
            original_duration = len(audio) / sr
            
            print(f"   Original: {original_duration:.1f}s, {len(audio):,} samples")

            # Trim long silences first
            audio, trimming_stats = self.trim_long_silences(audio, sr)
            
            # Step 1: Iterative chopping with peak width analysis
            clip_infos = self.iterative_chop_with_peak_widths(audio, sr, input_path.name)
            
            if not clip_infos:
                print(f"   No clips created from {input_path.name}")
                return []
            
            # Step 2: Process each clip (normalize + fades)
            processed_clips = []
            
            for clip_info in clip_infos:
                stereo_audio, channel_status = self.ensure_stereo(clip_info['audio_data'])
                
                # Normalize LUFS
                normalized_audio, lufs_stats = self.normalize_lufs(stereo_audio, sr, target_lufs=-16.0)
                
                # Apply fades
                final_audio = self.apply_fades(normalized_audio, sr)
                
                # Create visualization if requested
                if visualize and output_dir:
                    clip_peak_analysis = self.analyze_peaks_and_widths(clip_info['audio_data'], sr)
                    self.visualize_peak_analysis(clip_info['audio_data'], sr, 
                                               clip_peak_analysis, clip_info, output_dir)
                
                # Create final metadata
                processed_clip = {
                    'clip_filename': f"{surface}_{footwear}_{source_name}_seg{clip_info['clip_index']:02d}.wav",
                    'source_file': str(input_path.absolute()),
                    'surface': surface,
                    'footwear': footwear,
                    'duration': clip_info['duration'],
                    'sample_rate': sr,
                    'lufs_stats': lufs_stats,
                    'start_time_in_source': clip_info['start_time_in_original'],
                    'end_time_in_source': clip_info['end_time_in_original'],
                    'clip_index': clip_info['clip_index'],
                    'is_final_clip': clip_info['is_final_clip'],
                    'segmentation_method': clip_info['method'],
                    'cut_reason': clip_info['cut_reason'],
                    'peaks_detected': clip_info['peaks_in_clip'],
                    'quiet_regions_found': clip_info['quiet_regions_in_clip'],
                    'target_was_safe': clip_info['target_was_safe'],
                    'peak_level_db': 20 * np.log10(np.max(np.abs(final_audio))) if np.max(np.abs(final_audio)) > 0 else -np.inf,
                    'audio_data': final_audio,
                    'processing_timestamp': datetime.now().isoformat(),
                    'trimming_stats': trimming_stats  # NEW: Include trimming info
                }
                
                processed_clips.append(processed_clip)
            
            print(f"   Processed {len(processed_clips)} clips with peak width analysis")
            return processed_clips
            
        except Exception as e:
            print(f"   Error processing {input_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def export_clips_structured(self, all_clips_metadata: List[Dict], output_base_dir: str) -> Dict:
        """Export all processed clips to structured directory with peak analysis summary"""

        output_base_dir = Path(output_base_dir)
        
        print("\n" + "=" * 60)
        print("EXPORTING PEAK WIDTH PROCESSED CLIPS")
        print("=" * 60)
        
        export_summary = {
            'export_timestamp': datetime.now().isoformat(),
            'total_clips': len(all_clips_metadata),
            'output_directory': str(output_base_dir.absolute()),
            'surface_breakdown': {},
            'peak_analysis_summary': {
                'clips_with_peaks': 0,
                'clips_where_target_was_safe': 0,
                'clips_extended_for_safety': 0,
                'average_peaks_per_clip': 0
            },
            'exported_files': []
        }
        
        # Create structured directories and export clips
        total_peaks = 0
        
        for clip_metadata in all_clips_metadata:
            surface = clip_metadata['surface']
            footwear = clip_metadata['footwear']
            
            # Create directory structure
            clip_dir = output_base_dir / "audio_dataset_cleaned" / surface / footwear
            clip_dir.mkdir(parents=True, exist_ok=True)
            
            # Export audio file
            clip_filename = clip_metadata['clip_filename']
            clip_path = clip_dir / clip_filename
            
            sf.write(str(clip_path), clip_metadata['audio_data'], clip_metadata['sample_rate'])
            
            # Update metadata
            clip_metadata['final_path'] = str(clip_path.absolute())
            
            # Track statistics
            surface_key = f"{surface}_{footwear}"
            if surface_key not in export_summary['surface_breakdown']:
                export_summary['surface_breakdown'][surface_key] = 0
            export_summary['surface_breakdown'][surface_key] += 1
            
            # Peak analysis statistics
            if clip_metadata['peaks_detected'] > 0:
                export_summary['peak_analysis_summary']['clips_with_peaks'] += 1
                total_peaks += clip_metadata['peaks_detected']
            
            if clip_metadata['target_was_safe']:
                export_summary['peak_analysis_summary']['clips_where_target_was_safe'] += 1
            else:
                export_summary['peak_analysis_summary']['clips_extended_for_safety'] += 1
            
            export_summary['exported_files'].append({
                'filename': clip_filename,
                'path': str(clip_path.absolute()),
                'surface': surface,
                'footwear': footwear,
                'duration': clip_metadata['duration'],
                'peaks_detected': clip_metadata['peaks_detected'],
                'cut_reason': clip_metadata['cut_reason']
            })
        
        # Calculate average peaks per clip
        if len(all_clips_metadata) > 0:
            export_summary['peak_analysis_summary']['average_peaks_per_clip'] = total_peaks / len(all_clips_metadata)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Save complete metadata
        metadata_file = output_base_dir / "audio_segments_metadata.json"
        complete_metadata = {
            'export_summary': export_summary,
            'all_clips_metadata': [{k: v for k, v in clip.items() if k != 'audio_data'} for clip in all_clips_metadata],
            'processing_config': {
                'target_chunk_size': self.target_chunk_size,
                'fade_duration': self.fade_duration,
                'sample_rate': self.sample_rate,
                'min_peak_distance': self.min_peak_distance,
                'peak_prominence_factor': self.peak_prominence_factor,
                'peak_height_factor': self.peak_height_factor,
                'peak_width_rel_height': self.peak_width_rel_height,
                'first_clip_buffer_samples': self.first_clip_buffer_samples,
                'long_silence_threshold': self.long_silence_threshold
            }
        }
        
        # Convert all numpy types before saving
        complete_metadata = convert_numpy_types(complete_metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(complete_metadata, f, indent=2)
        
        print(f"Clips exported by surface/footwear:")
        for surface_footwear, count in export_summary['surface_breakdown'].items():
            print(f"   {surface_footwear}: {count} clips")
        
        print(f"\nPeak Analysis Summary:")
        summary = export_summary['peak_analysis_summary']
        print(f"   Clips with detected peaks: {summary['clips_with_peaks']}")
        print(f"   Target time was safe: {summary['clips_where_target_was_safe']}")
        print(f"   Extended for safety: {summary['clips_extended_for_safety']}")
        print(f"   Average peaks per clip: {summary['average_peaks_per_clip']:.1f}")
        
        print(f"\nComplete metadata: {metadata_file}")
        print(f"Total clips exported: {export_summary['total_clips']}")
        
        return export_summary
    
    def batch_process_inventory(self, inventory_path: str, output_base_dir: str, 
                               visualize_samples: int = 0) -> Dict:
        """Process entire asset inventory using peak width analysis"""
        print("=" * 60)
        print("PEAK WIDTH FOOTSTEP CHOPPING PIPELINE")
        print("=" * 60)
        
        # Load inventory
        with open(inventory_path, 'r') as f:
            inventory_data = json.load(f)
        
        all_clips_metadata = []
        processing_summary = {
            'start_time': datetime.now().isoformat(),
            'total_source_files': 0,
            'total_clips_created': 0,
            'failed_files': []
        }
        
        visualized_count = 0
        
        # Process all files
        for surface, footwear_dict in inventory_data['inventory'].items():
            print(f"\nProcessing surface: {surface}")
            
            for footwear, files in footwear_dict.items():
                if not files:
                    continue
                    
                print(f"   Processing {footwear}: {len(files)} files")
                
                for file_info in files:
                    try:
                        visualize_this = visualized_count < visualize_samples
                        
                        clip_metadata = self.process_single_file(
                            file_info['file_path'],
                            surface,
                            footwear,
                            visualize=visualize_this,
                            output_dir=output_base_dir if visualize_this else None
                        )
                        
                        all_clips_metadata.extend(clip_metadata)
                        processing_summary['total_source_files'] += 1
                        
                        if visualize_this:
                            visualized_count += 1
                        
                    except Exception as e:
                        error_info = {
                            'file': file_info['file_path'],
                            'error': str(e)
                        }
                        processing_summary['failed_files'].append(error_info)
                        print(f"   Failed: {file_info.get('filename', 'unknown')} - {e}")
        
        processing_summary['total_clips_created'] = len(all_clips_metadata)
        processing_summary['end_time'] = datetime.now().isoformat()
        
        # Export all clips
        export_summary = self.export_clips_structured(all_clips_metadata, output_base_dir)
        
        # Final summary
        print("\n" + "=" * 60)
        print("PEAK WIDTH CHOPPING COMPLETE!")
        print("=" * 60)
        print(f"Source files processed: {processing_summary['total_source_files']}")
        print(f"Total clips created: {processing_summary['total_clips_created']}")
        print(f"Clips exported to: {output_base_dir}/clips/")
        
        if processing_summary['failed_files']:
            print(f"Failed files: {len(processing_summary['failed_files'])}")
        
        return {
            'processing_summary': processing_summary,
            'export_summary': export_summary,
            'total_clips': len(all_clips_metadata)
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Audio preprocessing pipeline with peak width analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process inventory with default settings
  python audio_preprocessing.py --inventory data/asset_inventory.json --output data/preprocessed
  
  # With debug visualizations for first 5 files
  python audio_preprocessing.py --inventory data/inventory.json --output data/out --visualize 5
  
  # Custom chunk size and fade duration
  python audio_preprocessing.py --inventory data/inventory.json --output data/out --chunk_size 6.0 --fade 0.1
        """
    )
    
    parser.add_argument(
        '--inventory',
        type=str,
        required=True,
        help='Path to asset inventory JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed clips'
    )
    parser.add_argument(
        '--chunk_size',
        type=float,
        default=5.0,
        help='Target chunk size in seconds (default: 5.0)'
    )
    parser.add_argument(
        '--fade',
        type=float,
        default=0.05,
        help='Fade in/out duration in seconds (default: 0.05)'
    )
    parser.add_argument(
        '--buffer_samples',
        type=int,
        default=6500,
        help='Silent buffer before first footstep in samples (default: 6500)'
    )
    parser.add_argument(
        '--visualize',
        type=int,
        default=0,
        help='Number of files to create debug visualizations for (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Initialize chopper with command-line arguments
    chopper = PeakWidthFootstepChopper(
        target_chunk_size=args.chunk_size,
        fade_duration=args.fade,
        first_clip_buffer_samples=args.buffer_samples
    )
    
    # Validate paths
    if not Path(args.inventory).exists():
        print(f"❌ ERROR: Inventory file not found: {args.inventory}")
        print("   Run the asset scanner first to create the inventory!")
        exit(1)
    
    try:
        results = chopper.batch_process_inventory(
            inventory_path=args.inventory,
            output_base_dir=args.output,
            visualize_samples=args.visualize
        )
        
        print(f"\n✅ Peak Width Processing Complete!")
        print(f"   Training clips created: {results['total_clips']}")
        print(f"   Directory structure: {args.output}/audio_dataset_cleaned/surface/footwear/")
        print(f"   Peak analysis metadata: {args.output}/peak_width_metadata.json")
        
        # Show detailed results
        if 'export_summary' in results:
            summary = results['export_summary']['peak_analysis_summary']
            success_rate = (summary['clips_where_target_was_safe'] / results['total_clips'] * 100) if results['total_clips'] > 0 else 0
            
            print(f"\nPeak Width Analysis Results:")
            print(f"   Target cut was safe: {success_rate:.1f}%")
            print(f"   Clips extended for safety: {summary['clips_extended_for_safety']}")
            print(f"   Average footsteps per clip: {summary['average_peaks_per_clip']:.1f}")
            print(f"   Clips with detected peaks: {summary['clips_with_peaks']}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)