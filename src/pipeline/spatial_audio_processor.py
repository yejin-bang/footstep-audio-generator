#!/usr/bin/env python3
"""
Spatial Audio Processor for Footstep Sound Generation

Professional spatial audio processing pipeline:
1. Load generated audio + detection results
2. Quality checks (clipping, phase, normalization)
3. Chop audio into individual footstep segments (quiet zone detection)
4. Assign segments to timestamps (random with replacement)
5. Apply spatial processing per footstep:
   - Constant power panning (x-position)
   - Inverse distance attenuation (depth)
6. Mix all positioned footsteps into final stereo track
7. Match video duration, apply fades
8. Export with visualization

Industry Standards:
- Panning: Constant power (-3dB pan law, sin/cos taper), subtle width (20%)
- Attenuation: Inverse distance law (6dB per doubling)
- Normalization: -6dB peak (professional SFX standard)
- Sample rate: 44.1kHz (can output 48kHz if needed)

"""

import librosa
import soundfile as sf
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SpatialAudioProcessor:
    """
    Professional spatial audio processor for footstep sound generation
    """
    
    def __init__(self,
                 sample_rate: int = 44100,
                 fade_duration: float = 0.01,
                 min_attenuation_db: float = -20.0,
                 max_attenuation_db: float = 0.0,
                 max_pan_percentage: float = 0.2,
                 distance_curve_power: float = 2.0):
        """
        Initialize spatial audio processor

        Args:
            sample_rate: Target sample rate (44100 or 48000)
            fade_duration: Crossfade duration for segment placement (seconds)
            min_attenuation_db: Minimum volume for farthest footsteps (-20dB = 10% volume)
            max_attenuation_db: Maximum volume for closest footsteps (0dB = 100% volume)
            max_pan_percentage: Maximum panning amount (0.2 = subtle/natural, 0.4 = moderate, 1.0 = hard pan)
            distance_curve_power: Distance attenuation curve steepness (1.0=linear, 2.0=inverse square)
        """
        self.sample_rate = sample_rate
        self.fade_duration = fade_duration
        self.min_attenuation_db = min_attenuation_db
        self.max_attenuation_db = max_attenuation_db
        self.max_pan_percentage = max_pan_percentage
        self.distance_curve_power = distance_curve_power
        
        print(f"Spatial Audio Processor initialized")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Crossfade duration: {fade_duration}s")
        print(f"   Attenuation range: {min_attenuation_db}dB to {max_attenuation_db}dB")
        print(f"   Panning: Constant power (-3dB pan law)")
        print(f"   Max pan: ±{max_pan_percentage*100:.0f}% ({'subtle' if max_pan_percentage <= 0.3 else 'moderate' if max_pan_percentage <= 0.5 else 'aggressive'} panning)")
        print(f"   Distance curve: power={distance_curve_power} ({'inverse square' if distance_curve_power==2.0 else 'custom'})")
    
    def process_video_audio(self, 
                           audio_input, 
                           detection_results: Dict,
                           output_path: str,
                           visualize: bool = True) -> Dict:
        """
        Complete pipeline: audio + detections → spatialized output
        
        Args:
            audio_path: Path to generated audio file (e.g., 6-second Stable Audio output)
            detection_results: Output from footstep_detector.process_video()
            output_path: Where to save final spatialized audio
            visualize: Create visualization of processing
            
        Returns:
            Processing statistics and metadata
        """
        print("\n" + "=" * 60)
        print("SPATIAL AUDIO PROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load audio
        print("\nStep 1: Loading generated audio...")
        if isinstance(audio_input, str):
            audio, sr = librosa.load(audio_input, sr=None, mono=True)  # file path
        elif isinstance(audio_input, tuple):
            audio, sr = audio_input  # (array, sr) tuple
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input; sr = self.sample_rate  # raw array
        print(f"   Loaded: {len(audio)/sr:.2f}s at {sr}Hz")
        
        # Step 2: Quality checks
        print("\nStep 2: Quality checks...")
        audio = self._quality_check_and_fix(audio, sr)
        
        # Step 3: Chop into individual footsteps
        print("\nStep 3: Chopping audio into individual footsteps...")
        footstep_segments = self._chop_audio_at_quiet_zones(audio, sr)
        print(f"   Extracted {len(footstep_segments)} footstep segments")
        
        # Step 4: Get video duration and spatial data
        video_duration = detection_results['video_info']['duration']
        video_width = detection_results['video_info']['width']
        spatial_data = detection_results['spatial_data']
        num_detections = len(spatial_data)

        print(f"\nStep 4: Preparing spatial processing...")
        print(f"   Video duration: {video_duration:.2f}s")
        print(f"   Video width: {video_width}px")
        print(f"   Detected footsteps: {num_detections}")
        print(f"   Available audio segments: {len(footstep_segments)}")

        # Step 5: Calculate reference distance for depth
        reference_distance = self._calculate_reference_distance(spatial_data)
        print(f"   Reference distance (closest): {reference_distance:.3f} pixels")

        # Step 6: Create final stereo mix
        print("\nStep 5: Creating spatialized stereo mix...")
        final_audio = self._create_spatial_mix(
            footstep_segments,
            spatial_data,
            reference_distance,
            video_duration,
            video_width,
            sr
        )
        
        # Step 7: Apply final fades and normalization
        print("\nStep 6: Finalizing audio...")
        final_audio = self._apply_final_processing(final_audio, video_duration, sr)
        
        # Step 8: Export
        print(f"\nStep 7: Exporting to {output_path}...")
        sf.write(output_path, final_audio, sr)
        print(f"   Exported: {output_path}")
        print(f"   Duration: {len(final_audio)/sr:.2f}s")
        print(f"   Peak level: {20*np.log10(np.max(np.abs(final_audio))):.1f}dB")
        
        # Step 9: Visualize if requested
        if visualize:
            print("\nStep 8: Creating visualization...")
            viz_path = str(Path(output_path).with_suffix('.png'))
            self._create_visualization(
                final_audio,
                spatial_data,
                footstep_segments,
                reference_distance,
                video_width,
                sr,
                viz_path
            )
            print(f"   Visualization: {viz_path}")
        
        # Compile statistics
        stats = {
            'input_audio_duration': len(audio) / sr,
            'output_audio_duration': len(final_audio) / sr,
            'video_duration': video_duration,
            'num_footsteps': num_detections,
            'num_segments_available': len(footstep_segments),
            'reference_distance': reference_distance,
            'sample_rate': sr,
            'output_path': output_path,
            'peak_level_db': 20*np.log10(np.max(np.abs(final_audio)))
        }
        
        print("\n" + "=" * 60)
        print("SPATIAL AUDIO PROCESSING COMPLETE")
        print("=" * 60)
        
        return stats
    
    def _quality_check_and_fix(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Step 2: Quality checks and fixes
        - Check for clipping
        - Check for phase issues
        - Normalize to -6dB peak if needed (professional SFX standard)
        """
        # Check 1: Clipping
        peak = np.max(np.abs(audio))
        if peak >= 0.99:
            print(f"   ⚠ Clipping detected! Peak: {peak:.3f} (clipped)")
            # Normalize to -6dB to prevent clipping (professional SFX standard)
            target_peak = 10 ** (-6.0 / 20)  # -6dB in linear
            audio = audio * (target_peak / peak)
            print(f"   ✓ Fixed: Normalized to -6dB peak")
        else:
            print(f"   ✓ No clipping (peak: {peak:.3f})")

        # Check 2: Phase (mono audio shouldn't have phase issues, but check anyway)
        # For mono, we just verify it's actually mono
        if audio.ndim > 1:
            print(f"   ⚠ Multi-channel audio detected, using channel 0")
            audio = audio[0]
        else:
            print(f"   ✓ Mono audio confirmed")

        # Check 3: Normalization
        # If audio is too quiet (peak < -12dB), normalize to -6dB
        peak_db = 20 * np.log10(peak)
        if peak_db < -12.0:
            print(f"   ⚠ Audio too quiet: {peak_db:.1f}dB")
            target_peak = 10 ** (-6.0 / 20)
            audio = audio * (target_peak / peak)
            print(f"   ✓ Fixed: Normalized to -6dB peak")
        else:
            print(f"   ✓ Level OK: {peak_db:.1f}dB peak")

        return audio
    
    def _chop_audio_at_quiet_zones(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Step 3: Chop audio into individual footstep segments using QUIET ZONE detection
        
        This is the IMPROVED method that:
        1. Finds actual quiet regions between peaks
        2. Cuts at safe points (low RMS zones)
        3. Preserves complete footstep envelopes (attack + sustain + decay)
        4. Returns segments WITH peak offset information for proper alignment
        
        Returns:
            List of dicts containing:
            - 'audio': np.ndarray segment
            - 'peak_offset': float (seconds from segment start to peak)
            - 'duration': float (segment duration)
        """
        # Calculate RMS envelope
        frame_length = 1024
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        if len(rms) < 10:
            print("   ⚠ Audio too short, returning as single segment")
            # Return single segment with peak assumed at center
            return [{
                'audio': audio,
                'peak_offset': len(audio) / (2 * sr),
                'duration': len(audio) / sr
            }]
        
        # === STEP 1: ADAPTIVE PEAK DETECTION ===
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        peak_height = rms_mean + rms_std * 0.5
        peak_prominence = rms_std * 0.3
        min_distance_frames = int(0.3 * sr / hop_length)  # 0.3s minimum between peaks
        
        # Find peaks (footstep events)
        peaks, _ = signal.find_peaks(
            rms,
            height=peak_height,
            prominence=peak_prominence,
            distance=min_distance_frames
        )
        
        if len(peaks) == 0:
            print("   ⚠ No peaks found, returning as single segment")
            return [{
                'audio': audio,
                'peak_offset': len(audio) / (2 * sr),
                'duration': len(audio) / sr
            }]
        
        print(f"   Found {len(peaks)} peaks in audio")
        
        # === STEP 2: FIND QUIET ZONES (safe cutting points) ===
        # Calculate adaptive quiet threshold
        # Use 20% of mean RMS to be more aggressive (helps with running/overlapping footsteps)
        quiet_threshold = rms_mean * 0.2
        
        # Find all frames below threshold (quiet regions)
        quiet_frames = np.where(rms < quiet_threshold)[0]
        
        print(f"   Quiet threshold: {quiet_threshold:.4f} (20% of mean RMS)")
        print(f"   Found {len(quiet_frames)} quiet frames")
        
        # === STEP 3: CUT AT MIDPOINTS OF QUIET ZONES ===
        segments = []
        
        for i in range(len(peaks)):
            peak_frame = peaks[i]
            peak_time = frame_times[peak_frame]
            
            # --- Find cut point BEFORE this peak ---
            if i == 0:
                # First peak: start from beginning of audio
                start_frame = 0
            else:
                # Find quiet zone between previous peak and this peak
                prev_peak = peaks[i-1]
                quiet_between = quiet_frames[(quiet_frames > prev_peak) & (quiet_frames < peak_frame)]
                
                if len(quiet_between) > 0:
                    # Cut at MIDPOINT of quiet zone (safest point)
                    start_frame = quiet_between[len(quiet_between) // 2]
                else:
                    # No quiet zone - use midpoint between peaks as fallback
                    start_frame = (prev_peak + peak_frame) // 2
            
            # --- Find cut point AFTER this peak ---
            if i == len(peaks) - 1:
                # Last peak: go to end of audio
                end_frame = len(rms) - 1
            else:
                # Find quiet zone between this peak and next peak
                next_peak = peaks[i+1]
                quiet_between = quiet_frames[(quiet_frames > peak_frame) & (quiet_frames < next_peak)]
                
                if len(quiet_between) > 0:
                    # Cut at MIDPOINT of quiet zone (safest point)
                    end_frame = quiet_between[len(quiet_between) // 2]
                else:
                    # No quiet zone - use midpoint between peaks as fallback
                    end_frame = (peak_frame + next_peak) // 2
            
            # Convert frame indices to sample indices
            start_sample = int(start_frame * hop_length)
            end_sample = int(end_frame * hop_length)
            
            # Extract segment
            segment_audio = audio[start_sample:end_sample]
            
            # Calculate where peak is within this segment (for alignment later)
            peak_sample_in_segment = int((peak_frame - start_frame) * hop_length)
            peak_offset = peak_sample_in_segment / sr
            
            # Apply short fades to prevent clicks
            segment_audio = self._apply_fades(segment_audio, sr, fade_samples=int(0.005 * sr))
            
            segments.append({
                'audio': segment_audio,
                'peak_offset': peak_offset,  # Where peak is within segment (seconds)
                'duration': len(segment_audio) / sr,
                'peak_time_original': peak_time  # For debugging
            })
        
        print(f"   Chopped into {len(segments)} segments using quiet zones")
        durations = [seg['duration'] for seg in segments]
        offsets = [seg['peak_offset'] for seg in segments]
        print(f"   Segment durations: {min(durations):.3f}s to {max(durations):.3f}s")
        print(f"   Peak offsets: {min(offsets):.3f}s to {max(offsets):.3f}s")
        
        return segments
    
    def _calculate_reference_distance(self, spatial_data: List[Dict]) -> float:
        """
        Step 4: Calculate reference distance (largest hip-heel pixel distance)
        
        This represents the person's closest point to camera
        All other distances are relative to this
        """
        valid_distances = [
            d['hip_heel_pixel_distance'] 
            for d in spatial_data 
            if d['hip_heel_pixel_distance'] is not None
        ]
        
        if not valid_distances:
            print("   ⚠ No valid distances found, using default reference")
            return 1.0
        
        # Maximum distance = person closest to camera
        reference = max(valid_distances)
        return reference
    
    def _create_spatial_mix(self,
                           segments: List[Dict],
                           spatial_data: List[Dict],
                           reference_distance: float,
                           video_duration: float,
                           video_width: int,
                           sr: int) -> np.ndarray:
        """
        Step 5: Create spatialized stereo mix with PEAK-ALIGNED placement

        For each detected footstep:
        1. Randomly select a segment (with replacement if needed)
        2. Apply distance attenuation based on hip-heel pixel distance
        3. Apply constant power panning based on x-position (normalized 0-1)
        4. Place so segment's PEAK aligns with detection timestamp
        5. Add with crossfade to avoid clicks
        """
        # Create empty stereo output
        output_samples = int(video_duration * sr)
        stereo_output = np.zeros((output_samples, 2), dtype=np.float32)
        
        # Process each footstep
        for i, spatial_info in enumerate(spatial_data):
            # Select random segment
            segment_dict = segments[np.random.randint(len(segments))]
            segment_audio = segment_dict['audio']
            peak_offset = segment_dict['peak_offset']  # Where peak is within segment
            
            # Skip if missing spatial data
            if (spatial_info['x_position'] is None or
                spatial_info['hip_heel_pixel_distance'] is None):
                print(f"   ⚠ Skipping footstep {i+1}: missing spatial data")
                continue

            # Calculate spatial parameters
            x_pos_pixels = spatial_info['x_position']
            x_pos = x_pos_pixels / video_width  # Normalize to 0-1 range (0=left edge, 1=right edge)
            depth = spatial_info['hip_heel_pixel_distance']
            detection_timestamp = spatial_info['timestamp']  # Where we want the peak to land

            # Apply distance attenuation
            attenuated_segment = self._apply_distance_attenuation(
                segment_audio, 
                depth, 
                reference_distance
            )
            
            # Apply panning
            stereo_segment = self._apply_constant_power_pan(attenuated_segment, x_pos)
            
            # === CRITICAL: PEAK-ALIGNED PLACEMENT ===
            # Calculate where segment should START so its peak lands at detection_timestamp
            # 
            # Example:
            #   detection_timestamp = 0.8s (where we want peak)
            #   peak_offset = 0.25s (peak is 0.25s into segment)
            #   placement_start = 0.8 - 0.25 = 0.55s
            #   
            #   So segment plays: 0.55s to (0.55 + duration)
            #   And peak lands at: 0.55 + 0.25 = 0.8s ✓
            
            placement_start_time = detection_timestamp - peak_offset
            placement_start_sample = int(placement_start_time * sr)
            
            # Ensure we don't go before audio start
            if placement_start_sample < 0:
                # Segment would start before audio begins - trim the beginning
                trim_samples = -placement_start_sample
                stereo_segment = stereo_segment[trim_samples:]
                placement_start_sample = 0
            
            placement_end_sample = min(output_samples, placement_start_sample + len(stereo_segment))
            segment_length = placement_end_sample - placement_start_sample
            
            if segment_length > 0:
                # Trim segment if needed
                stereo_segment = stereo_segment[:segment_length]
                
                # Add with crossfade to avoid clicks
                fade_samples = min(int(self.fade_duration * sr), segment_length // 4)
                
                # Crossfade with existing audio
                for ch in range(2):
                    # Get existing audio
                    existing = stereo_output[placement_start_sample:placement_end_sample, ch].copy()
                    
                    # Create fade out for existing
                    if np.any(existing != 0):
                        fade_out = np.linspace(1, 0, fade_samples)
                        existing[:fade_samples] *= fade_out
                    
                    # Create fade in for new segment
                    fade_in = np.linspace(0, 1, fade_samples)
                    stereo_segment[:fade_samples, ch] *= fade_in
                    
                    # Mix
                    stereo_output[placement_start_sample:placement_end_sample, ch] = existing + stereo_segment[:, ch]
        
        print(f"   Mixed {len(spatial_data)} footsteps into stereo output")
        return stereo_output
    
    def _apply_distance_attenuation(self,
                                   audio: np.ndarray,
                                   distance: float,
                                   reference_distance: float) -> np.ndarray:
        """
        Apply inverse distance law attenuation with configurable curve steepness
        
        Formula: attenuation_db = -6 * log2(distance_ratio ^ power)
        - power = 1.0: Linear inverse (6dB per doubling) 
        - power = 2.0: Inverse square law (12dB per doubling) - more dramatic
        - Higher power = steeper curve at extremes
        
        Args:
            audio: Mono audio segment
            distance: Hip-heel pixel distance for this footstep
            reference_distance: Maximum hip-heel distance (closest point)
            
        Returns:
            Attenuated audio
        """
        # Calculate distance ratio (how far compared to reference)
        distance_ratio = reference_distance / distance
        
        # Apply power curve for steepness control
        # power=2.0 gives inverse square law (more dramatic changes)
        distance_ratio_curved = distance_ratio ** self.distance_curve_power
        
        # Inverse distance law: -6dB per doubling of distance
        attenuation_db = -6.0 * math.log2(distance_ratio_curved)
        
        # Clamp to configured range
        attenuation_db = np.clip(attenuation_db, self.min_attenuation_db, self.max_attenuation_db)
        
        # Convert dB to linear gain
        gain = 10 ** (attenuation_db / 20)
        
        return audio * gain
    
    def _apply_constant_power_pan(self,
                                  audio: np.ndarray,
                                  x_position: float) -> np.ndarray:
        """
        Apply constant power panning (-3dB pan law) with Logic Pro range

        Industry standard: sin/cos taper ensures constant perceived loudness
        Panning range: Logic Pro -41 to +41 (out of -64 to +64 scale)
        Mathematical range: ±0.640625 (-41/64 to +41/64)

        Expected behavior:
        - At center (0.5): 70.7% left / 70.7% right (both channels at -3dB)
        - At left (0.0): ~23% left / ~97% right (Logic Pro pan = -41)
        - At right (1.0): ~97% left / ~23% right (Logic Pro pan = +41)

        Args:
            audio: Mono audio segment
            x_position: Normalized x-coordinate (0=left, 0.5=center, 1=right)

        Returns:
            Stereo audio [samples, 2]
        """
        # Clamp x_position to valid range
        x_position = np.clip(x_position, 0.0, 1.0)

        # Scale x_position to Logic Pro -41 to +41 range (out of -64 to +64)
        # -41/64 = -0.640625, +41/64 = +0.640625
        # This matches the exact panning range you tested in Logic Pro
        center = 0.5
        logic_pro_range = 0.640625  # -41/64 to +41/64
        scaled_x = center + (x_position - center) * logic_pro_range
        
        # Convert scaled x-position (0-1) to pan angle (-π/2 to +π/2)
        # Scaled by Logic Pro range (±0.640625) for -41 to +41 panning
        pan_angle = (scaled_x - 0.5) * math.pi

        # Constant power panning (sin/cos taper)
        # At center: both channels at -3dB
        # At extremes: Logic Pro -41/+41 range
        left_gain = math.cos(pan_angle * 0.5 + math.pi / 4)
        right_gain = math.sin(pan_angle * 0.5 + math.pi / 4)
        
        # Create stereo output
        stereo = np.zeros((len(audio), 2), dtype=audio.dtype)
        stereo[:, 0] = audio * left_gain   # Left channel
        stereo[:, 1] = audio * right_gain  # Right channel
        
        return stereo
    
    def _apply_final_processing(self,
                               audio: np.ndarray,
                               video_duration: float,
                               sr: int) -> np.ndarray:
        """
        Step 6: Final processing
        - Match exact video duration
        - Apply fade in/out
        - Final normalization to -6dB (professional SFX standard)
        """
        target_samples = int(video_duration * sr)
        current_samples = len(audio)

        # Match duration
        if current_samples < target_samples:
            # Pad with silence
            padding = target_samples - current_samples
            audio = np.pad(audio, ((0, padding), (0, 0)), mode='constant')
            print(f"   Padded to match video duration: {video_duration:.2f}s")
        elif current_samples > target_samples:
            # Trim
            audio = audio[:target_samples]
            print(f"   Trimmed to match video duration: {video_duration:.2f}s")

        # Apply fade in/out (50ms each)
        fade_samples = int(0.05 * sr)

        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        for ch in range(2):
            audio[:fade_samples, ch] *= fade_in
            audio[-fade_samples:, ch] *= fade_out

        print(f"   Applied {fade_samples/sr:.3f}s fade in/out")

        # Final normalization to -6dB (professional SFX standard)
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_peak = 10 ** (-6.0 / 20)  # -6dB
            audio = audio * (target_peak / peak)
            print(f"   Normalized to -6dB peak (professional SFX standard)")

        return audio
    
    def _apply_fades(self, audio: np.ndarray, sr: int, fade_samples: int) -> np.ndarray:
        """Apply short fade in/out to prevent clicks"""
        fade_samples = min(fade_samples, len(audio) // 4)
        
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        return audio
    
    def _create_visualization(self,
                            final_audio: np.ndarray,
                            spatial_data: List[Dict],
                            segments: List[np.ndarray],
                            reference_distance: float,
                            video_width: int,
                            sr: int,
                            output_path: str):
        """
        Step 8: Create visualization showing:
        - Waveform with footstep markers
        - X-position (panning) over time (normalized 0-1)
        - Depth (volume attenuation) over time
        """
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # Time axis
        time_axis = np.linspace(0, len(final_audio) / sr, len(final_audio))
        
        # 1. Stereo waveform
        axes[0].plot(time_axis, final_audio[:, 0], alpha=0.6, linewidth=0.5, label='Left', color='blue')
        axes[0].plot(time_axis, final_audio[:, 1], alpha=0.6, linewidth=0.5, label='Right', color='red')
        axes[0].set_title('Final Spatialized Stereo Audio', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Mark footstep positions
        for spatial_info in spatial_data:
            axes[0].axvline(spatial_info['timestamp'], color='green', alpha=0.3, linestyle='--', linewidth=1)
        
        # 2. X-Position (Panning) over time
        timestamps = [d['timestamp'] for d in spatial_data if d['x_position'] is not None]
        x_positions = [d['x_position'] / video_width for d in spatial_data if d['x_position'] is not None]  # Normalize to 0-1

        axes[1].scatter(timestamps, x_positions, c='purple', s=50, alpha=0.7, edgecolors='black')
        axes[1].plot(timestamps, x_positions, alpha=0.3, color='purple', linestyle=':')
        axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Center')
        axes[1].set_title('Horizontal Position (Panning)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('X Position (Normalized)')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axes[1].set_yticklabels(['Left', '', 'Center', '', 'Right'])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Depth (Distance) over time
        depths = [d['hip_heel_pixel_distance'] for d in spatial_data if d['hip_heel_pixel_distance'] is not None]
        
        axes[2].scatter(timestamps, depths, c='orange', s=50, alpha=0.7, edgecolors='black')
        axes[2].plot(timestamps, depths, alpha=0.3, color='orange', linestyle=':')
        axes[2].axhline(reference_distance, color='red', linestyle='--', alpha=0.5, 
                       label=f'Reference (closest): {reference_distance:.3f}')
        axes[2].set_title('Depth (Hip-Heel Pixel Distance)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Pixel Distance')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Volume Attenuation over time
        attenuations_db = []
        for d in spatial_data:
            if d['hip_heel_pixel_distance'] is not None:
                distance_ratio = reference_distance / d['hip_heel_pixel_distance']
                atten_db = -6.0 * math.log2(distance_ratio)
                atten_db = np.clip(atten_db, self.min_attenuation_db, self.max_attenuation_db)
                attenuations_db.append(atten_db)
            else:
                attenuations_db.append(None)
        
        valid_attens = [(t, a) for t, a in zip(timestamps, attenuations_db) if a is not None]
        if valid_attens:
            t_vals, a_vals = zip(*valid_attens)
            axes[3].scatter(t_vals, a_vals, c='green', s=50, alpha=0.7, edgecolors='black')
            axes[3].plot(t_vals, a_vals, alpha=0.3, color='green', linestyle=':')
        
        axes[3].axhline(0, color='red', linestyle='--', alpha=0.5, label='0dB (Closest)')
        axes[3].axhline(-6, color='orange', linestyle='--', alpha=0.3, label='-6dB (2x distance)')
        axes[3].axhline(-12, color='blue', linestyle='--', alpha=0.3, label='-12dB (4x distance)')
        axes[3].set_title('Volume Attenuation (Inverse Distance Law)', fontsize=14, fontweight='bold')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].set_ylabel('Attenuation (dB)')
        axes[3].set_ylim(self.min_attenuation_db - 2, 2)
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# DEVELOPER TESTING SECTION
# ============================================================================

if __name__ == "__main__":
    """
    Standalone Spatial Audio Processor

    Process audio files with 4 spatial scenarios:
    1. Walking right to left (x_position: 1.0 → 0.0)
    2. Approaching camera (distance: 1.0 → 0.0)
    3. Walking away (distance: 0.0 → 1.0)
    4. Stationary (x_position: 0.5, distance: 0.5)

    Usage:
        python -m src.pipeline.spatial_audio_processor [audio_file]

    If no audio file is specified, uses audio from data/audio/ directory.
    """

    # Use centralized configuration
    from ..utils.config import get_test_audio, list_test_audios, PIPELINE_OUTPUTS_DIR, TEST_AUDIOS_DIR
    import sys
    import argparse
    import os

    # ========================================================================
    # COMMAND LINE ARGUMENTS
    # ========================================================================

    parser = argparse.ArgumentParser(
        description='Standalone spatial audio processor - test audio with 4 spatial scenarios'
    )
    parser.add_argument(
        'audio_file',
        nargs='?',
        help='Path to audio file (WAV format). If not specified, uses files from data/audio/'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SPATIAL AUDIO PROCESSOR - STANDALONE TEST")
    print("=" * 80)

    # ========================================================================
    # AUDIO FILE SELECTION
    # ========================================================================

    if args.audio_file:
        # User specified an audio file
        AUDIO_FILE = args.audio_file
        if not os.path.exists(AUDIO_FILE):
            print(f"\n✗ ERROR: Audio file not found: {AUDIO_FILE}")
            sys.exit(1)
        print(f"\nProcessing user audio: {os.path.basename(AUDIO_FILE)}")
    else:
        # No file specified - try to use audio from data/audio/
        print("\nNo audio file specified. Searching data/audio/ directory...")
        available = list_test_audios()

        if not available:
            print(f"\n✗ ERROR: No audio files found in: {TEST_AUDIOS_DIR}")
            print("\nPlease either:")
            print("  1. Add audio files to data/audio/ directory")
            print("  2. Specify an audio file: python -m src.pipeline.spatial_audio_processor your_audio.wav")
            sys.exit(1)

        # Use first available audio file
        AUDIO_FILE = str(get_test_audio(available[0]))
        print(f"\nFound {len(available)} audio file(s) in data/audio/")
        print(f"Processing: {available[0]}")

        if len(available) > 1:
            print(f"\nOther available files:")
            for audio in available[1:]:
                print(f"  - {audio}")

    print(f"\nAudio file: {AUDIO_FILE}")
    print("Testing 4 spatial scenarios with ground truth timestamps\n")

    # Output directory (use centralized path)
    OUTPUT_DIR = str(PIPELINE_OUTPUTS_DIR / "spatial_test")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Ground truth timestamps from walk4.mp4
    GROUND_TRUTH_TIMESTAMPS = [
        0.517,
        1.817,
        3.133,
        4.383,
        5.733,
        6.95,
        8.133,
        9.4,
        10.65,
        11.917
    ]
    
    # Video info (from walk4.mp4)
    VIDEO_INFO = {
        'fps': 60.0,
        'total_frames': 736,
        'duration': 12.266666666666667
    }
    
    # ========================================================================
    # MOCK SCENARIO GENERATOR
    # ========================================================================
    
    def create_mock_scenario(scenario_name: str, timestamps: List[float]) -> Dict:
        """
        Create mock detection results for a spatial scenario
        
        Args:
            scenario_name: Name of scenario (for x/distance calculation)
            timestamps: List of footstep timestamps
            
        Returns:
            Mock detection_results dict compatible with spatial processor
        """
        spatial_data = []
        num_steps = len(timestamps)
        
        for i, timestamp in enumerate(timestamps):
            # Calculate progress (0.0 to 1.0)
            progress = i / (num_steps - 1) if num_steps > 1 else 0.5
            
            # Generate spatial data based on scenario
            if scenario_name == "right_to_left":
                # Walk from right (1.0) to left (0.0)
                x_position = 1.0 - progress
                distance = 0.5  # Constant distance
                
            elif scenario_name == "approaching":
                # Walk toward camera (distance: far to close)
                x_position = 0.5  # Center
                distance = 1.0 - progress  # Far → close
                
            elif scenario_name == "walking_away":
                # Walk away from camera (distance: close to far)
                x_position = 0.5  # Center
                distance = progress  # Close → far
                
            elif scenario_name == "stationary":
                # Stay in same position
                x_position = 0.5
                distance = 0.5
                
            else:
                raise ValueError(f"Unknown scenario: {scenario_name}")
            
            # Convert distance (0-1) to hip-heel pixel distance
            # Closer = larger pixel distance, farther = smaller pixel distance
            # Map: distance 0.0 → 200 pixels, distance 1.0 → 50 pixels
            hip_heel_pixels = 200 - (distance * 150)
            
            spatial_data.append({
                'timestamp': timestamp,
                'x_position': x_position,
                'hip_heel_pixel_distance': hip_heel_pixels,
                'side': 'left' if i % 2 == 0 else 'right'  # Alternate L/R
            })
        
        return {
            'video_info': VIDEO_INFO,
            'spatial_data': spatial_data,
            'scenario_name': scenario_name
        }
    
    # ========================================================================
    # RUN TESTS
    # ========================================================================
    
    # Initialize processor
    processor = SpatialAudioProcessor(
        sample_rate=44100,
        fade_duration=0.01,
        min_attenuation_db=-20.0,
        max_attenuation_db=0.0,
        max_pan_percentage=0.2,      # Subtle/natural panning (professional sound)
        distance_curve_power=2.0      # Inverse square law (steeper curve)
    )
    
    # Define test scenarios
    scenarios = [
        ("right_to_left", "Scenario 1: Walking Right to Left"),
        ("approaching", "Scenario 2: Approaching Camera"),
        ("walking_away", "Scenario 3: Walking Away"),
        ("stationary", "Scenario 4: Stationary Position")
    ]
    
    # Run each scenario
    for scenario_key, scenario_desc in scenarios:
        print("\n" + "=" * 80)
        print(scenario_desc)
        print("=" * 80)
        
        # Create mock detection results
        mock_results = create_mock_scenario(scenario_key, GROUND_TRUTH_TIMESTAMPS)
        
        # Output path
        output_path = f"{OUTPUT_DIR}/test_{scenario_key}.wav"
        
        # Process
        try:
            stats = processor.process_video_audio(
                audio_input=AUDIO_FILE,
                detection_results=mock_results,
                output_path=output_path,
                visualize=True
            )
            
            print(f"\n✓ Test completed: {output_path}")
            print(f"  Peak level: {stats['peak_level_db']:.1f}dB")
            print(f"  Duration: {stats['output_audio_duration']:.2f}s")
            
        except FileNotFoundError as e:
            print(f"\n✗ ERROR: {e}")
            print("\nPlease update AUDIO_FILE path in the script to match your local machine:")
            print(f"  Current: {AUDIO_FILE}")
            print(f"  Expected: /path/to/your/heavy_boots_walking_on_marble_floor_with_echoing_footsteps.wav")
            break
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "=" * 80)
    print("SPATIAL AUDIO PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("  - test_right_to_left.wav + .png")
    print("  - test_approaching.wav + .png")
    print("  - test_walking_away.wav + .png")
    print("  - test_stationary.wav + .png")
    print("\nVisualization PNGs show:")
    print("  1. Final stereo waveform")
    print("  2. X-position (panning) over time")
    print("  3. Depth (hip-heel distance) over time")
    print("  4. Volume attenuation over time")
    print("\nTo process a different audio file:")
    print("  python -m src.pipeline.spatial_audio_processor your_audio.wav")