#!/usr/bin/env python3
"""
Asset Scanner for Footstep Audio Library

Analyzes existing audio assets and prepares them for LoRA training dataset.
Handles inconsistent naming and varying file lengths.
"""

import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import re
from collections import defaultdict
from datetime import datetime

class FootstepAssetScanner:
    def __init__(self, assets_path: str):
        """Initialize scanner with path to raw assets"""
        self.assets_path = Path(assets_path)
        self.inventory = defaultdict(lambda: defaultdict(list))
        self.file_stats = []
        
        # Common footwear keywords for detection
        self.footwear_keywords = {
            'barefoot': ['barefoot', 'bare', 'naked', 'skin'],
            'boots': ['boots', 'boot', 'heavy', 'combat', 'work'],
            'sneakers': ['sneakers', 'sneaker', 'tennis', 'athletic', 'running', 'casual'],
            'dress_shoes': ['dress', 'formal', 'leather', 'oxford', 'loafer'],
            'heels': ['heels', 'heel', 'high', 'stiletto', 'pump']
        }
        
        # Audio file extensions
        self.audio_extensions = {'.wav', '.mp3', '.aiff', '.flac', '.m4a'}
    
    def scan_assets(self, verbose=True):
        """Scan all audio assets and build inventory"""
        print("=" * 60)
        print("FOOTSTEP ASSET SCANNER")
        print("=" * 60)
        print(f"Scanning: {self.assets_path}")
        
        if not self.assets_path.exists():
            print(f"❌ Assets path not found: {self.assets_path}")
            return
        
        # Scan each surface directory
        surface_dirs = [d for d in self.assets_path.iterdir() if d.is_dir()]
        
        if not surface_dirs:
            print("❌ No surface directories found")
            return
        
        print(f"Found {len(surface_dirs)} surface directories")
        print()
        
        total_files = 0
        total_duration = 0
        
        for surface_dir in sorted(surface_dirs):
            surface_name = surface_dir.name
            print(f"📁 Scanning {surface_name}/")
            
            # Get all audio files in this surface directory
            audio_files = []
            for ext in self.audio_extensions:

                audio_files.extend(surface_dir.rglob(f"*{ext}"))
            
            surface_files = 0
            surface_duration = 0
            
            for audio_file in audio_files:
                try:
                    # Analyze audio file
                    file_info = self._analyze_audio_file(audio_file, surface_name)
                    if file_info:
                        self.file_stats.append(file_info)
                        
                        # Detect footwear type from filename
                        footwear = self._detect_footwear(audio_file.name)
                        
                        # Add to inventory
                        self.inventory[surface_name][footwear].append({
                            'file_path': str(audio_file),
                            'filename': audio_file.name,
                            'duration': file_info['duration'],
                            'sample_rate': file_info['sample_rate'],
                            'channels': file_info['channels']
                        })
                        
                        surface_files += 1
                        surface_duration += file_info['duration']
                        
                        if verbose:
                            print(f"   ✅ {audio_file.name[:50]:<50} | {footwear:<12} | {file_info['duration']:.1f}s")
                            
                except Exception as e:
                    print(f"   ❌ Error processing {audio_file.name}: {e}")
            
            print(f"   📊 {surface_files} files, {surface_duration:.1f}s total")
            print()
            
            total_files += surface_files
            total_duration += surface_duration
        
        print("=" * 60)
        print(f"SCAN COMPLETE: {total_files} files, {total_duration:.1f}s total")
        print("=" * 60)
        
        return self.inventory, self.file_stats
    
    def _analyze_audio_file(self, file_path: Path, surface: str) -> Dict:
        """Analyze a single audio file"""
        try:
            # Get basic info without loading full audio (faster)
            info = sf.info(str(file_path))
            
            return {
                'file_path': str(file_path),
                'filename': file_path.name,
                'surface': surface,
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'bit_depth': info.subtype,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            print(f"Warning: Could not analyze {file_path.name}: {e}")
            return None
    
    def _detect_footwear(self, filename: str) -> str:
        """Detect footwear type from filename using keywords"""
        filename_lower = filename.lower()
        
        # Check each footwear type
        for footwear_type, keywords in self.footwear_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return footwear_type
        
        # Default fallback
        return 'unknown'
    def check_format_compliance(self):
        """Check which files are not 44.1kHz or not 16-bit"""
        print("\n" + "=" * 60)
        print("FORMAT COMPLIANCE CHECK")
        print("=" * 60)
        
        non_compliant_files = []
        
        for file_stat in self.file_stats:
            issues = []
            
            if file_stat['sample_rate'] != 44100:
                issues.append(f"Sample rate: {file_stat['sample_rate']}Hz")
            
            if 'PCM_16' not in file_stat['bit_depth']:
                issues.append(f"Bit depth: {file_stat['bit_depth']}")
            
            if issues:
                non_compliant_files.append({
                    'filename': file_stat['filename'],
                    'path': file_stat['file_path'],
                    'issues': issues
                })
        
        if not non_compliant_files:
            print("✅ ALL FILES ARE COMPLIANT!")
            print("   44.1kHz, 16-bit PCM")
        else:
            print(f"⚠️  {len(non_compliant_files)} files need attention:\n")
            
            for i, file_info in enumerate(non_compliant_files, 1):
                print(f"{i}. {file_info['filename']}")
                for issue in file_info['issues']:
                    print(f"   - {issue}")
                print()
        
        print(f"Total files scanned: {len(self.file_stats)}")
        print(f"Compliant: {len(self.file_stats) - len(non_compliant_files)}")
        print(f"Non-compliant: {len(non_compliant_files)}")
        
        return non_compliant_files

    def generate_inventory_report(self) -> str:
        """Generate detailed inventory report"""
        report = []
        report.append("FOOTSTEP AUDIO INVENTORY REPORT")
        report.append("=" * 50)
        
        # Summary statistics
        total_files = sum(len(files) for surface in self.inventory.values() 
                         for files in surface.values())
        total_surfaces = len(self.inventory)
        
        report.append(f"Total Surfaces: {total_surfaces}")
        report.append(f"Total Files: {total_files}")
        report.append("")
        
        # Surface breakdown
        for surface, footwear_dict in sorted(self.inventory.items()):
            report.append(f"📁 {surface.upper()}")
            surface_total = sum(len(files) for files in footwear_dict.values())
            report.append(f"   Total files: {surface_total}")
            
            for footwear, files in sorted(footwear_dict.items()):
                if files:  # Only show footwear types that have files
                    total_duration = sum(f['duration'] for f in files)
                    avg_duration = total_duration / len(files) if files else 0
                    report.append(f"   {footwear:<12} | {len(files):3d} files | {total_duration:6.1f}s total | {avg_duration:4.1f}s avg")
            
            report.append("")
        
        return "\n".join(report)
    
    def get_combination_matrix(self) -> Dict:
        """Get surface x footwear combination matrix for training planning"""
        matrix = {}
        
        for surface, footwear_dict in self.inventory.items():
            matrix[surface] = {}
            for footwear, files in footwear_dict.items():
                if files:  # Only include combinations that have files
                    matrix[surface][footwear] = len(files)
        
        return matrix
    
    def save_inventory(self, output_path: str = None):
        """Save inventory to JSON file"""
        if output_path is None:
            output_path = self.assets_path.parent / "asset_inventory.json"
        
        inventory_data = {
            'scan_timestamp': str(datetime.now()),
            'assets_path': str(self.assets_path),
            'inventory': dict(self.inventory),
            'file_stats': self.file_stats,
            'combination_matrix': self.get_combination_matrix()
        }
        
        with open(output_path, 'w') as f:
            json.dump(inventory_data, f, indent=2, default=str)
        
        print(f"📄 Inventory saved to: {output_path}")
        return output_path
    
    def recommend_dataset_strategy(self):
        """Recommend dataset creation strategy based on inventory"""
        print("\n" + "=" * 50)
        print("DATASET STRATEGY RECOMMENDATIONS")
        print("=" * 50)
        
        matrix = self.get_combination_matrix()
        total_combinations = sum(1 for surface in matrix.values() for footwear in surface.keys())
        total_files = sum(count for surface in matrix.values() for count in surface.values())
        
        print(f"Available combinations: {total_combinations}")
        print(f"Total source files: {total_files}")
        
        # Check for file length issues
        long_files = [f for f in self.file_stats if f['duration'] > 10]
        short_files = [f for f in self.file_stats if f['duration'] < 3]
        
        print(f"\nFile length analysis:")
        print(f"  Files > 10s: {len(long_files)} (need chopping)")
        print(f"  Files < 3s: {len(short_files)} (may need padding/grouping)")
        
        return matrix


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scan raw audio assets and create inventory')
    parser.add_argument('--assets_path', required=True, help='Path to raw audio assets directory')
    parser.add_argument('--output', default=None, help='Output path for inventory JSON (default: assets_path/../asset_inventory.json)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed file information')
    
    args = parser.parse_args()
    
    scanner = FootstepAssetScanner(args.assets_path)
    inventory, file_stats = scanner.scan_assets(verbose=args.verbose)
    
    report = scanner.generate_inventory_report()
    print(report)
    
    non_compliant = scanner.check_format_compliance()
    scanner.recommend_dataset_strategy()
    
    scanner.save_inventory(args.output)
    
    print("\n🎯 Ready for dataset creation!")