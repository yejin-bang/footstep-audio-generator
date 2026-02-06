import json
from pathlib import Path
import random
from typing import List, Dict, Tuple
from collections import defaultdict
import re

class HighQualityCaptionGenerator:
    def __init__(self, caption_config_path: str):
        """Load caption_config.json with all vocabulary banks"""
        with open(caption_config_path, 'r') as f:
            self.config = json.load(f)
        
        self.surface_sound_map = self.config['surface_sound_map']
        self.footwear_materials = self.config['footwear_materials']
        
        self.timbres = self.config['timbres']
        self.intensities = self.config['intensities']
        self.locations = self.config['locations']
        
        self.actions = self.config['actions']
        self.forbidden_words = self.config['forbidden_words']
        
        # Statistics tracking
        self.stats = defaultdict(int)
        self.warnings = []
    
    def get_location(self, surface: str) -> str:
        """Get location for surface"""
        return random.choice(self.locations.get(surface, ['surface']))
    
    def get_sound_descriptor(self, surface: str, footwear: str) -> str:
        """Get sound descriptor from surface_sound_map"""
        if surface not in self.surface_sound_map:
            self.warnings.append(f"Missing surface '{surface}' in surface_sound_map")
            return "stepping"
        
        if footwear not in self.surface_sound_map[surface]:
            self.warnings.append(f"Missing footwear '{footwear}' for surface '{surface}'")
            available = list(self.surface_sound_map[surface].keys())
            if available:
                footwear_alt = available[0]
                return random.choice(self.surface_sound_map[surface][footwear_alt])
            return "stepping"
        
        sounds = self.surface_sound_map[surface][footwear]
        if len(sounds) < 3:
            self.warnings.append(f"Only {len(sounds)} sound options for {surface}/{footwear}")
        
        return random.choice(sounds)
    
    def check_forbidden_words(self, caption: str) -> bool:
        """Check if caption contains forbidden reverb words"""
        for word in self.forbidden_words:
            if re.search(r'\b' + word + r'\b', caption, re.IGNORECASE):
                self.warnings.append(f"FORBIDDEN WORD '{word}' in caption: {caption}")
                return False
        return True
    
    def count_words(self, text: str) -> int:
        """Count words in caption"""
        return len(text.split())
    
    def validate_length(self, caption: str, target_length: str) -> bool:
        """Validate caption length matches target"""
        word_count = self.count_words(caption)
        
        ranges = {
            'short': (5, 8),
            'medium': (9, 15),
            'long': (16, 25)
        }
        
        min_words, max_words = ranges[target_length]
        if min_words <= word_count <= max_words:
            return True
        else:
            self.warnings.append(f"Length mismatch: {caption} ({word_count} words, expected {target_length})")
            return False
    
    def generate_material_caption(self, footwear: str, surface: str, target_length: str) -> str:
        """
        Generate material-focused caption (physical objects only)
        Focus: What physical objects are involved
        """
        material = random.choice(self.footwear_materials[footwear])
        action = random.choice(self.actions)
        location = random.choice(self.locations[surface])
        sound = self.get_sound_descriptor(surface, footwear)
        
        if target_length == 'short':
            # 5-6 words
            templates = [
                f"{material} {action} on {surface}",
                f"{material} {action} across {surface}",
                f"{material} on {surface} {location}"
            ]
            caption = random.choice(templates)
        
        elif target_length == 'medium':
            # 9-12 words
            templates = [
                f"{material} {action} steadily on {surface} {location}",
                f"{material} {action} rhythmically across {surface} {location}",
                f"{material} {action} continuously on the {surface} {location}",
                f"{material} {action} on {surface} {location}"
            ]
            caption = random.choice(templates)
        
        else:  # long
            # 16-20 words - every word adds meaningful information
            templates = [
                f"{material} {action} across the {surface} {location} producing distinct {sound}",
                f"{material} {action} continuously in a regular pattern on the {surface} {location}",
                f"{material} {action} steadily on {surface} {location} creating regular {sound}",
                f"{material} {action} continuously across the {surface} {location} with even spacing"
            ]
            caption = random.choice(templates)
        
        return caption

    def generate_context_caption(self, footwear: str, surface: str, target_length: str) -> str:
        """
        Generate context-focused caption (scenario/perspective)
        Focus: Who is walking and where (the scene/situation)
        """
        material = random.choice(self.footwear_materials[footwear])
        action = random.choice(self.actions)
        location = random.choice(self.locations[surface])

        
        if target_length == 'short':
            # 5-7 words
            templates = [
                f"person in {footwear.replace('_',' ')} {action} on {surface}",
                f"person wearing {footwear.replace('_',' ')} on {surface}",
                f"{footwear.replace('_',' ')} {action} on {surface}",
                f"someone in {footwear.replace('_',' ')} {action}"
            ]
            caption = random.choice(templates)
        
        elif target_length == 'medium':
            # 10-13 words
            templates = [
                f"person wearing {material} {action} steadily across the {surface} {location}",
                f"someone in {material} {action} on {surface} {location}",
                f"person in {footwear.replace('_',' ')} {action} rhythmically on the {surface} {location}",
                f"individual wearing {material} {action} continuously across {surface} {location}"
            ]
            caption = random.choice(templates)
        
        else:  # long
            # 17-20 words - every word adds meaningful information
            templates = [
                f"person wearing {material} {action} at moderate pace across the {surface} {location}",
                f"individual in {material} {action} continuously in a regular pattern on the {surface} {location}",
                f"individual in {material} {action} continuously on the {surface} {location}",
                f"person in {footwear.replace('_',' ')} {action} at steady pace on the {surface} {location}"
            ]
            caption = random.choice(templates)
        
        return caption

    def generate_character_caption(self, footwear: str, surface: str, target_length: str) -> str:
        """
        Generate character-focused caption (sound qualities only)
        Focus: How the sound feels/sounds (timbre, intensity, sound type)
        """
        material = random.choice(self.footwear_materials[footwear])
        timbre = random.choice(self.timbres[footwear])
        intensity = random.choice(self.intensities[footwear])
        sound = self.get_sound_descriptor(surface, footwear)
        
        if target_length == 'short':
            # 5-7 words
            templates = [
                f"{timbre} {sound} from {footwear.replace('_',' ')}",
                f"{timbre} {sound} on {surface}",
                f"{intensity} {sound} from {footwear.replace('_',' ')}",
                f"{sound} from {footwear.replace('_',' ')} on {surface}",
                f"{timbre} {intensity} {sound}"
            ]
            caption = random.choice(templates)
        
        elif target_length == 'medium':
            # 10-13 words
            templates = [
                f"{timbre} {intensity} {sound} from {footwear.replace('_',' ')} on the {surface} surface",
                f"{timbre} {sound} with {intensity} contact from {footwear.replace('_',' ')} on {surface}",
                f"{intensity} {timbre} {sound} created by {footwear.replace('_',' ')} against {surface}",
                f"{timbre} {sound} from {material} making {intensity} contact on {surface}"
            ]
            caption = random.choice(templates)
        
        else:  # long
            # 16-20 words
            templates = [
                f"{timbre} {intensity} {sound} from {material} on {surface} with consistent rhythmic pattern",
                f"{timbre} {sound} with {intensity} impact as {material} contacts {surface} repeatedly",
                f"{intensity} {timbre} {sound} produced by {material} against {surface} at moderate pace",
                f"{timbre} {sound} from {material} on {surface} creating distinct acoustic signature"
            ]
            caption = random.choice(templates)
        
        return caption
    
    def generate_3_captions(self, footwear: str, surface: str) -> List[Dict]:
        """
        Generate 3 high-quality captions per clip:
        - 1 Material caption (required)
        - 1 Context caption (required)
        - 1 Character caption (required)
        
        Each gets a random length: distribution aims for 30/50/20 short/medium/long
        """
        captions = []
        
        # Weighted random selection to achieve 30/50/20 distribution
        length_pool = ['short'] * 30 + ['medium'] * 50 + ['long'] * 20
        lengths = random.sample(length_pool, 3)
        
        # Caption 1: Material-focused
        material_caption = self.generate_material_caption(footwear, surface, lengths[0])
        if self.check_forbidden_words(material_caption):
            self.validate_length(material_caption, lengths[0])
            captions.append({
                'text': material_caption,
                'type': 'material',
                'length': lengths[0],
                'word_count': self.count_words(material_caption)
            })
            self.stats['material_captions'] += 1
        
        # Caption 2: Context-focused
        context_caption = self.generate_context_caption(footwear, surface, lengths[1])
        if self.check_forbidden_words(context_caption):
            self.validate_length(context_caption, lengths[1])
            captions.append({
                'text': context_caption,
                'type': 'context',
                'length': lengths[1],
                'word_count': self.count_words(context_caption)
            })
            self.stats['context_captions'] += 1
        
        # Caption 3: Character-focused
        character_caption = self.generate_character_caption(footwear, surface, lengths[2])
        if self.check_forbidden_words(character_caption):
            self.validate_length(character_caption, lengths[2])
            captions.append({
                'text': character_caption,
                'type': 'character',
                'length': lengths[2],
                'word_count': self.count_words(character_caption)
            })
            self.stats['character_captions'] += 1
        
        return captions
    
    def process_metadata_file(self, input_jsonl: str, output_jsonl: str):
        """
        Process existing metadata JSONL and add 3 captions per clip
        
        Output JSONL format (3 lines per clip):
        {"file_name": "...", "caption": "caption1", "caption_type": "material", ...}
        {"file_name": "...", "caption": "caption2", "caption_type": "context", ...}
        {"file_name": "...", "caption": "caption3", "caption_type": "character", ...}
        """
        print("=" * 60)
        print("HIGH-QUALITY CAPTION GENERATOR")
        print("=" * 60)
        print(f"Processing: {input_jsonl}")
        
        # Read input clips
        with open(input_jsonl, 'r') as f:
            clips = [json.loads(line) for line in f]
        
        print(f"Found {len(clips)} clips")
        
        # Generate captions
        expanded_clips = []
        
        for clip in clips:
            footwear = clip['footwear']
            surface = clip['surface']
            
            # Generate 3 captions
            captions = self.generate_3_captions(footwear, surface)
            
            # Create 3 entries with same audio but different captions
            for cap in captions:
                entry = clip.copy()
                entry['caption'] = cap['caption']
                entry['caption_type'] = cap['type']
                entry['caption_length'] = cap['length']
                entry['caption_word_count'] = cap['word_count']
                expanded_clips.append(entry)
            
            self.stats['clips_processed'] += 1
        
        # Save expanded metadata
        with open(output_jsonl, 'w') as f:
            for clip in expanded_clips:
                f.write(json.dumps(clip) + '\n')
        
        print(f"\nGenerated {len(expanded_clips)} caption entries from {len(clips)} clips")
        print(f"Saved to: {output_jsonl}")
        
        return expanded_clips
    
    def print_statistics(self):
        """Print generation statistics"""
        print("\n" + "=" * 60)
        print("CAPTION GENERATION STATISTICS")
        print("=" * 60)
        
        print(f"Total clips processed: {self.stats['clips_processed']}")
        print(f"\nCaption type distribution:")
        print(f"  Material: {self.stats['material_captions']}")
        print(f"  Context: {self.stats['context_captions']}")
        print(f"  Character: {self.stats['character_captions']}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warnings generated")
            print("Check warnings.txt for details")
    
    def save_warnings(self, output_path: str):
        """Save warnings to file"""
        if self.warnings:
            with open(output_path, 'w') as f:
                f.write("CAPTION GENERATION WARNINGS\n")
                f.write("=" * 60 + "\n\n")
                for warning in set(self.warnings):
                    f.write(f"⚠️  {warning}\n")
            print(f"Warnings saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate captions for audio clips')
    parser.add_argument('--config', required=True, help='Path to caption_config.json')
    parser.add_argument('--metadata_dir', required=True, help='Directory containing train/val/test.jsonl files')
    
    args = parser.parse_args()
    
    generator = HighQualityCaptionGenerator(args.config)
    
    for split in ['train', 'val', 'test']:
        input_jsonl = Path(args.metadata_dir) / f"{split}.jsonl"
        output_jsonl = Path(args.metadata_dir) / f"{split}_captioned.jsonl"
        
        if input_jsonl.exists():
            generator.process_metadata_file(str(input_jsonl), str(output_jsonl))
    
    generator.print_statistics()
    generator.save_warnings(str(Path(args.metadata_dir) / "caption_warnings.txt"))
    
    print("\n✅ Caption generation complete!")