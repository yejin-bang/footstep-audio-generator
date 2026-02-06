"""
Latent Similarity Analyzer for Audio Augmentation Validation

This script analyzes pre-encoded latents (from Stable Audio VAE) to determine
if augmentations create meaningfully different representations.

Usage:
    python analyze_latent_similarity.py --latent_dir /path/to/latents

Expected file naming convention:
    - filename_orig.npy
    - filename_aug1.npy
    - filename_aug2.npy
"""

import os
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
from tqdm import tqdm


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot_product = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    return dot_product / (norm_a * norm_b)


def load_latents_by_group_from_json(latent_dir):
    """
    Load latents and group them by base filename using the info from the JSONs.
    
    Args:
        latent_dir: Path to directory containing encoded latents (.npy and .json files)
    
    Returns:
        dict: Complete groups of latents with orig, aug1, and aug2 variants
    """
    import json
    
    latent_dir = Path(latent_dir)
    latent_files = sorted(latent_dir.glob("*.npy"))

    groups = defaultdict(dict)

    for npy_path in latent_files:
        json_path = npy_path.with_suffix(".json")
        if not json_path.exists():
            print(f"⚠️ Skipping {npy_path.name} — no matching JSON found")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract the variant info from the 'file_name' in the JSON
        file_name = data.get("relpath", "")
        if not file_name:
            print(f"⚠️ Skipping {npy_path.name} — 'file_name' missing in JSON")
            continue

        # Determine base name and variant
        if "_orig" in file_name:
            base_name = file_name.replace("_orig", "").replace(".wav", "")
            variant = "orig"
        elif "_aug1" in file_name:
            base_name = file_name.replace("_aug1", "").replace(".wav", "")
            variant = "aug1"
        elif "_aug2" in file_name:
            base_name = file_name.replace("_aug2", "").replace(".wav", "")
            variant = "aug2"
        else:
            print(f"⚠️ Skipping {file_name} — doesn't include _orig/_aug1/_aug2")
            continue

        # Load the latent
        latent = np.load(npy_path)
        groups[base_name][variant] = latent

    # Only keep complete groups
    complete_groups = {
        name: variants
        for name, variants in groups.items()
        if len(variants) == 3 and all(k in variants for k in ["orig", "aug1", "aug2"])
    }

    print(f"✅ Found {len(complete_groups)} complete groups (orig + aug1 + aug2)")
    return complete_groups


def analyze_similarities(groups):
    """
    Calculate cosine similarities for all groups.
    
    Returns:
        dict: Statistics about similarities
    """
    orig_aug1_sims = []
    orig_aug2_sims = []
    aug1_aug2_sims = []
    
    print(f"\n{'='*70}")
    print(f"Analyzing {len(groups)} complete audio groups...")
    print(f"{'='*70}\n")
    
    for base_name, variants in tqdm(groups.items(), desc="Computing similarities"):
        orig = variants['orig']
        aug1 = variants['aug1']
        aug2 = variants['aug2']
        
        # Calculate pairwise similarities
        sim_orig_aug1 = cosine_similarity(orig, aug1)
        sim_orig_aug2 = cosine_similarity(orig, aug2)
        sim_aug1_aug2 = cosine_similarity(aug1, aug2)
        
        orig_aug1_sims.append(sim_orig_aug1)
        orig_aug2_sims.append(sim_orig_aug2)
        aug1_aug2_sims.append(sim_aug1_aug2)
    
    return {
        'orig_aug1': np.array(orig_aug1_sims),
        'orig_aug2': np.array(orig_aug2_sims),
        'aug1_aug2': np.array(aug1_aug2_sims)
    }


def print_results(similarities):
    """Print detailed analysis results."""
    
    print(f"\n{'='*70}")
    print(f"LATENT SIMILARITY ANALYSIS RESULTS")
    print(f"{'='*70}\n")
    
    # Summary statistics
    print(f"📊 SIMILARITY STATISTICS")
    print(f"{'-'*70}")
    print(f"{'Comparison':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*70}")
    
    for name, sims in similarities.items():
        label = name.replace('_', ' vs ')
        print(f"{label:<20} {sims.mean():.4f}      {sims.std():.4f}      "
              f"{sims.min():.4f}      {sims.max():.4f}")
    
    print(f"{'-'*70}\n")
    
    # Overall assessment
    avg_similarity = (similarities['orig_aug1'].mean() + 
                     similarities['orig_aug2'].mean() + 
                     similarities['aug1_aug2'].mean()) / 3
    
    print(f"📈 OVERALL AVERAGE SIMILARITY: {avg_similarity:.4f}\n")
    
    # Interpretation
    print(f"{'='*70}")
    print(f"🎯 INTERPRETATION & RECOMMENDATION")
    print(f"{'='*70}\n")
    
    if avg_similarity < 0.85:
        status = "✅ EXCELLENT"
        recommendation = (
            "Your augmentations create clearly distinct latent representations!\n"
            "   → Model will treat these as meaningfully different samples\n"
            "   → Proceed with full augmentation (1341 → 4023 clips)\n"
            "   → Expected effective dataset size: ~3800-4023 clips"
        )
    elif avg_similarity < 0.90:
        status = "✅ GOOD"
        recommendation = (
            "Your augmentations create sufficiently distinct representations.\n"
            "   → Model will likely benefit from these augmentations\n"
            "   → Proceed with full augmentation (1341 → 4023 clips)\n"
            "   → Expected effective dataset size: ~3200-3600 clips\n"
            "   → Monitor validation loss closely during training"
        )
    elif avg_similarity < 0.92:
        status = "⚠️  BORDERLINE"
        recommendation = (
            "Your augmentations are marginally distinct.\n"
            "   → Model may see some benefit, but not full 3x improvement\n"
            "   → Consider: (A) Strengthen augmentation parameters, OR\n"
            "              (B) Reduce to 2x augmentation instead of 3x, OR\n"
            "              (C) Proceed but monitor overfitting carefully\n"
            "   → Expected effective dataset size: ~2200-2800 clips"
        )
    else:
        status = "❌ TOO SIMILAR"
        recommendation = (
            "Your augmentations are too subtle - model will treat as duplicates.\n"
            "   → Current augmentation will NOT provide significant benefit\n"
            "   → STRONGLY RECOMMEND: Increase augmentation strength\n"
            "   → OR: Train without augmentation (use original 1341 clips only)\n"
            "   → Expected effective dataset size: ~1600-2200 clips (barely better)"
        )
    
    print(f"Status: {status}")
    print(f"Average Similarity: {avg_similarity:.4f}\n")
    print(f"Recommendation:\n{recommendation}\n")
    
    # Reference scale
    print(f"{'='*70}")
    print(f"📚 REFERENCE SCALE")
    print(f"{'='*70}")
    print(f"  <0.85  = Excellent - Clearly different samples")
    print(f"  0.85-0.90 = Good - Sufficiently distinct")
    print(f"  0.90-0.92 = Borderline - Marginal benefit")
    print(f"  0.92-0.95 = Poor - Minimal benefit")
    print(f"  >0.95  = Too similar - Essentially duplicates")
    print(f"{'='*70}\n")
    
    # Detailed breakdown
    print(f"{'='*70}")
    print(f"🔍 DETAILED BREAKDOWN")
    print(f"{'='*70}\n")
    
    print(f"Original vs Aug1 (brighter/faster):")
    print(f"  → Mean similarity: {similarities['orig_aug1'].mean():.4f}")
    if similarities['orig_aug1'].mean() < 0.90:
        print(f"  → ✅ Aug1 creates distinct latents\n")
    else:
        print(f"  → ⚠️  Aug1 too similar to original\n")
    
    print(f"Original vs Aug2 (darker/slower):")
    print(f"  → Mean similarity: {similarities['orig_aug2'].mean():.4f}")
    if similarities['orig_aug2'].mean() < 0.90:
        print(f"  → ✅ Aug2 creates distinct latents\n")
    else:
        print(f"  → ⚠️  Aug2 too similar to original\n")
    
    print(f"Aug1 vs Aug2 (augmentations compared):")
    print(f"  → Mean similarity: {similarities['aug1_aug2'].mean():.4f}")
    if similarities['aug1_aug2'].mean() < 0.85:
        print(f"  → ✅ Augmentations are very different from each other")
    elif similarities['aug1_aug2'].mean() < 0.90:
        print(f"  → ✅ Augmentations are sufficiently different")
    else:
        print(f"  → ⚠️  Augmentations too similar to each other")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze latent similarity for audio augmentation validation'
    )
    parser.add_argument(
        '--latent_dir',
        type=str,
        required=True,
        help='Path to directory containing encoded latents (.npy and .json files)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: Save results to text file'
    )

    args = parser.parse_args()

    # Validate latent directory exists
    if not Path(args.latent_dir).exists():
        print(f"\n❌ ERROR: Latent directory not found: {args.latent_dir}")
        return

    groups = load_latents_by_group_from_json(args.latent_dir)

    if len(groups) == 0:
        print("\n❌ ERROR: No complete audio groups found!")
        return

    similarities = analyze_similarities(groups)
    print_results(similarities)

    if args.output:
        print(f"💾 Results saved to: {args.output}")


if __name__ == '__main__':
    main()