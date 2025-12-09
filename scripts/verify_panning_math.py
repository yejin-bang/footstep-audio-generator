#!/usr/bin/env python3
"""
Verify constant power panning calculations for different max_pan_percentage values
"""
import math

def calculate_pan_gains(x_position, max_pan_percentage):
    """Calculate left/right gains for a given x_position and max_pan_percentage"""
    # Clamp x_position
    x_position = max(0.0, min(1.0, x_position))

    # Scale x_position to limited pan range
    center = 0.5
    scaled_x = center + (x_position - center) * max_pan_percentage

    # Convert to pan angle
    pan_angle = (scaled_x - 0.5) * math.pi

    # Constant power panning
    left_gain = math.cos(pan_angle * 0.5 + math.pi / 4)
    right_gain = math.sin(pan_angle * 0.5 + math.pi / 4)

    return left_gain, right_gain

# Test scenarios
test_cases = [
    ("Far Left (x=0.0)", 0.0),
    ("Slight Left (x=0.3)", 0.3),
    ("Center (x=0.5)", 0.5),
    ("Slight Right (x=0.7)", 0.7),
    ("Far Right (x=1.0)", 1.0),
]

max_pan_values = [
    ("MODERATE (0.4)", 0.4),
    ("SUBTLE (0.2)", 0.2),
]

print("=" * 80)
print("CONSTANT POWER PANNING COMPARISON")
print("=" * 80)
print("\nComparing how aggressive the panning feels at different positions")
print("Note: 'Aggressive' = large difference between L/R channels")
print("      'Subtle' = small difference between L/R channels\n")

for max_pan_label, max_pan in max_pan_values:
    print(f"\n{'='*80}")
    print(f"{max_pan_label} panning")
    print('='*80)

    for label, x_pos in test_cases:
        left, right = calculate_pan_gains(x_pos, max_pan)

        # Calculate metrics
        diff = abs(left - right)
        ratio = left / right if right > 0 else float('inf')

        # Check constant power (should always be ~1.0)
        power = left**2 + right**2

        print(f"\n{label:20s} (x={x_pos:.1f})")
        print(f"  Left:  {left:.3f} ({left*100:.1f}%)")
        print(f"  Right: {right:.3f} ({right*100:.1f}%)")
        print(f"  Difference: {diff:.3f} ({diff*100:.1f} percentage points)")
        print(f"  L/R Ratio: {ratio:.2f}:1")
        print(f"  Power check: {power:.3f} (should be ~1.0)")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("""
How to read these numbers:

1. DIFFERENCE: Smaller = more subtle, Larger = more aggressive
   - 0.2 (subtle) has SMALLER differences than 0.4 (moderate)

2. RATIO: Closer to 1.0 = more subtle, Higher = more aggressive
   - 0.2 (subtle) has ratios CLOSER to 1.0 than 0.4 (moderate)

3. AT CENTER (x=0.5): Both should be 70.7% (-3dB pan law)
   - This is IDENTICAL for all max_pan_percentage values

4. AT EXTREMES:
   - 0.2 stays closer to center (more subtle)
   - 0.4 moves farther from center (more aggressive)

CONCLUSION: 0.2 is definitely MORE SUBTLE than 0.4! ✓
""")
