#!/usr/bin/env python3

from src.config import CURATED_MODELS, DEFAULT_MODEL

print('=== CURATED MODELS ===')
for model_id, metadata in CURATED_MODELS.items():
    print(f'{model_id}: {metadata.tier} - ${metadata.pricing.input_per_1m:.2f}/${metadata.pricing.output_per_1m:.2f} per 1M tokens')

print(f'\nDefault model: {DEFAULT_MODEL}')
print(f'Total models: {len(CURATED_MODELS)}')

# Test tier distribution
tiers = {}
for model_id, metadata in CURATED_MODELS.items():
    tiers[metadata.tier] = tiers.get(metadata.tier, 0) + 1

print(f'\nTier distribution:')
for tier, count in tiers.items():
    print(f'  {tier}: {count} models')

# Test cost range
costs = [metadata.pricing.input_per_1m for metadata in CURATED_MODELS.values()]
print(f'\nCost range: ${min(costs):.2f} - ${max(costs):.2f} per 1M input tokens')