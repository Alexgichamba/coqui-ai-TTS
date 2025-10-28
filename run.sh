#!/bin/bash
# Example training scripts for different languages and configurations

# ============================================================================
# Example 1: Basic Swahili Multi-Speaker Training
# ============================================================================
# d60d0508-18f2-44ad-8826-48dac52f5fe3 - male speaker
# 8694dcef-21e3-455b-b98f-c0baadf2b015 - female speaker
# c3e169c0-4944-446e-bcb5-cb71c6b6bb4d - female speaker
# d0c3f159-b268-475d-9607-8e77988d3a25 - male speaker
CUDA_VISIBLE_DEVICES=0 python train_afro_vits.py \
    --dataset-path /home/alex/2k_dataset_resampled \
    --metadata-file /home/alex/29_sep_index.csv \
    --language swahili \
    --speaker-ids \
        d60d0508-18f2-44ad-8826-48dac52f5fe3 \
        8694dcef-21e3-455b-b98f-c0baadf2b015 \
        c3e169c0-4944-446e-bcb5-cb71c6b6bb4d \
        d0c3f159-b268-475d-9607-8e77988d3a25 \
    --output-path ./swahili_output \
    --run-name swahili_vits_2f_2m

# # ============================================================================
# # Example 2: Basic Yoruba Multi-Speaker Training
# # ============================================================================
# CUDA_VISIBLE_DEVICES=1 python train_afro_vits.py \
#     --dataset-path /home/alex/2k_dataset_resampled \
#     --metadata-file /home/alex/29_sep_index.csv \
#     --language yoruba \
#     --speaker-ids \
#         d0c3f159-b268-475d-9607-8e77988d3a25 \
#         a60c700e-9596-4b3c-8a72-3259d963c4e5 \
#     --output-path ./yoruba_output \
#     --run-name yoruba_vits_2f_2m