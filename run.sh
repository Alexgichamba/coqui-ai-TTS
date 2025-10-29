#!/bin/bash
# Script to launch training sessions in separate tmux sessions

# ============================================================================
# Session 1: Swahili Multi-Speaker Training
# ============================================================================
SESSION_NAME_1="swahili"

# Check if session already exists
tmux has-session -t "$SESSION_NAME_1" 2>/dev/null

if [ $? != 0 ]; then
    echo "Creating tmux session: $SESSION_NAME_1"
    tmux new-session -d -s "$SESSION_NAME_1"
    
    # Activate conda environment
    tmux send-keys -t "$SESSION_NAME_1" "conda activate coqui" C-m
    
    # Wait a moment for activation to complete
    sleep 1
    
    # Send the training command to the session
    tmux send-keys -t "$SESSION_NAME_1" "CUDA_VISIBLE_DEVICES=0 python train_afro_vits.py \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "    --dataset-path /home/alex/2k_dataset_resampled \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "    --metadata-file /home/alex/29_sep_index.csv \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "    --language swahili \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "    --speaker-ids \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "        d60d0508-18f2-44ad-8826-48dac52f5fe3 \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "        8694dcef-21e3-455b-b98f-c0baadf2b015 \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "        c3e169c0-4944-446e-bcb5-cb71c6b6bb4d \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "        d0c3f159-b268-475d-9607-8e77988d3a25 \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "    --output-path ./swahili_output \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "    --run-name swahili_vits_2f_2m \\" C-m
    tmux send-keys -t "$SESSION_NAME_1" "    --epochs 5291" C-m
    
    echo "✓ Started Swahili training in session: $SESSION_NAME_1"
else
    echo "! Session $SESSION_NAME_1 already exists. Skipping..."
fi

# ============================================================================
# Session 2: Yoruba Multi-Speaker Training
# ============================================================================
SESSION_NAME_2="yoruba"

# Check if session already exists
tmux has-session -t "$SESSION_NAME_2" 2>/dev/null

if [ $? != 0 ]; then
    echo "Creating tmux session: $SESSION_NAME_2"
    tmux new-session -d -s "$SESSION_NAME_2"
    
    # Activate conda environment
    tmux send-keys -t "$SESSION_NAME_2" "conda activate coqui" C-m
    
    # Wait a moment for activation to complete
    sleep 1
    
    # Send the training command to the session
    tmux send-keys -t "$SESSION_NAME_2" "CUDA_VISIBLE_DEVICES=1 python train_afro_vits.py \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "    --dataset-path /home/alex/2k_dataset_resampled \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "    --metadata-file /home/alex/29_sep_index.csv \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "    --language yoruba \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "    --speaker-ids \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "        O0GW8 \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "        HNSNF \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "        HPTGN \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "        ZLOOP \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "    --output-path ./yoruba_output \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "    --run-name yoruba_vits_2f_2m \\" C-m
    tmux send-keys -t "$SESSION_NAME_2" "    --epochs 1567" C-m
    
    echo "✓ Started Yoruba training in session: $SESSION_NAME_2"
else
    echo "! Session $SESSION_NAME_2 already exists. Skipping..."
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================"
echo "Training Sessions Started"
echo "================================"
echo "To attach to a session, use:"
echo "  tmux attach -t $SESSION_NAME_1"
echo "  tmux attach -t $SESSION_NAME_2"
echo ""
echo "To list all sessions:"
echo "  tmux ls"
echo ""
echo "To detach from a session: Ctrl+b then d"
echo "================================"