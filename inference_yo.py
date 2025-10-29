import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor

print("="*80)
print("VITS MODEL PHONEMIZATION TEST - YORUBA")
print("="*80)

# 1. Load the configuration
print("\n[1/6] Loading configuration...")
config_path = "yoruba_output/yoruba_vits_2f_2m-October-28-2025_08+39PM-507c039d/config.json"
config = VitsConfig()
config.load_json(config_path)
print(f"✓ Config loaded from: {config_path}")
print(f"  - use_phonemes: {config.use_phonemes}")
print(f"  - phonemizer: {config.phonemizer}")
print(f"  - phoneme_language: {config.phoneme_language}")
print(f"  - text_cleaner: {config.text_cleaner}")

# 2. Initialize audio processor
print("\n[2/6] Initializing audio processor...")
ap = AudioProcessor.init_from_config(config)
print(f"✓ Audio processor initialized")
print(f"  - sample_rate: {ap.sample_rate}")
print(f"  - hop_length: {ap.hop_length}")

# 3. Load the model
print("\n[3/6] Initializing and loading model...")
model = Vits.init_from_config(config)
checkpoint_path = "yoruba_output/yoruba_vits_2f_2m-October-28-2025_08+39PM-507c039d/best_model.pth"
model.load_checkpoint(config, checkpoint_path, eval=True)
print(f"✓ Model loaded from: {checkpoint_path}")

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"✓ Model moved to: {device}")

# 4. Check speaker information
print("\n[4/6] Checking speaker information...")
if model.speaker_manager:
    print(f"✓ Number of speakers: {model.speaker_manager.num_speakers}")
    print(f"  Available speakers: {list(model.speaker_manager.name_to_id.keys())}")
else:
    print("✗ No speaker manager found (single-speaker model?)")

# 5. Test phonemization pipeline
print("\n[5/6] Testing phonemization pipeline...")
print("-"*80)

text = "Báwo ni ìwọ ṣe wà?"
print(f"Original text: '{text}'")
print()

# Step 1: Text cleaning
try:
    if model.tokenizer.text_cleaner is not None:
        cleaned_text = model.tokenizer.text_cleaner(text)
        print(f"✓ Cleaned text: '{cleaned_text}'")
    else:
        cleaned_text = text
        print(f"⚠ No text cleaner configured, using original text")
except Exception as e:
    print(f"✗ Error during text cleaning: {e}")
    cleaned_text = text

print()

# Step 2: Phonemization
try:
    if model.tokenizer.use_phonemes and model.tokenizer.phonemizer:
        # Test with pipe separator for visibility
        phonemes_visible = model.tokenizer.phonemizer.phonemize(
            cleaned_text, 
            separator="|", 
            language="yor-Latn"
        )
        print(f"✓ Phonemes (pipe-separated): '{phonemes_visible}'")
        
        # Test without separator (as used in training)
        phonemes_no_sep = model.tokenizer.phonemizer.phonemize(
            cleaned_text, 
            separator="", 
            language="yor-Latn"
        )
        print(f"✓ Phonemes (no separator): '{phonemes_no_sep}'")
        print(f"  Length: {len(phonemes_no_sep)} characters")
    else:
        print("⚠ Phonemization not enabled or phonemizer not configured")
        phonemes_no_sep = cleaned_text
except Exception as e:
    print(f"✗ Error during phonemization: {e}")
    import traceback
    traceback.print_exc()
    phonemes_no_sep = cleaned_text

print()

# Step 3: Get token IDs
try:
    token_ids = model.tokenizer.text_to_ids(text, language="yor-Latn")
    print(f"✓ Token IDs: {token_ids}")
    print(f"  Number of tokens: {len(token_ids)}")
    
    # Show character vocabulary size
    print(f"  Vocabulary size: {len(model.tokenizer.characters.vocab)}")
    
    # Check for special tokens
    if model.tokenizer.add_blank:
        print(f"  Blank ID: {model.tokenizer.blank_id}")
    if model.tokenizer.pad_id is not None:
        print(f"  Pad ID: {model.tokenizer.pad_id}")
        
except Exception as e:
    print(f"✗ Error during tokenization: {e}")
    import traceback
    traceback.print_exc()
    token_ids = None

print()

# Step 4: Decode token IDs back to text
if token_ids:
    try:
        # Remove blank tokens if present for readability
        if model.tokenizer.add_blank and model.tokenizer.blank_id is not None:
            token_ids_no_blank = [tid for tid in token_ids if tid != model.tokenizer.blank_id]
            decoded_text = model.tokenizer.ids_to_text(token_ids_no_blank)
            print(f"✓ Decoded text (blanks removed): '{decoded_text}'")
        else:
            decoded_text = model.tokenizer.ids_to_text(token_ids)
            print(f"✓ Decoded text: '{decoded_text}'")
    except Exception as e:
        print(f"✗ Error during decoding: {e}")

print("-"*80)

# 6. Test full inference
print("\n[6/6] Testing full inference...")
print("-"*80)

try:
    # Select a speaker from Yoruba training
    # Your Yoruba speakers: O0GW8, HNSNF, HPTGN, ZLOOP
    speaker_name = "O0GW8"
    
    if model.speaker_manager and speaker_name in model.speaker_manager.name_to_id:
        speaker_id = model.speaker_manager.name_to_id[speaker_name]
        print(f"✓ Using speaker: {speaker_name} (ID: {speaker_id})")
    else:
        print(f"⚠ Speaker not found, trying first available speaker...")
        if model.speaker_manager and model.speaker_manager.num_speakers > 0:
            speaker_name = list(model.speaker_manager.name_to_id.keys())[0]
            speaker_id = model.speaker_manager.name_to_id[speaker_name]
            print(f"✓ Using speaker: {speaker_name} (ID: {speaker_id})")
        else:
            speaker_id = None
            print(f"⚠ No speakers available (single-speaker model)")
    
    print()
    
    # Prepare inputs
    token_ids_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    aux_input = {
        "x_lengths": torch.LongTensor([len(token_ids)]).to(device),
    }
    
    if speaker_id is not None:
        aux_input["speaker_ids"] = torch.LongTensor([speaker_id]).to(device)
    
    print(f"Input tensor shape: {token_ids_tensor.shape}")
    print(f"Generating audio...")
    
    # Generate
    with torch.no_grad():
        outputs = model.inference(token_ids_tensor, aux_input)
    
    # Get waveform
    wav = outputs["model_outputs"].squeeze().cpu().numpy()
    
    print(f"✓ Audio generated successfully!")
    print(f"  Shape: {wav.shape}")
    print(f"  Duration: {len(wav) / ap.sample_rate:.2f} seconds")
    print(f"  Sample rate: {ap.sample_rate} Hz")
    
    # Save the audio
    output_path = "test_yoruba_phonemization_output.wav"
    ap.save_wav(wav, output_path)
    print(f"✓ Audio saved to: {output_path}")
    
except Exception as e:
    print(f"✗ Error during inference: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("TEST COMPLETE")
print("="*80)

# Optional: Test with multiple sentences
print("\n[BONUS] Testing with multiple Yoruba sentences...")
print("-"*80)

# From your training config test sentences
test_sentences = [
    "Báwo ni ìwọ ṣe wà?",
    "Mo nífẹ̀ẹ́ rẹ púpọ̀.",
    "Ṣé o lè sọ fún mi nípa ara rẹ?",
    "Ẹ jọ̀ọ́, ẹ jẹ́ kí n mọ̀ọ́ diẹ̀ síi.",
]

for i, test_text in enumerate(test_sentences, 1):
    print(f"\n[{i}/{len(test_sentences)}] Text: '{test_text}'")
    try:
        # Get phonemes
        if model.tokenizer.use_phonemes:
            phonemes = model.tokenizer.phonemizer.phonemize(
                model.tokenizer.text_cleaner(test_text) if model.tokenizer.text_cleaner else test_text,
                separator="|",
                language="yor-Latn"
            )
            print(f"  Phonemes: '{phonemes}'")
        
        # Get token IDs
        tids = model.tokenizer.text_to_ids(test_text, language="yor-Latn")
        print(f"  Tokens: {len(tids)} tokens")
        
        # Quick inference
        tids_tensor = torch.LongTensor(tids).unsqueeze(0).to(device)
        aux_in = {"x_lengths": torch.LongTensor([len(tids)]).to(device)}
        if speaker_id is not None:
            aux_in["speaker_ids"] = torch.LongTensor([speaker_id]).to(device)
        
        with torch.no_grad():
            out = model.inference(tids_tensor, aux_in)
        
        wav_out = out["model_outputs"].squeeze().cpu().numpy()
        duration = len(wav_out) / ap.sample_rate
        print(f"  ✓ Generated: {duration:.2f}s audio")
        
        # Save
        save_path = f"test_yoruba_sentence_{i}.wav"
        ap.save_wav(wav_out, save_path)
        print(f"  ✓ Saved to: {save_path}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*80)
print("ALL YORUBA TESTS COMPLETE!")
print("="*80)

print("\n" + "="*80)
print("YORUBA TONE MARKS TEST")
print("="*80)
print("\nYoruba has 3 tone marks that are CRUCIAL for pronunciation:")
print("  - Acute (´): High tone - e.g., á, é, ó")
print("  - Grave (`): Low tone - e.g., à, è, ò")
print("  - Macron (¯): Mid tone - e.g., ā, ē, ō")
print("\nLet's verify the phonemizer preserves these tones:")
print("-"*80)

tone_test_words = [
    ("bá", "father (high tone)"),
    ("bà", "to meet (low tone)"),
    ("ba", "to hide (mid tone)"),
    ("ọ́kọ̀", "husband"),
    ("òkò", "hoe/vehicle"),
    ("owó", "money"),
    ("ọwọ́", "hand"),
]

for word, meaning in tone_test_words:
    print(f"\nWord: '{word}' ({meaning})")
    if model.tokenizer.use_phonemes:
        phonemes = model.tokenizer.phonemizer.phonemize(
            word.lower(),
            separator="|",
            language="yor-Latn"
        )
        print(f"  Phonemes: '{phonemes}'")
        print(f"  Tone preserved: {'✓' if any(c in phonemes for c in 'àáèéìíòóùú') else '✗'}")

print("\n" + "="*80)