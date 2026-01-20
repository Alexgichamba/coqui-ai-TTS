#!/usr/bin/env python3
"""
Multi-language VITS TTS Training Script
Supports training VITS models for multiple languages with command-line configuration.
"""
import os
import argparse
import json
from pathlib import Path
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


# Language-specific defaults
LANGUAGE_CONFIGS = {
    "swahili": {
        "lang_code": "swa-Latn",
        "text_cleaner": "swahili_cleaners",
        "phonemizer": "epitran",
        "test_sentences": [
            "Habari yako, wewe ni mzuri.",
            "Ninakupenda sana.",
            "Waambaje?",
            "Tafadhali niambie zaidi kuhusu wewe.",
        ]
    },
    "yoruba": {
        "lang_code": "yor-Latn",
        "text_cleaner": "yoruba_cleaners",
        "phonemizer": "epitran",
        "test_sentences": [
            "Báwo ni ìwọ ṣe wà?",
            "Mo nífẹ̀ẹ́ rẹ púpọ̀.",
            "Ṣé o lè sọ fún mi nípa ara rẹ?",
            "Ẹ jọ̀ọ́, ẹ jẹ́ kí n mọ̀ọ́ diẹ̀ síi.",
        ]
    },
    "kinyarwanda": {
        "lang_code": "kin-Latn",
        "text_cleaner": "kinyarwanda_cleaners",
        "phonemizer": "epitran",
        "test_sentences": [
            "Amakuru yawe, urasa neza.",
            "Ndagukunda cyane.",
            "Wiriwe?",
            "Nyamuneka mbwira byinshi kuri wowe.",
        ]
    },
    "hausa": {
        "lang_code": "hau-Latn",
        "text_cleaner": "hausa_cleaners",
        "phonemizer": "epitran",
        "test_sentences": [
            "Lafiya lau, kana da kyau.",
            "Ina sonka sosai.",
            "Yaya ake ciki?",
            "Don Allah, ka gaya mini ƙari game da kai.",
        ]
    }
}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a multi-speaker VITS TTS model for any language",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory containing audio files"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        required=True,
        help="Path to the metadata CSV file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output directory for training artifacts (default: current directory)"
    )
    
    # Language arguments
    parser.add_argument(
        "--language",
        type=str,
        default="swahili",
        choices=list(LANGUAGE_CONFIGS.keys()) + ["custom"],
        help="Language to train on (use 'custom' for manual configuration)"
    )
    parser.add_argument(
        "--lang-code",
        type=str,
        default=None,
        help="Language code for phonemizer (required if --language=custom)"
    )
    parser.add_argument(
        "--text-cleaner",
        type=str,
        default=None,
        help="Text cleaner to use (required if --language=custom)"
    )
    parser.add_argument(
        "--phonemizer",
        type=str,
        default=None,
        choices=["epitran", "espeak", "gruut"],
        help="Phonemizer backend (required if --language=custom)"
    )
    parser.add_argument(
        "--test-sentences",
        type=str,
        nargs="+",
        default=None,
        help="Test sentences for evaluation (space-separated)"
    )
    
    # Speaker arguments
    parser.add_argument(
        "--speaker-ids",
        type=str,
        nargs="+",
        default=None,
        help="List of speaker IDs to include (space-separated)"
    )
    parser.add_argument(
        "--speaker-ids-file",
        type=str,
        default=None,
        help="JSON file containing list of speaker IDs"
    )
    parser.add_argument(
        "--language-filter",
        type=str,
        default=None,
        help="Filter dataset by language name in metadata"
    )
    
    # Dataset formatter arguments
    parser.add_argument(
        "--formatter",
        type=str,
        default="afro_formatter",
        help="Dataset formatter name"
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=15.0,
        help="Minimum SNR threshold for audio filtering"
    )
    
    # Audio configuration
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Audio sample rate"
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="Hop length for audio processing"
    )
    parser.add_argument(
        "--win-length",
        type=int,
        default=1024,
        help="Window length for audio processing"
    )
    parser.add_argument(
        "--num-mels",
        type=int,
        default=80,
        help="Number of mel filterbanks"
    )
    
    # Training arguments
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this training run (default: vits_{language}_multispeaker)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=48,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr-gen",
        type=float,
        default=2e-4,
        help="Generator learning rate"
    )
    parser.add_argument(
        "--lr-disc",
        type=float,
        default=2e-4,
        help="Discriminator learning rate"
    )
    parser.add_argument(
        "--T_0",
        type=int,
        default=100000,
        help="Number of iterations for the first restart in cosine annealing"
    )
    parser.add_argument(
        "--num-loader-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--max-audio-length",
        type=float,
        default=20.0,
        help="Maximum audio length in seconds"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_false",
        dest="mixed_precision",
        help="Disable mixed precision training"
    )
    
    # Phoneme arguments
    parser.add_argument(
        "--use-phonemes",
        action="store_true",
        default=True,
        help="Use phonemes for training"
    )
    parser.add_argument(
        "--no-phonemes",
        action="store_false",
        dest="use_phonemes",
        help="Disable phoneme usage"
    )
    
    # Resume training
    parser.add_argument(
        "--restore-path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    # Multi-speaker
    parser.add_argument(
        "--single-speaker",
        action="store_true",
        default=False,
        help="Train a single-speaker model instead of multi-speaker"
    )
    
    return parser.parse_args()


def load_speaker_ids(args):
    """Load speaker IDs from arguments or file."""
    if args.speaker_ids:
        return args.speaker_ids
    elif args.speaker_ids_file:
        with open(args.speaker_ids_file, 'r') as f:
            data = json.load(f)
            return data if isinstance(data, list) else data.get("speaker_ids", [])
    else:
        # No specific speaker filtering
        return None


def get_language_config(args):
    """Get language configuration based on arguments."""
    if args.language == "custom":
        # Validate custom configuration
        if not all([args.lang_code, args.text_cleaner, args.phonemizer]):
            raise ValueError(
                "When using --language=custom, you must specify "
                "--lang-code, --text-cleaner, and --phonemizer"
            )
        return {
            "lang_code": args.lang_code,
            "text_cleaner": args.text_cleaner,
            "phonemizer": args.phonemizer,
            "test_sentences": args.test_sentences or ["This is a test sentence."]
        }
    else:
        # Use predefined language configuration
        config = LANGUAGE_CONFIGS[args.language].copy()
        
        # Override with command-line arguments if provided
        if args.lang_code:
            config["lang_code"] = args.lang_code
        if args.text_cleaner:
            config["text_cleaner"] = args.text_cleaner
        if args.phonemizer:
            config["phonemizer"] = args.phonemizer
        if args.test_sentences:
            config["test_sentences"] = args.test_sentences
        
        return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup paths
    output_path = args.output_path or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_path, exist_ok=True)
    
    # Get language configuration
    lang_config = get_language_config(args)
    
    # Load speaker IDs
    speaker_ids = load_speaker_ids(args)
    
    # Set run name
    run_name = args.run_name or f"vits_{args.language}_{'single' if args.single_speaker else 'multi'}speaker"
    
    # Prepare dataset config kwargs
    dataset_kwargs = {
        # "language": args.language_filter or args.language,
        "min_snr": args.min_snr,
    }
    if speaker_ids:
        dataset_kwargs["speaker_ids"] = speaker_ids
    
    # Dataset configuration
    dataset_config = BaseDatasetConfig(
        formatter=args.formatter,
        meta_file_train=args.metadata_file,
        language=lang_config["lang_code"],
        path=args.dataset_path,
        **dataset_kwargs
    )
    
    # Audio configuration
    audio_config = VitsAudioConfig(
        sample_rate=args.sample_rate,
        win_length=args.win_length,
        hop_length=args.hop_length,
        num_mels=args.num_mels,
        mel_fmin=0,
        mel_fmax=None
    )
    
    # Model arguments
    vits_args = VitsArgs(
        use_speaker_embedding=not args.single_speaker,
        num_speakers=len(speaker_ids) if speaker_ids and not args.single_speaker else 0,
    )
    
    # Main config
    config = VitsConfig(
        model_args=vits_args,
        audio=audio_config,
        run_name=run_name,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        batch_group_size=5,
        num_loader_workers=args.num_loader_workers,
        num_eval_loader_workers=args.num_loader_workers,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        
        # Text processing configuration
        text_cleaner=lang_config["text_cleaner"],
        use_phonemes=args.use_phonemes,
        phoneme_language=lang_config["lang_code"],
        phonemizer=lang_config["phonemizer"],
        phoneme_cache_path=os.path.join(output_path, f"phoneme_cache_{args.language}"),
        compute_input_seq_cache=True,

        # ADD THESE FILTERING PARAMETERS:
        min_text_len=1,           # Minimum text length in characters
        max_text_len=325,         # Maximum text length in characters
        min_audio_len=int(0.2 * args.sample_rate),   # 0.2 seconds minimum (4,410 samples)
        max_audio_len=int(args.max_audio_length * args.sample_rate),  # 20 seconds maximum (441,000 samples)
        
        print_step=25,
        print_eval=False,
        mixed_precision=args.mixed_precision,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        
        # Training parameters
        lr_gen=args.lr_gen,
        lr_disc=args.lr_disc,
        
        # Generator Scheduler
        lr_scheduler_gen="CosineAnnealingWarmRestarts",
        lr_scheduler_gen_params={
            "T_0": 100000,      # Number of iterations for the first restart
            "T_mult": 1,       # A factor by which T_0 increases after a restart
            "eta_min": 1e-9    # Minimum learning rate
        },
        # Discriminator Scheduler
        lr_scheduler_disc="CosineAnnealingWarmRestarts",
        lr_scheduler_disc_params={
            "T_0": 100000,
            "T_mult": 1,
            "eta_min": 1e-9
        },
        # Crucial: Keep this False so it updates every step
        scheduler_after_epoch=False,
    )

    
    # Add test sentences - use simple string format
    config.test_sentences = lang_config["test_sentences"][:4]
    
    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)
    
    # Initialize tokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # Load data samples
    print(f"\n{'='*60}")
    print(f"Loading dataset from: {args.dataset_path}")
    print(f"Metadata file: {args.metadata_file}")
    print(f"{'='*60}\n")
    
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    # Initialize speaker manager if multi-speaker
    speaker_manager = None
    if not args.single_speaker:
        speaker_manager = SpeakerManager()
        speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
        config.model_args.num_speakers = speaker_manager.num_speakers
    
    # Print training info
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Language: {args.language}")
    print(f"Language code: {lang_config['lang_code']}")
    print(f"Phonemizer: {lang_config['phonemizer']}")
    print(f"Text cleaner: {lang_config['text_cleaner']}")
    print(f"Run name: {run_name}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")
    if speaker_manager:
        print(f"Number of speakers: {speaker_manager.num_speakers}")
        print(f"Speaker names: {speaker_manager.speaker_names[:10]}" + 
              (f"... (+{len(speaker_manager.speaker_names)-10} more)" 
               if len(speaker_manager.speaker_names) > 10 else ""))
    else:
        print(f"Mode: Single-speaker")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Output path: {output_path}")
    print(f"{'='*60}\n")
    
    # Initialize model
    model = Vits(config, ap, tokenizer, speaker_manager)
    
    # Initialize trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=args.restore_path,
            skip_train_epoch=False,
        ),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Start training
    print(f"Starting training...\n")
    trainer.fit()


if __name__ == "__main__":
    main()