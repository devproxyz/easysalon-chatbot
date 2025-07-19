# from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
# from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
# import torch
# import nltk
# import numpy as np
# import sounddevice as sd
# models = None
# cfg = None
# task = None
# generator = None
# nltk.download('averaged_perceptron_tagger_eng')
# def play_audio_tensor(wav, rate):
#     """Play audio tensor using sounddevice."""
#     try:
#         # Convert tensor to numpy array if needed
#         if isinstance(wav, torch.Tensor):
#             wav = wav.cpu().numpy()
#         elif not isinstance(wav, np.ndarray):
#             raise ValueError("Input waveform must be a torch.Tensor or numpy.ndarray")

#         # Ensure waveform is 1D for mono audio
#         if wav.ndim > 1:
#             wav = wav.squeeze()

#         # Play audio
#         sd.play(wav, samplerate=rate)
#         sd.wait()  # Wait until playback is finished
#         print("Audio playback completed")
#     except Exception as e:
#         print(f"Error playing audio: {str(e)}")

# def generate_tts(text):
#     try:
#         # This line of code is necessary to ensure that the NLTK data files are downloaded
#         # Step 1: Clone and load the pre-trained TTS model from Hugging Face
#         # Using facebook/fastspeech2-en-ljspeech as specified
#         global models
#         global cfg
#         global task
#         global generator
#         if models is None or cfg is None or task is not None:
#             models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
#                 "facebook/fastspeech2-en-ljspeech",
#                 arg_overrides={"vocoder": "hifigan", "fp16": False}
#             )
#         model = models[0]
#         # Step 2: Update configuration and build generator
#         TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
#         if generator is None:
#             generator = task.build_generator([model], cfg)
#         print(f"Input text: {text}")

#         # Step 4: Prepare model input and generate audio
#         sample = TTSHubInterface.get_model_input(task, text)
#         wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
#         return wav, rate

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         return None, None
#         # Challenges faced:
#         # 1. Ensuring compatible versions of fairseq, torchaudio, and transformers
#         # 2. Handling potential memory issues with large models
#         # 3. Verifying correct vocoder configuration (hifigan)
#         # 4. Managing audio output format compatibility

# def play_tts(text):
#     wav, rate  = generate_tts(text)
#     if wav is not None and rate is not None:
#         play_audio_tensor(wav, rate)
#     else:
#         print(f"Can not play {text}")