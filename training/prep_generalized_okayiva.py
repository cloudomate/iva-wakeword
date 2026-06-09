import os
os.chdir("/Users/yashwant.singh/coderepo/micro-wake-word")
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from mmap_ninja.ragged import RaggedMmap
import yaml

# Feature-extract the device-recorded HARD NEGATIVES (near-miss phrases +
# ambient that crosses the trigger). Same augmentation recipe as the positive
# prep so the negatives are seen under matching room/noise conditions.
print("[gen] augmented features from recorded hard negatives...", flush=True)
clips = Clips(input_directory='neg_hard_okayiva', file_pattern='*.wav', max_clip_duration_s=None,
              remove_silence=False, random_split_seed=10, split_count=0.15)
augmenter = Augmentation(augmentation_duration_s=3.2,
    augmentation_probabilities={"SevenBandParametricEQ":0.25,"TanhDistortion":0.25,"PitchShift":0.25,
        "BandStopFilter":0.25,"AddColorNoise":0.25,"AddBackgroundNoise":0.75,"Gain":1.0,"RIR":0.5},
    impulse_paths=['mit_rirs'], background_paths=['fma_16k'],
    background_min_snr_db=-5, background_max_snr_db=10, min_jitter_s=0.195, max_jitter_s=0.205)
OUT = "generated_augmented_features_neg_hard"
os.makedirs(OUT, exist_ok=True)
for split, sn, rep, sf in [("training","train",40,10),("validation","validation",5,10),("testing","test",5,1)]:
    o = f"{OUT}/{split}"; os.makedirs(o, exist_ok=True)
    spg = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=sf, step_ms=10)
    RaggedMmap.from_generator(out_dir=os.path.join(o, "wakeword_mmap"),
        sample_generator=spg.spectrogram_generator(split=sn, repeat=rep), batch_size=100, verbose=True)

# Second REAL voice (female) -> better real-speaker generalization ("other people").
print("[gen] augmented features from female-voice positives...", flush=True)
clips_w = Clips(input_directory='real_samples_female', file_pattern='*.wav', max_clip_duration_s=None,
                remove_silence=False, random_split_seed=10, split_count=0.15)
OUTW = "generated_augmented_features_female"
os.makedirs(OUTW, exist_ok=True)
for split, sn, rep, sf in [("training","train",40,10),("validation","validation",5,10),("testing","test",5,1)]:
    o = f"{OUTW}/{split}"; os.makedirs(o, exist_ok=True)
    spg = SpectrogramGeneration(clips=clips_w, augmenter=augmenter, slide_frames=sf, step_ms=10)
    RaggedMmap.from_generator(out_dir=os.path.join(o, "wakeword_mmap"),
        sample_generator=spg.spectrogram_generator(split=sn, repeat=rep), batch_size=100, verbose=True)

# Third REAL voice (child) -> covers higher-pitched / kid speakers.
print("[gen] augmented features from child-voice positives...", flush=True)
clips_k = Clips(input_directory='real_samples_kid', file_pattern='*.wav', max_clip_duration_s=None,
                remove_silence=False, random_split_seed=10, split_count=0.15)
OUTK = "generated_augmented_features_kid"
os.makedirs(OUTK, exist_ok=True)
for split, sn, rep, sf in [("training","train",40,10),("validation","validation",5,10),("testing","test",5,1)]:
    o = f"{OUTK}/{split}"; os.makedirs(o, exist_ok=True)
    spg = SpectrogramGeneration(clips=clips_k, augmenter=augmenter, slide_frames=sf, step_ms=10)
    RaggedMmap.from_generator(out_dir=os.path.join(o, "wakeword_mmap"),
        sample_generator=spg.spectrogram_generator(split=sn, repeat=rep), batch_size=100, verbose=True)

# GENERALIZED config: synthetic-heavy positives (works for any speaker), the
# user's real voice DE-EMPHASIZED (weight 1.0, was 4.0 -> fixes overfit), the
# recorded hard negatives added at high weight (fixes misfires), and spec
# augmentation turned ON (was off -> better generalization/robustness).
config = {"window_step_ms":10, "train_dir":"trained_models/wakeword_generalized_okayiva", "features":[
  {"features_dir":"generated_augmented_features_okayeeva","sampling_weight":3.0,"penalty_weight":1.0,"truth":True,"truncation_strategy":"truncate_start","type":"mmap"},
  {"features_dir":"generated_augmented_features_okayiva","sampling_weight":2.0,"penalty_weight":1.0,"truth":True,"truncation_strategy":"truncate_start","type":"mmap"},
  {"features_dir":"generated_augmented_features_real_okayiva","sampling_weight":1.0,"penalty_weight":1.0,"truth":True,"truncation_strategy":"truncate_start","type":"mmap"},
  {"features_dir":"generated_augmented_features_female","sampling_weight":2.0,"penalty_weight":1.0,"truth":True,"truncation_strategy":"truncate_start","type":"mmap"},
  {"features_dir":"generated_augmented_features_kid","sampling_weight":1.0,"penalty_weight":1.0,"truth":True,"truncation_strategy":"truncate_start","type":"mmap"},
  {"features_dir":"negative_datasets/speech","sampling_weight":10.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
  {"features_dir":"negative_datasets/dinner_party","sampling_weight":10.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
  {"features_dir":"negative_datasets/no_speech","sampling_weight":5.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
  {"features_dir":OUT,"sampling_weight":20.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
  {"features_dir":"negative_datasets/dinner_party_eval","sampling_weight":0.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"split","type":"mmap"}],
 "training_steps":[12000], "positive_class_weight":[1], "negative_class_weight":[25], "learning_rates":[0.001], "batch_size":128,
 "time_mask_max_size":[10], "time_mask_count":[1], "freq_mask_max_size":[5], "freq_mask_count":[1],
 "eval_step_interval":500, "clip_duration_ms":1500, "target_minimization":0.9, "minimization_metric":None, "maximization_metric":"average_viable_recall"}
yaml.dump(config, open("training_parameters_generalized_okayiva.yaml","w"))
print("[gen] DONE features+config -> training_parameters_generalized_okayiva.yaml", flush=True)
