"""
    app
    ~~~
    Streamlit app to run DDSP.
    Based of https://github.com/magenta/ddsp/blob/master/ddsp/colab/demos/timbre_transfer.ipynb (date 18/01/2020, commit bfd5209)
    and https://github.com/magenta/ddsp/blob/master/ddsp/colab/colab_utils.py (date 18/01/2020, commit 9748b50) 
"""
from __future__ import absolute_import, division, print_function

# Ignore a bunch of deprecation warnings
import warnings

warnings.filterwarnings("ignore")

import copy
import os
import time

import crepe
import ddsp
import ddsp.training

import gin
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import tensorflow.compat.v1 as tf

########################################################################
# Initializing things
########################################################################

tf.compat.v1.disable_v2_behavior()
TARGET = ""
CKPT_DIR = "models"

########################################################################
# Helper methods
########################################################################


def reset_crepe():
    """Reset the global state of CREPE to force model re-building."""
    for k in crepe.core.models:
        crepe.core.models[k] = None


@st.cache
def load_audio(wav_data):
    """Load audio into numpy array, cache it with Streamlit"""
    # TODO : should detect format to pass sample rate when enabling mp3, 16000 for mp3
    audio_np, unused_sr = librosa.core.load(uploaded_file, sr=44100)
    return audio_np, unused_sr


@st.cache
def compute_audio_features(audio_np):
    """Compute audio features, cache it with Streamlit"""
    audio_features = ddsp.training.eval_util.compute_audio_features(audio_np)
    audio_features_mod = None
    return audio_features, audio_features_mod


def specplot(
    audio_np, vmin=-5, vmax=1, rotate=True, size=512 + 256, sess=None, **matshow_kwargs
):
    """Plot the log magnitude spectrogram of audio_np."""
    # If batched, take first element.
    if len(audio_np.shape) == 2:
        audio_np = audio_np[0]

    logmag = ddsp.spectral_ops.compute_logmag(ddsp.core.tf_float32(audio_np), size=size)
    if sess is not None:
        logmag = sess.run(logmag)

    if rotate:
        logmag = np.rot90(logmag)
    # Plotting.
    plt.matshow(
        logmag, vmin=vmin, vmax=vmax, cmap=plt.cm.magma, aspect="auto", **matshow_kwargs
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Time")
    plt.ylabel("Frequency")


def shift_ld(audio_features, ld_shift=0.0):
    """Shift loudness by a number of ocatves."""
    audio_features["loudness_db"] += ld_shift
    return audio_features


def shift_f0(audio_features, f0_octave_shift=0.0):
    """Shift f0 by a number of ocatves."""
    audio_features["f0_hz"] *= 2.0 ** (f0_octave_shift)
    audio_features["f0_hz"] = np.clip(
        audio_features["f0_hz"], 0.0, librosa.midi_to_hz(110.0)
    )
    return audio_features


def mask_by_confidence(audio_features, confidence_level=0.1):
    """For the violin model, the masking causes fast dips in loudness. 
    This quick transient is interpreted by the model as the "plunk" sound.
    """
    mask_idx = audio_features["f0_confidence"] < confidence_level
    audio_features["f0_hz"][mask_idx] = 0.0
    # audio_features['loudness_db'][mask_idx] = -ddsp.spectral_ops.LD_RANGE
    return audio_features


def smooth_loudness(audio_features, filter_size=3):
    """Smooth loudness with a box filter."""
    smoothing_filter = np.ones([filter_size]) / float(filter_size)
    audio_features["loudness_db"] = np.convolve(
        audio_features["loudness_db"], smoothing_filter, mode="same"
    )
    return audio_features


########################################################################
# Parameters
########################################################################

CKPT_DIR = "./models"
TITLE_LABEL = "DDSP Timbre Transfer demo"
UPLOAD_FILE_LABEL = "Choose a .wav file"
BUTTON_LABEL = "Compute transfer (can take a while on CPU)"

########################################################################
# UI
########################################################################

########################### Sidebar
instrument_model = st.sidebar.selectbox("Choose a model", ("Violin", "Flute", "Flute2"))


st.sidebar.markdown(
    "This button will at least adjusts the average loudness and pitch to be similar to the training data (although not for user trained models)."
)


auto_adjust = st.sidebar.checkbox("auto-adjust", value=True)

st.sidebar.markdown(
    """
You can also make additional manual adjustments:
    
    - Shift the fundmental frequency to a more natural register.
    - Silence audio_np below a threshold on f0_confidence.
    - Adjsut the overall loudness level.
"""
)

f0_octave_shift = st.sidebar.slider("f0_octave_shift", min_value=-2, max_value=2, value=0, step=1)
f0_confidence_threshold = st.sidebar.slider(
    "f0_confidence_threshold",  min_value=0.0, max_value=1.0, value=0.0, step=0.05
)
loudness_db_shift = st.sidebar.slider("loudness_db_shift", min_value=-20, max_value=20, value=0, step=1)

########################### Main panel

st.title(TITLE_LABEL)

uploaded_file = st.file_uploader(UPLOAD_FILE_LABEL, type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file)
    audio_np, unused_sr = load_audio(uploaded_file)

    if st.button(BUTTON_LABEL):
        tf.reset_default_graph()
        sess = tf.Session(TARGET)
        tf.keras.backend.set_session(sess)
        reset_crepe()

        audio_features, audio_features_mod = compute_audio_features(audio_np)

        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))
        ax[0].plot(audio_features["loudness_db"])
        ax[0].set_ylabel("loudness_db")

        ax[1].plot(librosa.hz_to_midi(audio_features["f0_hz"]))
        ax[1].set_ylabel("f0 [midi]")

        ax[2].plot(audio_features["f0_confidence"])
        ax[2].set_ylabel("f0 confidence")
        _ = ax[2].set_xlabel("Time step [frame]")

        st.pyplot()

        # Load chosen model, cache it too ?
        model_dir = os.path.join(CKPT_DIR, "solo_%s_ckpt" % instrument_model.lower())
        ckpt_files = [f for f in tf.gfile.ListDirectory(model_dir) if "model.ckpt" in f]
        ckpt_name = ".".join(ckpt_files[0].split(".")[:2])
        ckpt = os.path.join(model_dir, ckpt_name)

        # Parse gin config
        with gin.unlock_config():
            gin_file = os.path.join(model_dir, "operative_config-0.gin")
            gin.parse_config_file(gin_file, skip_unknown=True)

        # Ensure dimensions sampling rates are equal
        time_steps_train = gin.query_parameter("DefaultPreprocessor.time_steps")
        n_samples_train = gin.query_parameter("Additive.n_samples")
        hop_size = int(n_samples_train / time_steps_train)

        time_steps = int(audio_np.shape[0] / hop_size)
        n_samples = time_steps * hop_size

        gin_params = [
            "Additive.n_samples = {}".format(n_samples),
            "FilteredNoise.n_samples = {}".format(n_samples),
            "DefaultPreprocessor.time_steps = {}".format(time_steps),
        ]

        with gin.unlock_config():
            gin.parse_config(gin_params)

        # Trim all input vectors to correct lengths
        for key in ["f0_hz", "f0_confidence", "loudness_db"]:
            audio_features[key] = audio_features[key][:time_steps]
        audio_features["audio"] = audio_features["audio"][:n_samples]

        # Set up the model just to predict audio given new conditioning
        tf.reset_default_graph()

        ph_f0_hz = tf.placeholder(tf.float32, shape=[1, time_steps])
        ph_loudness_db = tf.placeholder(tf.float32, shape=[1, time_steps])
        ph_audio = tf.placeholder(tf.float32, shape=[1, n_samples])
        ph_features = {
            "loudness_db": ph_loudness_db,
            "f0_hz": ph_f0_hz,
            "audio": ph_audio,
        }

        model = ddsp.training.models.Autoencoder()
        predictions = model.get_outputs(ph_features, training=False)

        sess = tf.Session(TARGET)

        model.restore(sess, ckpt)

        # Resynth audio
        audio_features_mod = {k: v.copy() for k, v in audio_features.items()}
        if auto_adjust:
            # Adjust the peak loudness.
            l = audio_features["loudness_db"]
            model_ld_avg_max = {"Violin": -34.0, "Flute": -45.0, "Flute2": -44.0,}[
                instrument_model
            ]
            ld_max = np.max(audio_features["loudness_db"])
            ld_diff_max = model_ld_avg_max - ld_max
            audio_features_mod = shift_ld(audio_features_mod, ld_diff_max)

            # Further adjust the average loudness above a threshold.
            l = audio_features_mod["loudness_db"]
            model_ld_mean = {"Violin": -44.0, "Flute": -51.0, "Flute2": -53.0,}[
                instrument_model
            ]
            ld_thresh = -50.0
            ld_mean = np.mean(l[l > ld_thresh])
            ld_diff_mean = model_ld_mean - ld_mean
            audio_features_mod = shift_ld(audio_features_mod, ld_diff_mean)

            # Shift the pitch register.
            model_p_mean = {"Violin": 73.0, "Flute": 81.0, "Flute2": 74.0,}[
                instrument_model
            ]
            p = librosa.hz_to_midi(audio_features["f0_hz"])
            p[p == -np.inf] = 0.0
            p_mean = p[l > ld_thresh].mean()
            p_diff = model_p_mean - p_mean
            p_diff_octave = p_diff / 12.0
            round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
            p_diff_octave = round_fn(p_diff_octave)
            audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)

        audio_features_mod = shift_ld(audio_features_mod, loudness_db_shift)
        audio_features_mod = shift_f0(audio_features_mod, f0_octave_shift)
        audio_features_mod = mask_by_confidence(
            audio_features_mod, f0_confidence_threshold
        )

        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))
        ax[0].plot(audio_features["loudness_db"])
        ax[0].plot(audio_features_mod["loudness_db"])
        ax[0].set_ylabel("loudness_db")

        ax[1].plot(librosa.hz_to_midi(audio_features["f0_hz"]))
        ax[1].plot(librosa.hz_to_midi(audio_features_mod["f0_hz"]))
        ax[1].set_ylabel("f0 [midi]")

        ax[2].plot(audio_features_mod["f0_confidence"])
        ax[2].plot(
            np.ones_like(audio_features_mod["f0_confidence"]) * f0_confidence_threshold
        )
        ax[2].set_ylabel("f0 confidence")
        _ = ax[2].set_xlabel("Time step [frame]")

        st.pyplot()

        af = audio_features if audio_features_mod is None else audio_features_mod
        feed_dict = {}
        feed_dict[ph_features["loudness_db"]] = af["loudness_db"][
            np.newaxis, :, np.newaxis
        ]
        feed_dict[ph_features["f0_hz"]] = af["f0_hz"][np.newaxis, :, np.newaxis]
        feed_dict[ph_features["audio"]] = af["audio"][np.newaxis, :]
        audio_gen = sess.run(predictions["audio_gen"], feed_dict=feed_dict)[0]

        st.write("Hello world")

        # Plot results
        with tf.Session() as sess:
            specplot(audio_np, sess=sess)
            plt.title("Original")
            st.pyplot()
            specplot(audio_gen, sess=sess)
            _ = plt.title("Resynthesis")
            st.pyplot()

        # Export
        sf.write("output.wav", audio_gen, 44100)
