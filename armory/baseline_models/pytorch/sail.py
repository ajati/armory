"""
CNN model for raw audio classification

Model contributed by: MITRE Corporation
Adapted from: https://github.com/mravanelli/SincNet
"""
import logging

from art.classifiers import PyTorchClassifier
import numpy as np
import torch

from armory.data.utils import maybe_download_weights_from_s3

# Load model from MITRE external repo: https://github.com/hkakitani/SincNet
# This needs to be defined in your config's `external_github_repo` field to be
# downloaded and placed on the PYTHONPATH
from SAIL import dnn_models
import pdb

logger = logging.getLogger(__name__)

# NOTE: Underlying dataset sample rate is 16 kHz. SincNet uses this SAMPLE_RATE to
# determine internal filter high cutoff frequency.
#SAMPLE_RATE = 8000
#WINDOW_STEP_SIZE = 375
#WINDOW_LENGTH = int(SAMPLE_RATE * WINDOW_STEP_SIZE / 1000)
WINDOW_LENGTH=48000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#adpted from sincnet.py
def preprocessing_fn(batch):
    """
    Standardize, then normalize sound clips
    """
    processed_batch = []
    for clip in batch:
        signal = clip.astype(np.float64)
        # Signal normalization
        signal = signal / np.max(np.abs(signal))

        # get pseudorandom chunk of fixed length (from SincNet's create_batches_rnd)
        signal_length = len(signal)

        if signal_length < WINDOW_LENGTH:
           signal = np.concatenate((signal, np.zeros(WINDOW_LENGTH-signal_length)))
        else:
            #np.random.seed(signal_length)
            #signal_start = (
            #    np.random.randint(signal_length / WINDOW_LENGTH - 1)
            #    * WINDOW_LENGTH
            #    % signal_length
            #)
            signal_start = np.random.randint(0, signal_length-WINDOW_LENGTH)
            signal_stop = signal_start + WINDOW_LENGTH
            signal = signal[signal_start:signal_stop]

        processed_batch.append(signal)

    return np.array(processed_batch)


# NOTE: PyTorchClassifier expects numpy input, not torch.Tensor input
def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    model = dnn_models.get_model(weights_file=weights_file, **model_kwargs)
    model.to(DEVICE)

    wrapped_model = PyTorchClassifier(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(
            model.parameters(), lr=1e-3, betas=(.5, .999)
        ),
        input_shape=(WINDOW_LENGTH,),
        nb_classes=40,
    )
    return wrapped_model
