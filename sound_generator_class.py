#!/usr/bin/env python
# coding: utf-8

# # Building Sound Generation class
# 
# the VAE was tranined on spectograms not raw audio so our task here is to reconstruct the audio from spectograms using inverse short time fourirer transform (ISTFT) using Griffin-lim method

# In[1]:


from vae_class import VAE
from preprocessing_model import MinMaxNormalizer
import librosa


# ## sound generator class
# responasible for genearting audios from spectrogram
# 
# ### Reconstruction the audio
# 
# 
# 1.   Reshaping the signal
# 2.   DeNormalize the signals; by using the saved values of min, max for every spectrogram
# 3.  convert the signal from log scale to linear scale
# 4.  apply ISTFT to convert signal from spectrogram to audio in time domain
# 5.  append the reconstructed signal to the list
# 
# 

# In[1]:


class SoundGenerator:
  def __init__(self, vae, hop_length):
    self.vae = vae
    self.hop_length = hop_length
    self._min_max_normalizer = MinMaxNormalizer(0,1)
  
  def generate(self, spectrograms, min_max_values):
    #reconstruct images/spects after being encoded
    generated_spectograms, latent_representations = self.vae.reconstruct(spectrograms)
    signals= self.convert_spectograms_to_audio(generated_spectograms, min_max_values)
    return signals, latent_representations
  
  def convert_spectograms_to_audio(self, spectrograms,min_max_values):
    signals = []
    for spectrogram, min_max_val in zip( spectrograms,min_max_values):
      #1 reshaping; removing the dummy third dim of input signals, (the channel size)
      log_spectrogram = spectrogram[:, : , 0]
      #2 applying deNorm
      denorm_log_spec = self._min_max_normalizer.denormlize(log_spectrogram, min_max_val["min"], min_max_val["max"])
      #3 linearize the spectrogram
      spec = librosa.db_to_amplitude(denorm_log_spec)
      #4 apply STFT
      signal = librosa.istft(spec, hop_length= self.hop_length)
      #5 append
      signals.append(signal)
    return signals

