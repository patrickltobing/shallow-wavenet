#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
from distutils.util import strtobool
import multiprocessing as mp
import os
import sys

import pyworld as pw
import pysptk as ps
from pysptk.synthesis import MLSADF
import numpy as np
#from scipy.io import wavfile
import soundfile as sf

from feature_extract import low_cut_filter
from utils import find_files
from utils import read_hdf5
from utils import read_txt

#FS = 16000
FS = 22050
FS = 24000
#FS = 44100
#FS = 48000
SHIFTMS = 5.0
FFTL = 1024
#MCEP_DIM_START = 3
MCEP_DIM_START = 4
MCEP_DIM_START = 5
#MCEP_DIM_START = 2
#MCEP_DIM_END = 27
#MCEP_DIM_END = 37
#MCEP_DIM_END = 42
#MCEP_DIM_END = 52
#MCEP_ALPHA=0.41000000000000003
MCEP_ALPHA = 0.455
MCEP_ALPHA = 0.466
#MCEP_ALPHA = 0.544
#MCEP_ALPHA = 0.554
MAG = 0.5


def synthesis_diff(x, diffmcep, rmcep=None, alpha=MCEP_ALPHA, fs=FS, shiftms=SHIFTMS):
        """filtering with a differential mel-cesptrum
        Parameters
        ----------
        x : array, shape (`samples`)
            array of waveform sequence
        diffmcep : array, shape (`T`, `dim`)
            array of differential mel-cepstrum sequence
        rmcep : array, shape (`T`, `dim`)
            array of reference mel-cepstrum sequence
            Default set to None
        alpha : float, optional
            Parameter of all-path transfer function
            Default set to 0.42
        Return
        ----------
        wav: array, shape (`samples`)
            Synethesized waveform
        """

        x = x.astype(np.float64)
        dim = diffmcep.shape[1] - 1
        shiftl = int(fs / 1000 * shiftms)

        if rmcep is not None:
            # power modification
            diffmcep = mod_power(rmcep + diffmcep, rmcep, alpha=alpha) - rmcep

        b = np.apply_along_axis(ps.mc2b, 1, diffmcep, alpha)
        assert np.isfinite(b).all()

        mlsa_fil = ps.synthesis.Synthesizer(
            MLSADF(dim, alpha=alpha), shiftl)
        wav = mlsa_fil.synthesis(x, b)

        return wav


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument(
        "--waveforms", default=None,
        help="directory or list of filename of input wavfile")
    parser.add_argument(
        "--stats", default=None,
        help="filename of hdf5 format")
    parser.add_argument(
        "--writedir", default=None,
        help="directory to save preprocessed wav file")
    parser.add_argument(
        "--fs", default=FS,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--shiftms", default=SHIFTMS,
        type=float, help="Frame shift in msec")
    parser.add_argument(
        "--fftl", default=FFTL,
        type=int, help="FFT length")
    parser.add_argument(
        "--mcep_dim_start", default=MCEP_DIM_START,
        type=int, help="Start index of mel cepstrum")
    #parser.add_argument(
    #    "--mcep_dim_end", default=MCEP_DIM_END,
    #    type=int, help="End index of mel cepstrum")
    parser.add_argument(
        "--mcep_alpha", default=MCEP_ALPHA,
        type=float, help="Alpha of mel cepstrum")
    parser.add_argument(
        "--mag", default=MAG,
        type=float, help="magnification of noise shaping")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")
    parser.add_argument(
        '--n_jobs', default=1,
        type=int, help="number of parallel jobs")
    parser.add_argument(
        '--inv', default=False, type=strtobool,
        help="if True, inverse filtering will be performed")
    args = parser.parse_args()

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)

    # check directory existence
    if not os.path.exists(args.writedir):
        os.makedirs(args.writedir)

    def noise_shaping(wav_list):
        for wav_name in wav_list:
            # load wavfile and apply low cut filter
            #fs, x = wavfile.read(wav_name)
            #wav_type = x.dtype
            #x = np.array(x, dtype=np.float64)
            x, fs = sf.read(wav_name)

            # check sampling frequency
            if not fs == args.fs:
                print("ERROR: sampling frequency is not matched.")
                sys.exit(1)

            # extract features (only for get the number of frames)
            #def analyze(wav, fs=22050, minf0=40, maxf0=700, fperiod=5.0, fftl=1024, f0=None, time_axis=None):
            #print(fs)
            #_, f0, _, _ = analyze(x, fs=fs)
            f0, _ = pw.harvest(x, fs, frame_period=5.0)
            num_frames = f0.shape[0]

            # load average mcep
            #mlsa_coef = read_hdf5(args.stats, "/mean_org")
            mlsa_coef = read_hdf5(args.stats, "/mean_org_lf0")
            #mlsa_coef = read_hdf5(args.stats, "/mean_feat_org_lf0")
            #mlsa_coef = mlsa_coef[args.mcep_dim_start:args.mcep_dim_end] * args.mag
            mlsa_coef = mlsa_coef[args.mcep_dim_start:] * args.mag
            mlsa_coef[0] = 0.0
            if args.inv:
                mlsa_coef[1:] = -1.0 * mlsa_coef[1:]
            mlsa_coef = np.tile(mlsa_coef, [num_frames, 1])

            # synthesis and write
            x_ns = synthesis_diff(
                x, mlsa_coef, alpha=args.mcep_alpha, fs=fs, shiftms=args.shiftms)
            x_ns = low_cut_filter(x_ns, args.fs, cutoff=70)
            write_name = args.writedir + "/" + os.path.basename(wav_name)
            sf.write(write_name, x_ns, args.fs, 'PCM_16')
            #if wav_type == np.int16:
            #    #wavfile.write(write_name, args.fs, np.int16(x_ns))
            #    sf.write(write_name, x_ns, args.fs, 'PCM_16')
            #else:
            #    #wavfile.write(write_name, args.fs, x_ns)
            #    sf.write(write_name, x_ns, args.fs, 'PCM_16')

    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    for f in file_lists:
        p = mp.Process(target=noise_shaping, args=(f,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
