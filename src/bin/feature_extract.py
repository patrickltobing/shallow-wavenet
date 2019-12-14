#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import sys
from distutils.util import strtobool

import logging
import numpy as np
from numpy.matlib import repmat
from scipy.interpolate import interp1d
#from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter

from utils import find_files
from utils import read_txt
from utils import write_hdf5, read_hdf5

from multiprocessing import Array

import pysptk as ps
import pyworld as pw
#import librosa
import soundfile as sf

np.set_printoptions(threshold=np.inf)

FS = 22050
FS = 24000
#FS = 44100
#FS = 48000
SHIFTMS = 5.0
MINF0 = 40
MAXF0 = 700
#MCEP_DIM = 34
MCEP_DIM = 49
MCEP_ALPHA = 0.455
MCEP_ALPHA = 0.466
#MCEP_ALPHA = 0.544
#MCEP_ALPHA = 0.554
FFTL = 1024
LOWPASS_CUTOFF = 20
HIGHPASS_CUTOFF = 70
OVERWRITE = True


def low_cut_filter(x, fs, cutoff=HIGHPASS_CUTOFF):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def analyze(wav, fs=FS, minf0=MINF0, maxf0=MAXF0, fperiod=SHIFTMS, fftl=FFTL, f0=None, time_axis=None):
    #f0_flr = pw.get_cheaptrick_f0_floor(fs, fftl)
    #logging.info(f0_flr)
    #fft_size = pw.get_cheaptrick_fft_size(fs, f0_flr)
    #logging.info(fft_size)
    #f0_flr = pw.get_cheaptrick_f0_floor(fs, fft_size)
    #logging.info(f0_flr)
    if f0 is None or time_axis is None:
        _f0, time_axis = pw.harvest(wav, fs, f0_floor=60.0, frame_period=fperiod)
        f0 = pw.stonemask(wav, _f0, time_axis, fs)
    sp = pw.cheaptrick(wav, f0, time_axis, fs, fft_size=fftl)
    ap = pw.d4c(wav, f0, time_axis, fs, fft_size=fftl)

    return time_axis, f0, sp, ap


def analyze_range(wav, fs=FS, minf0=MINF0, maxf0=MAXF0, fperiod=SHIFTMS, fftl=FFTL, f0=None, time_axis=None):
    if f0 is None or time_axis is None:
        #logging.info("%lf %lf %lf %lf" % (minf0, maxf0, fperiod, fftl))
        #logging.info("1")
        _f0, time_axis = pw.harvest(wav, fs, f0_floor=minf0, f0_ceil=maxf0, frame_period=fperiod)
        #_f0, time_axis = pw.harvest(wav, fs, f0_floor=60, f0_ceil=maxf0, frame_period=fperiod)
        #_f0, time_axis = pw.harvest(wav, fs, f0_floor=60, frame_period=fperiod)
        #_f0, time_axis = pw.harvest(wav, fs, f0_floor=minf0, frame_period=fperiod)
        #_f0, time_axis = pw.harvest(wav, fs, f0_floor=minf0, frame_period=fperiod)
        #logging.info("2")
        f0 = pw.stonemask(wav, _f0, time_axis, fs)
        #logging.info("3")
        #f0, time_axis = pw.harvest(wav, fs, f0_floor=minf0, f0_ceil=maxf0, frame_period=fperiod)
    sp = pw.cheaptrick(wav, f0, time_axis, fs, fft_size=fftl)
    #logging.info("4")
    ap = pw.d4c(wav, f0, time_axis, fs, fft_size=fftl)
    #logging.info("5")

    return time_axis, f0, sp, ap


#def read_wav(wav_file, cutoff=HIGHPASS_CUTOFF, fftl_ns=None):
def read_wav(wav_file, cutoff=HIGHPASS_CUTOFF):
    #fs, x = wavfile.read(wav_file)
    #x = librosa.util.fix_length(x, len(x) + fftl_ns // 2)
    x, fs = sf.read(wav_file)
    #x = np.array(x, dtype=np.float64)
    if cutoff != 0:
        x = low_cut_filter(x, fs, cutoff)

    return fs, x


def low_pass_filter(x, fs, cutoff=LOWPASS_CUTOFF, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument("--expdir", required=True,
        type=str, help="directory to save the log")
    parser.add_argument(
        "--waveforms", default=None,
        help="directory or list of filename of input wavfile")
    parser.add_argument(
        "--hdf5dir", default=None,
        help="directory to save hdf5")
    parser.add_argument(
        "--wavdir", default=None,
        help="directory to save of preprocessed wav file")
    parser.add_argument(
        "--wavanasyndir", default=None,
        help="directory to save of preprocessed wav file")
    parser.add_argument(
        "--fs", default=FS,
        type=int, help="Sampling frequency")
    parser.add_argument(
        "--shiftms", default=SHIFTMS,
        type=float, help="Frame shift in msec")
    parser.add_argument(
        "--minf0", default=MINF0,
        type=int, help="minimum f0")
    parser.add_argument(
        "--maxf0", default=MAXF0,
        type=int, help="maximum f0")
    parser.add_argument(
        "--mcep_dim", default=MCEP_DIM,
        type=int, help="Dimension of mel cepstrum")
    parser.add_argument(
        "--mcep_alpha", default=MCEP_ALPHA,
        type=float, help="Alpha of mel cepstrum")
    parser.add_argument(
        "--fftl", default=FFTL,
        type=int, help="FFT length")
    parser.add_argument(
        "--fftl_ns", default=None,
        type=int, help="FFT length for noise shaped waveforms")
    parser.add_argument(
        "--highpass_cutoff", default=HIGHPASS_CUTOFF,
        type=int, help="Cut off frequency in lowpass filter")
    parser.add_argument("--init", default=False,
        type=strtobool, help="flag for computing stats of dtw-ed feature")
    parser.add_argument(
        "--n_jobs", default=10,
        type=int, help="number of parallel jobs")
    parser.add_argument(
        "--verbose", default=1,
        type=int, help="log message level")

    args = parser.parse_args()

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/feature_extract.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/feature_extract.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/feature_extract.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # read list
    if os.path.isdir(args.waveforms):
        file_list = sorted(find_files(args.waveforms, "*.wav"))
    else:
        file_list = read_txt(args.waveforms)

    # check directory existence
    if (args.wavdir is not None) and (not os.path.exists(args.wavdir)):
        os.makedirs(args.wavdir)
    if (args.wavanasyndir is not None) and (not os.path.exists(args.wavanasyndir)):
        os.makedirs(args.wavanasyndir)
    if not os.path.exists(args.hdf5dir):
        os.makedirs(args.hdf5dir)

    def feature_extract(wav_list, arr):
        n_wav = len(wav_list)
        n_sample = 0
        n_frame = 0
        count = 1
        max_frame = 0
        for wav_name in wav_list:
            # load wavfile and highpass-filter
            fs, x = read_wav(wav_name, cutoff=args.highpass_cutoff)
            n_sample += x.shape[0]
            logging.info(wav_name+" "+str(x.shape[0])+" "+str(n_sample)+" "+str(count))

            # check sampling frequency
            if not fs == args.fs:
                logging.debug("ERROR: sampling frequency is not matched.")
                sys.exit(1)

            hdf5name = args.hdf5dir + "/" + os.path.basename(wav_name).replace(".wav", ".h5")
            logging.info(hdf5name)

            if not args.init:
                _, f0, spc, ap = analyze_range(x, fs=fs, minf0=args.minf0, maxf0=args.maxf0, \
                                            fperiod=args.shiftms, fftl=args.fftl)
                # concatenate
                uv, cont_f0 = convert_continuos_f0(np.array(f0))
                cont_f0_lpf = low_pass_filter(cont_f0, int(1.0 / (args.shiftms * 0.001)), cutoff=20)
                codeap = pw.code_aperiodicity(ap, fs)
                #logging.info(codeap)
                logging.info(codeap.shape)
                mcep = ps.sp2mc(spc, args.mcep_dim, args.mcep_alpha)
                cont_f0_lpf = np.expand_dims(cont_f0_lpf, axis=-1)
                uv = np.expand_dims(uv, axis=-1)
                log_contf0_lpf = np.log(cont_f0_lpf)
                feats_lf0 = np.concatenate([uv, log_contf0_lpf, codeap, mcep], axis=1)
                logging.info(feats_lf0.shape)

                write_hdf5(hdf5name, "/feat_org_lf0", feats_lf0)
                n_frame += feats_lf0.shape[0]
                if max_frame < feats_lf0.shape[0]:
                    max_frame = feats_lf0.shape[0]

                # overwrite wav file
                if args.highpass_cutoff != 0:
                    #wavfile.write(args.wavdir + "/" + os.path.basename(wav_name), fs, np.int16(x))
                    sf.write(args.wavdir + "/" + os.path.basename(wav_name), x, fs, 'PCM_16')
                wavpath = args.wavanasyndir + "/" + os.path.basename(wav_name)
                logging.info(wavpath)
                sp_rec = ps.mc2sp(mcep, args.mcep_alpha, args.fftl)
                #wav = np.clip(pw.synthesize(f0, sp_rec, ap, fs, frame_period=args.shiftms), -32768, 32767)
                wav = np.clip(pw.synthesize(f0, sp_rec, ap, fs, frame_period=args.shiftms), -1, 1)
                #wavfile.write(wavpath, fs, np.int16(wav))
                sf.write(wavpath, wav, fs, 'PCM_16')
            else:
                _, f0, _, _ = analyze(x, fs=fs, fperiod=args.shiftms, fftl=args.fftl)
                write_hdf5(hdf5name, "/f0", f0)
                n_frame += f0.shape[0]
                if max_frame < f0.shape[0]:
                    max_frame = f0.shape[0]

            count += 1
        arr[0] += n_wav
        arr[1] += n_sample
        arr[2] += n_frame
        if (n_wav > 0):
            logging.info(str(arr[0])+" "+str(n_wav)+" "+str(arr[1])+" "+str(n_sample/n_wav)+" "+str(arr[2])\
                            +" "+str(n_frame/n_wav)+" max_frame = "+str(max_frame))

    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    arr = mp.Array('d', 3)
    #logging.info(arr[:])
    for f in file_lists:
        p = mp.Process(target=feature_extract, args=(f,arr))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()

    logging.info(str(arr[0])+" "+str(arr[1])+" "+str(arr[1]/arr[0])+" "+str(arr[2])+" "+str(arr[2]/arr[0]))


if __name__ == "__main__":
    main()
