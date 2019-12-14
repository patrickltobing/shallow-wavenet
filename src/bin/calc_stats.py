#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function

import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py
import soundfile as sf

from utils import read_hdf5
from utils import read_txt
from utils import write_hdf5


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feats", default=None, required=True,
        help="name of the list of hdf5 files")
    parser.add_argument(
        "--wavs", default=None,
        help="name of the list of wav files")
    parser.add_argument("--n_quantize", default=256,
                        type=int, help="number of quantization")
    parser.add_argument("--string_path", default="/feat_org_lf0",
                        type=str, help="path of h5 data")
    parser.add_argument(
        "--stats", default=None, required=True,
        help="filename of hdf5 format")

    args = parser.parse_args()

    # read list and define scaler
    filenames = read_txt(args.feats)
    #wavfiles = read_txt(args.wavs)

    #scaler = StandardScaler()
    #scaler_magspec = StandardScaler()
    #scaler_powspec = StandardScaler()
    #scaler_melsp_logc = StandardScaler()
    #scaler_melsp_org = StandardScaler()
    #scaler_melsp = StandardScaler()
    #scaler_melsp_sc = StandardScaler()
    #scaler_sdmelsp = StandardScaler()
    #scaler_mfcc = StandardScaler()
    #scaler_sdmfcc = StandardScaler()

    #scaler_org = StandardScaler()
    scaler_org_lf0 = StandardScaler()
    #scaler_wav_mulaw = StandardScaler()
    #scaler_wav = StandardScaler()

    #scaler_sc_org = StandardScaler()
    #scaler_f0mcep_org = StandardScaler()
    #scaler_f0mcep_sc_org = StandardScaler()
    #scaler_sd = StandardScaler()
    #scaler_sd_org = StandardScaler()
    print("number of training utterances =", len(filenames))
    print(args.string_path)

    # process over all of data
    #for filename, wavfile in zip(filenames, wavfiles):
    for filename in filenames:
        #hdf5_file = h5py.File(filename, "r")
        #print(hdf5_file.keys())
        #print(hdf5_file.values())
        #print(hdf5_file)
        #for i in hdf5_file:
        #    print(i)
        #print(hdf5_file["/melsp"].shape)
        #exit()
        #magspec = read_hdf5(filename, "/magspec")
        #powspec = read_hdf5(filename, "/powspec")
        #melsp_logc = read_hdf5(filename, "/melsp_logc")
        #melsp_org = read_hdf5(filename, "/melsp_org")
        #melsp = read_hdf5(filename, "/melsp")
        #melsp_sc = read_hdf5(filename, "/melsp_sc")
        #sdmelsp = read_hdf5(filename, "/sdmelsp")
        #mfcc = read_hdf5(filename, "/mfcc")
        #sdmfcc = read_hdf5(filename, "/sdmfcc")
        #feat = read_hdf5(filename, "/feat")

        #feat_org = read_hdf5(filename, "/feat_org")
        #feat_org_lf0 = read_hdf5(filename, "/feat_org_lf0")
        feat_org_lf0 = read_hdf5(filename, args.string_path)
        #wav, fs = sf.read(wavfile, dtype=np.float32)
        #wav_mulaw = encode_mu_law(wav, args.n_quantize).astype(np.float32)
        #wav = decode_mu_law(wav_mulaw, args.n_quantize)

        #feat_sc_org = read_hdf5(filename, "/feat_sc_org")
        #feat_f0mcep_org = read_hdf5(filename, "/feat_f0mcep_org")
        #feat_f0mcep_sc_org = read_hdf5(filename, "/feat_f0mcep_sc_org")
        #sdfeat = read_hdf5(filename, "/sdfeat")
        #sdfeat_org = read_hdf5(filename, "/sdfeat_org")
        #print(filename)
        #print(melsp)
        #scaler_magspec.partial_fit(magspec)
        #scaler_powspec.partial_fit(powspec)
        #scaler_melsp_logc.partial_fit(melsp_logc)
        #scaler_melsp_org.partial_fit(melsp_org)
        #scaler_melsp.partial_fit(melsp)
        #scaler_melsp_sc.partial_fit(melsp_sc)
        #scaler_sdmelsp.partial_fit(sdmelsp)
        #scaler_mfcc.partial_fit(mfcc)
        #scaler_sdmfcc.partial_fit(sdmfcc)
        #scaler.partial_fit(feat[:, 1:])
        #scaler_org.partial_fit(feat_org[:, 1:])
        #scaler_sc_org.partial_fit(feat_sc_org[:, 1:])
        #scaler_sd.partial_fit(sdfeat[:, 1:])
        #scaler_sd_org.partial_fit(sdfeat_org[:, 1:])
        #scaler.partial_fit(feat)

        #scaler_org.partial_fit(feat_org)
        scaler_org_lf0.partial_fit(feat_org_lf0)
        #scaler_wav_mulaw.partial_fit(np.expand_dims(wav_mulaw, axis=-1))
        #scaler_wav.partial_fit(np.expand_dims(wav, axis=-1))

        #scaler_sc_org.partial_fit(feat_sc_org)
        #scaler_f0mcep_org.partial_fit(feat_f0mcep_org)
        #scaler_f0mcep_sc_org.partial_fit(feat_f0mcep_sc_org)
        #scaler_sd.partial_fit(sdfeat)
        #scaler_sd_org.partial_fit(sdfeat_org)

    # add uv term
    #mean = np.zeros((feat.shape[1]))
    #scale = np.ones((feat.shape[1]))
    #mean[1:] = scaler.mean_
    #scale[1:] = scaler.scale_
    #mean = scaler.mean_
    #scale = scaler.scale_
    #mean_org = np.zeros((feat_org.shape[1]))
    #scale_org = np.ones((feat_org.shape[1]))
    #mean_sc_org = np.zeros((feat_sc_org.shape[1]))
    #scale_sc_org = np.ones((feat_sc_org.shape[1]))
    #mean_org[1:] = scaler_org.mean_
    #scale_org[1:] = scaler_org.scale_
    #mean_org[1:] = scaler_org.mean_
    #scale_org[1:] = scaler_org.scale_

    #mean_org = scaler_org.mean_
    #scale_org = scaler_org.scale_
    mean_org_lf0 = scaler_org_lf0.mean_
    scale_org_lf0 = scaler_org_lf0.scale_
    #mean_wav_mulaw = scaler_wav_mulaw.mean_
    #scale_wav_mulaw = scaler_wav_mulaw.scale_
    #mean_wav = scaler_wav.mean_
    #scale_wav = scaler_wav.scale_

    #mean_sc_org = scaler_sc_org.mean_
    #scale_sc_org = scaler_sc_org.scale_
    #mean_f0mcep_org = scaler_f0mcep_org.mean_
    #scale_f0mcep_org = scaler_f0mcep_org.scale_
    #mean_f0mcep_sc_org = scaler_f0mcep_sc_org.mean_
    #scale_f0mcep_sc_org = scaler_f0mcep_sc_org.scale_
    #sdmean = np.zeros((sdfeat.shape[1]))
    #sdscale = np.ones((sdfeat.shape[1]))
    #sdmean[1:] = scaler_sd.mean_
    #sdscale[1:] = scaler_sd.scale_
    #sdmean = scaler_sd.mean_
    #sdscale = scaler_sd.scale_
    #sdmean_org = np.zeros((sdfeat_org.shape[1]))
    #sdscale_org = np.ones((sdfeat_org.shape[1]))
    #sdmean_org[1:] = scaler_sd_org.mean_
    #sdscale_org[1:] = scaler_sd_org.scale_
    #sdmean_org = scaler_sd_org.mean_
    #sdscale_org = scaler_sd_org.scale_

    # write to hdf5
    #write_hdf5(args.stats, "/mean_magspec", scaler_magspec.mean_)
    #write_hdf5(args.stats, "/scale_magspec", scaler_magspec.scale_)
    #write_hdf5(args.stats, "/mean_powspec", scaler_powspec.mean_)
    #write_hdf5(args.stats, "/scale_powspec", scaler_powspec.scale_)
    #write_hdf5(args.stats, "/mean_melsp_logc", scaler_melsp_logc.mean_)
    #write_hdf5(args.stats, "/scale_melsp_logc", scaler_melsp_logc.scale_)
    #write_hdf5(args.stats, "/mean_melsp_org", scaler_melsp_org.mean_)
    #write_hdf5(args.stats, "/scale_melsp_org", scaler_melsp_org.scale_)
    #write_hdf5(args.stats, "/mean_melsp", scaler_melsp.mean_)
    #write_hdf5(args.stats, "/scale_melsp", scaler_melsp.scale_)
    #write_hdf5(args.stats, "/mean_melsp_sc", scaler_melsp_sc.mean_)
    #write_hdf5(args.stats, "/scale_melsp_sc", scaler_melsp_sc.scale_)
    #write_hdf5(args.stats, "/mean_sdmelsp", scaler_sdmelsp.mean_)
    #write_hdf5(args.stats, "/scale_sdmelsp", scaler_sdmelsp.scale_)
    #write_hdf5(args.stats, "/mean_mfcc", scaler_mfcc.mean_)
    #write_hdf5(args.stats, "/scale_mfcc", scaler_mfcc.scale_)
    #write_hdf5(args.stats, "/mean_sdmfcc", scaler_sdmfcc.mean_)
    #write_hdf5(args.stats, "/scale_sdmfcc", scaler_sdmfcc.scale_)
    #write_hdf5(args.stats, "/mean", mean)
    #write_hdf5(args.stats, "/scale", scale)

    #print(mean_org)
    #print(scale_org)
    #write_hdf5(args.stats, "/mean_org", mean_org)
    #write_hdf5(args.stats, "/scale_org", scale_org)
    print(mean_org_lf0)
    print(scale_org_lf0)
    if args.string_path == "/feat_org_lf0":
        write_hdf5(args.stats, "/mean_org_lf0", mean_org_lf0)
        write_hdf5(args.stats, "/scale_org_lf0", scale_org_lf0)
    else:
        write_hdf5(args.stats, "/mean_"+args.string_path, mean_org_lf0)
        write_hdf5(args.stats, "/scale_"+args.string_path, scale_org_lf0)
    #print(mean_wav_mulaw)
    #print(scale_wav_mulaw)
    #write_hdf5(args.stats, "/mean_mulaw", mean_wav_mulaw)
    #write_hdf5(args.stats, "/scale_mulaw", scale_wav_mulaw)
    #print(mean_wav)
    #print(scale_wav)
    #write_hdf5(args.stats, "/mean_wav", mean_wav)
    #write_hdf5(args.stats, "/scale_wav", scale_wav)

    #write_hdf5(args.stats, "/mean_sc_org", mean_sc_org)
    #write_hdf5(args.stats, "/scale_sc_org", scale_sc_org)
    #write_hdf5(args.stats, "/mean_f0mcep_org", mean_f0mcep_org)
    #write_hdf5(args.stats, "/scale_f0mcep_org", scale_f0mcep_org)
    #write_hdf5(args.stats, "/mean_f0mcep_sc_org", mean_f0mcep_sc_org)
    #write_hdf5(args.stats, "/scale_f0mcep_sc_org", scale_f0mcep_sc_org)
    #write_hdf5(args.stats, "/sdmean", sdmean)
    #write_hdf5(args.stats, "/sdscale", sdscale)
    #write_hdf5(args.stats, "/sdmean_org", sdmean_org)
    #write_hdf5(args.stats, "/sdscale_org", sdscale_org)


if __name__ == "__main__":
    main()
