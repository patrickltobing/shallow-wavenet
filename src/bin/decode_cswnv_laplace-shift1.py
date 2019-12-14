#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

from distutils.util import strtobool
import argparse
import logging
import math
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp

from utils import find_files
from utils import read_hdf5
from utils import read_txt
from utils import shape_hdf5
from cswnv_shift1 import CSWNV


def pad_list(batch_list, pad_value=0.0):
    """FUNCTION TO PAD VALUE

    Args:
        batch_list (list): list of batch, where the shape of i-th batch (T_i, C)
        pad_value (float): value to pad

    Return:
        (ndarray): padded batch with the shape (B, T_max, C)

    """
    batch_size = len(batch_list)
    maxlen = max([batch.shape[0] for batch in batch_list])
    n_feats = batch_list[0].shape[-1]
    batch_pad = np.zeros((batch_size, maxlen, n_feats))
    for idx, batch in enumerate(batch_list):
        batch_pad[idx, :batch.shape[0]] = batch

    return batch_pad


def decode_generator(feat_list, batch_size=7, string_path='/feat_org_lf0', seg=1, upsampling_factor=0, \
                        spk_trg=None, min_idx=None):
    """DECODE BATCH GENERATOR

    Args:
        featdir (str): directory including feat files
        batch_size (int): batch size in decoding
        upsampling_factor (int): upsampling factor

    Return:
        (object): generator instance
    """
    with torch.no_grad():
        #string_path = "/feat_org_lf0"
        #string_path = "/feat_lat_lf0_cyclevae-mult-jnt-scpost_gauss-2_32_97_VCC2TF1-300"
        #string_path = "/feat_lat_lf0_cyclevqvae-mult-jnt-scpost-1_50-50_101_VCC2TF1"
        if spk_trg is not None:
            if min_idx is not None:
                string_path = '/feat_cvmcep_cycvae-'+str(min_idx)+'-'+spk_trg
            else:
                string_path = '/feat_cvmcep_cycvae-'+spk_trg
            #string_path = '/feat_cvmcepgv_cycvae-'+spk_trg
            #string_path = '/feat_DiffGVF0_cycvae-'+spk_trg
            #string_path = '/feat_WsolaF0DiffGV_cycvae-'+spk_trg
        logging.info(string_path)
        # sort with the feature length
        shape_list = [shape_hdf5(f, string_path)[0] for f in feat_list]
        idx = np.argsort(shape_list)
        feat_list = [feat_list[i] for i in idx]

        # divide into batch list
        n_batch = math.ceil(len(feat_list) / batch_size)
        batch_lists = np.array_split(feat_list, n_batch)
        batch_lists = [f.tolist() for f in batch_lists]

        for batch_list in batch_lists:
            batch_x = []
            batch_h = []
            n_samples_list = []
            feat_ids = []
            for featfile in batch_list:
                # make seed waveform and load aux feature
                x = np.zeros(seg)
                h = read_hdf5(featfile, string_path)

                # append to list
                batch_x += [x]
                batch_h += [h]
                n_samples_list += [h.shape[0] * upsampling_factor]
                feat_ids += [os.path.basename(featfile).replace(".h5", "")]

            # convert list to ndarray
            batch_x = np.stack(batch_x, axis=0)
            batch_h = pad_list(batch_h)

            # convert to torch variable
            batch_x = torch.FloatTensor(batch_x)
            batch_h = torch.FloatTensor(batch_h).transpose(1, 2)
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_h = batch_h.cuda()

            yield feat_ids, (batch_x, batch_h, n_samples_list)


def main():
    parser = argparse.ArgumentParser()
    # decode setting
    parser.add_argument("--feats", required=True,
                        type=str, help="list or directory of aux feat files")
    parser.add_argument("--checkpoint", required=True,
                        type=str, help="model file")
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--outdir", required=True,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--fs", default=22050,
                        type=int, help="sampling rate")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="number of batch size in decoding")
    parser.add_argument("--n_gpus", default=1,
                        type=int, help="number of gpus")
    parser.add_argument("--spk_trg", default=None,
                        type=str, help="directory to save generated samples")
    parser.add_argument("--min_idx", default=None,
                        type=int, help="directory to save generated samples")
    # other setting
    parser.add_argument("--intervals", default=4410,
                        type=int, help="log interval")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--GPU_device", default=0,
                        type=int, help="selection of GPU device")
    parser.add_argument("--GPU_device_str", default=None,
                        type=str, help="selection of GPU device")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
    if args.GPU_device_str is None:
        os.environ["CUDA_VISIBLE_DEVICES"]  = str(args.GPU_device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]  = args.GPU_device_str

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # set log level
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.outdir + "/decode.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load config
    config = torch.load(args.config)
    logging.info(config)

    # get file list
    if os.path.isdir(args.feats):
        feat_list = sorted(find_files(args.feats, "*.h5"))
    elif os.path.isfile(args.feats):
        feat_list = read_txt(args.feats)
    else:
        logging.error("--feats should be directory or list.")
        sys.exit(1)

    # prepare the file list for parallel decoding
    feat_lists = np.array_split(feat_list, args.n_gpus)
    feat_lists = [f_list.tolist() for f_list in feat_lists]

    # define gpu decode function
    def gpu_decode(feat_list, gpu):
        with torch.cuda.device(gpu):
            with torch.no_grad():
                # define model and load parameters
                model = CSWNV(
                    n_aux=config.n_aux,
                    skip_chn=config.skip_chn,
                    hid_chn=config.hid_chn,
                    dilation_depth=config.dilation_depth,
                    dilation_repeat=config.dilation_repeat,
                    kernel_size=config.kernel_size,
                    aux_kernel_size=config.aux_kernel_size,
                    aux_dilation_size=config.aux_dilation_size,
                    seg=config.seg,
                    lpc=config.lpc,
                    aux_conv2d_flag=config.aux_conv2d_flag,
                    wav_conv_flag=config.wav_conv_flag,
                    upsampling_factor=config.upsampling_factor)
                logging.info(model)
                model.cuda()
                model.load_state_dict(torch.load(args.checkpoint)["model"])
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                torch.backends.cudnn.benchmark = True

                # define generator
                generator = decode_generator(
                    feat_list,
                    batch_size=args.batch_size,
                    string_path=config.string_path,
                    spk_trg=args.spk_trg,
                    min_idx=args.min_idx,
                    seg=model.seg,
                    upsampling_factor=config.upsampling_factor)

                # decode
                time_sample = []
                n_samples = []
                n_samples_t = []
                for feat_ids, (batch_x, batch_h, n_samples_list) in generator:
                    logging.info("decoding start")
                    start = time.time()
                    samples_list = model.batch_fast_generate(
                        batch_x, batch_h, n_samples_list, args.intervals)
                    time_sample.append(time.time()-start)
                    n_samples.append(max(n_samples_list))
                    n_samples_t.append(max(n_samples_list)*len(n_samples_list))
                    for feat_id, samples in zip(feat_ids, samples_list):
                        wav = np.clip(samples, -1, 1)
                        sf.write(args.outdir + "/" + feat_id + ".wav", wav, args.fs, "PCM_16")
                        logging.info("wrote %s.wav in %s." % (feat_id, args.outdir))
                logging.info("average time / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % (\
                    sum(time_sample)/sum(n_samples), sum(n_samples), sum(n_samples)/(1000*sum(time_sample))))
                logging.info("average throughput / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % (\
                sum(time_sample)/sum(n_samples_t), sum(n_samples_t), sum(n_samples_t)/(1000*sum(time_sample))))

    # parallel decode
    processes = []
    gpu = 0
    for i, feat_list in enumerate(feat_lists):
        p = mp.Process(target=gpu_decode, args=(feat_list, gpu,))
        p.start()
        processes.append(p)
        gpu += 1
        if (i + 1) % args.n_gpus == 0:
            gpu = 0

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
