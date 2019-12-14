#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import argparse
from dateutil.relativedelta import relativedelta
from distutils.util import strtobool
import logging
import os
import sys
import time

import numpy as np
import six
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import torch
from torch import nn
from torchvision import transforms

from utils import find_files
from utils import read_hdf5, check_hdf5
from utils import read_txt
from dswnv import encode_mu_law, decode_mu_law
from dswnv import OneHot, initialize, DSWNV

#np.set_printoptions(threshold=np.inf)
#torch.set_printoptions(threshold=np.inf)


def validate_length(x, y, upsampling_factor=0):
    """FUNCTION TO VALIDATE LENGTH

    Args:
        x (ndarray): numpy.ndarray with x.shape[0] = len_x
        y (ndarray): numpy.ndarray with y.shape[0] = len_y
        upsampling_factor (int): upsampling factor

    Returns:
        (ndarray): length adjusted x with same length y
        (ndarray): length adjusted y with same length x
    """
    if upsampling_factor == 0:
        if x.shape[0] < y.shape[0]:
            y = y[:x.shape[0]]
        if x.shape[0] > y.shape[0]:
            x = x[:y.shape[0]]
        assert len(x) == len(y)
    else:
        mod_sample = x.shape[0] % upsampling_factor
        if mod_sample > 0:
            x = x[:-mod_sample]
        if x.shape[0] > y.shape[0] * upsampling_factor:
            x = x[:-(x.shape[0]-y.shape[0]*upsampling_factor)]
        elif x.shape[0] < y.shape[0] * upsampling_factor:
            y = y[:-((y.shape[0]*upsampling_factor-x.shape[0])//upsampling_factor)]
        assert len(x) == len(y) * upsampling_factor

    return x, y


def train_generator(wav_file_list, feat_file_list, receptive_field, string_path='/feat_org_lf0', \
                    batch_size=1100, wav_transform=None, training=True, upsampling_factor=0):
    """TRAINING BATCH GENERATOR

    Args:
        wav_list (str): list of wav files
        feat_list (str): list of feat files
        receptive_field (int): size of receptive filed
        batch_size (int): batch size
        wav_transform (func): preprocessing function for waveform
        upsampling_factor (int): upsampling factor

    Return:
        (object): generator instance
    """
    # shuffle list
    n_files = len(wav_file_list)

    if training:
        idx = np.random.permutation(n_files)
        wav_list = [wav_file_list[i] for i in idx]
        feat_list = [feat_file_list[i] for i in idx]
    else:
        idx = np.arange(n_files)
        wav_list = [wav_file_list[i] for i in idx]
        feat_list = [feat_file_list[i] for i in idx]

    # check batch_size
    if batch_size != 0 and upsampling_factor != 0:
        batch_mod = (receptive_field + batch_size + 1) % upsampling_factor
        if batch_mod > 0:
            logging.warn("batch size is decreased due to upsampling (%d -> %d)" % (
                batch_size, batch_size - batch_mod))
            batch_size -= batch_mod

    while True:
        # process over all of files
        c_idx = 0
        for wavfile, featfile in zip(wav_list, feat_list):
            # load wavefrom and aux feature
            x, fs = sf.read(wavfile, dtype=np.float32)
            h = read_hdf5(featfile, string_path)

            # check both lengths are same
            logging.debug("before x length = %d" % x.shape[0])
            logging.debug("before h length = %d" % h.shape[0])
            x, h = validate_length(x, h, upsampling_factor)
            logging.debug("after x length = %d" % x.shape[0])
            logging.debug("after h length = %d" % h.shape[0])

            # use mini batch with upsampling
            x_mulaw = wav_transform(x)

            x_class = torch.LongTensor(x_mulaw)
            x = torch.LongTensor(x_mulaw)
            h = torch.FloatTensor(h)

            if torch.cuda.is_available():
                x_class = x_class.cuda()
                x = x.cuda()
                h = h.cuda()
            x = OneHot(x.unsqueeze(0)).squeeze(0)

            len_frm = len(h)
            h_ss = 0
            x_ss = 0
            h_bs = (receptive_field + batch_size + 1) // upsampling_factor
            x_bs = h_bs * upsampling_factor
            delta = batch_size // upsampling_factor
            while True:
                if len_frm > h_bs:
                    yield x_class, x, h, c_idx, idx[c_idx], wavfile, h_bs, x_bs, h_ss, x_ss 

                    h_ss += delta
                    x_ss = h_ss * upsampling_factor
                    len_frm -= delta
                elif len_frm*upsampling_factor > (receptive_field+1):
                    yield x_class, x, h, c_idx, idx[c_idx], wavfile, -1, -1, h_ss, x_ss 
                    break
                else:
                    break

            c_idx += 1
            #if c_idx > 0:
            #if c_idx > 1:
            #if c_idx > 2:
            #    break

        yield [], [], [], -1, -1, [], [], [], [], []

        # re-shuffle
        if training:
            idx = np.random.permutation(n_files)
            wav_list = [wav_file_list[i] for i in idx]
            feat_list = [feat_file_list[i] for i in idx]


def save_checkpoint(checkpoint_dir, model, optimizer, numpy_random_state, torch_random_state, iterations):
    """FUNCTION TO SAVE CHECKPOINT

    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    model.cpu()
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "numpy_random_state": numpy_random_state,
        "torch_random_state": torch_random_state,
        "iterations": iterations}
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)
    model.cuda()
    logging.info("%d-iter checkpoint created." % iterations)


def main():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument("--waveforms", required=True,
                        type=str, help="directory or list of wav files")
    parser.add_argument("--waveforms_eval", required=True,
                        type=str, help="directory or list of evaluation wav files")
    parser.add_argument("--feats", required=True,
                        type=str, help="directory or list of aux feat files")
    parser.add_argument("--feats_eval", required=True,
                        type=str, help="directory or list of evaluation aux feat files")
    parser.add_argument("--stats", required=True,
                        type=str, help="hdf5 file including statistics")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    # network structure setting
    parser.add_argument("--n_quantize", default=256,
                        type=int, help="number of quantization")
    parser.add_argument("--n_aux", default=39,
                        type=int, help="number of dimension of aux feats")
    parser.add_argument("--dilation_depth", default=3,
                        type=int, help="depth of dilation")
    parser.add_argument("--dilation_repeat", default=3,
                        type=int, help="depth of dilation")
    parser.add_argument("--hid_chn", default=192,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--skip_chn", default=256,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--kernel_size", default=6,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--aux_kernel_size", default=3,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--aux_dilation_size", default=2,
                        type=int, help="kernel size of dilated causal convolution")
    parser.add_argument("--upsampling_factor", default=110,
                        type=int, help="upsampling factor of aux features"
                                       "(if set 0, do not apply)")
    parser.add_argument("--string_path", default="/feat_org_lf0",
                        type=str, help="directory to save the model")
    # network training setting
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=1100,
                        type=int, help="batch size (if set 0, utterance batch will be used)")
    parser.add_argument("--epoch_count", default=500,
                        type=int, help="number of training epochs")
    parser.add_argument("--do_prob", default=0,
                        type=float, help="dropout probability")
    parser.add_argument("--wav_conv_flag", default=False,
                        type=strtobool, help="flag to use 1d conv of wav")
    # other setting
    parser.add_argument("--audio_in", default=False,
        type=strtobool, help="flag for including previous sample in conditioning feat")
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--pretrained", default=None,
                        type=str, help="model path to restart training")
    parser.add_argument("--GPU_device", default=None,
                        type=int, help="selection of GPU device")
    parser.add_argument("--verbose", default=1,
                        type=int, help="log level")
    args = parser.parse_args()

    if args.GPU_device is not None:
        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]  = str(args.GPU_device)

    # make experimental directory
    if not os.path.exists(args.expdir):
        os.makedirs(args.expdir)

    # set log level
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARN,
                            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S',
                            filename=args.expdir + "/train.log")
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.warn("logging is disabled.")

    # fix seed
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True #faster
    #torch.backends.cudnn.deterministic = True #reproducibility_slower
    #torch.backends.cudnn.benchmark = False #reproducibility_slower

    # save args as conf
    torch.save(args, args.expdir + "/model.conf")

    # # define network
    model = DSWNV(
        n_quantize=args.n_quantize,
        n_aux=args.n_aux,
        hid_chn=args.hid_chn,
        skip_chn=args.skip_chn,
        dilation_depth=args.dilation_depth,
        dilation_repeat=args.dilation_repeat,
        kernel_size=args.kernel_size,
        aux_kernel_size=args.aux_kernel_size,
        aux_dilation_size=args.aux_dilation_size,
        audio_in_flag=args.audio_in,
        do_prob=args.do_prob,
        wav_conv_flag=args.wav_conv_flag,
        upsampling_factor=args.upsampling_factor)
    logging.info(model)
    criterion = nn.CrossEntropyLoss()

    # define transforms
    string_path_name = args.string_path.split('feat_')[1]
    logging.info(string_path_name)
    scaler = StandardScaler()
    if check_hdf5(args.stats, "/mean_"+string_path_name):
        scaler.mean_ = read_hdf5(args.stats, "/mean_"+string_path_name)
        scaler.scale_ = read_hdf5(args.stats, "/scale_"+string_path_name)
    elif check_hdf5(args.stats, "/mean_"+args.string_path):
        scaler.mean_ = read_hdf5(args.stats, "/mean_"+args.string_path)
        scaler.scale_ = read_hdf5(args.stats, "/scale_"+args.string_path)
    else:
        scaler.mean_ = read_hdf5(args.stats, "/mean_feat_"+string_path_name)
        scaler.scale_ = read_hdf5(args.stats, "/scale_feat_"+string_path_name)
    mean_src = torch.FloatTensor(scaler.mean_)
    std_src = torch.FloatTensor(scaler.scale_)

    # send to gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
        mean_src = mean_src.cuda()
        std_src = std_src.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)

    model.train()
    model.apply(initialize)
    model.scale_in.weight = torch.nn.Parameter(torch.unsqueeze(torch.diag(1.0/std_src.data),2))
    model.scale_in.bias = torch.nn.Parameter(-(mean_src.data/std_src.data))

    for param in model.parameters():
        param.requires_grad = True
    for param in model.scale_in.parameters():
        param.requires_grad = False

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
    logging.info('Trainable Parameters: %.3f million' % parameters)

    module_list = list(model.conv_aux.parameters()) + list(model.upsampling.parameters())
    if model.wav_conv_flag:
        module_list += list(model.wav_conv.parameters())
    module_list += list(model.causal.parameters())
    module_list += list(model.in_x.parameters()) + list(model.dil_h.parameters())
    module_list += list(model.out_skip.parameters())
    module_list += list(model.out_1.parameters()) + list(model.out_2.parameters())
    optimizer = torch.optim.Adam(module_list, lr=args.lr)

    # resume
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint["model"])
        epoch_idx = checkpoint["iterations"]
        logging.info("pretrained from %d-iter checkpoint." % epoch_idx)
        epoch_idx = 0
    elif args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)
    else:
        epoch_idx = 0

    wav_transform = transforms.Compose([lambda x: encode_mu_law(x, args.n_quantize)])

    # define generator training
    if os.path.isdir(args.waveforms):
        filenames = sorted(find_files(args.waveforms, "*.wav", use_dir_name=False))
        wav_list = [args.waveforms + "/" + filename for filename in filenames]
        feat_list = [args.feats + "/" + filename.replace(".wav", ".h5") for filename in filenames]
    elif os.path.isfile(args.waveforms):
        wav_list = read_txt(args.waveforms)
        feat_list = read_txt(args.feats)
    else:
        logging.error("--waveforms should be directory or list.")
        sys.exit(1)
    assert len(wav_list) == len(feat_list)
    logging.info("number of training data = %d." % len(wav_list))
    if args.pretrained is None:
        generator = train_generator(
            wav_list, feat_list,
            model.receptive_field,
            string_path=args.string_path,
            batch_size=args.batch_size,
            wav_transform=wav_transform,
            training=True,
            upsampling_factor=args.upsampling_factor)
    else:
        generator = train_generator(
            wav_list, feat_list,
            model.receptive_field,
            string_path=args.string_path,
            batch_size=args.batch_size,
            wav_transform=wav_transform,
            training=True,
            upsampling_factor=args.upsampling_factor)

    # define generator evaluation
    if os.path.isdir(args.waveforms_eval):
        filenames_eval = sorted(find_files(args.waveforms_eval, "*.wav", use_dir_name=False))
        wav_list_eval = [args.waveforms_eval + "/" + filename for filename in filenames_eval]
        feat_list_eval = [args.feats_eval + "/" + filename.replace(".wav", ".h5") \
                        for filename in filenames_eval]
    elif os.path.isfile(args.waveforms_eval):
        wav_list_eval = read_txt(args.waveforms_eval)
        feat_list_eval = read_txt(args.feats_eval)
    else:
        logging.error("--waveforms_eval should be directory or list.")
        sys.exit(1)
    logging.info("number of evaluation data = %d." % len(wav_list_eval))
    assert len(wav_list_eval) == len(feat_list_eval)
    if args.pretrained is None:
        generator_eval = train_generator(
            wav_list_eval, feat_list_eval,
            model.receptive_field,
            string_path=args.string_path,
            batch_size=args.batch_size,
            wav_transform=wav_transform,
            training=False,
            upsampling_factor=args.upsampling_factor)
    else:
        generator_eval = train_generator(
            wav_list_eval, feat_list_eval,
            model.receptive_field,
            string_path=args.string_path,
            batch_size=args.batch_size,
            wav_transform=wav_transform,
            training=False,
            upsampling_factor=args.upsampling_factor)

    # train
    loss = []
    total = 0
    iter_idx = 0
    iter_count = 0
    min_eval_loss = 99999999.99
    min_eval_loss_std = 99999999.99
    min_idx = -1
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < args.epoch_count:
        start = time.time()
        batch_x_class, batch_x, batch_h, c_idx, utt_idx, wavfile, h_bs, x_bs, h_ss, x_ss = next(generator)
        if c_idx < 0: # summarize epoch
            numpy_random_state = np.random.get_state()
            torch_random_state = torch.get_rng_state()
            # save current epoch model
            save_checkpoint(args.expdir, model, optimizer, numpy_random_state, torch_random_state, epoch_idx+1)
            # report current epoch
            logging.info("(EPOCH:%d) average training loss = %.6f (+- %.6f) (%.3f min., %.3f sec / batch)" % (
                epoch_idx + 1, np.mean(np.array(loss, dtype=np.float64)), \
                np.std(np.array(loss, dtype=np.float64)), total / 60.0, total / iter_count))
            logging.info("estimated training required time = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((args.epoch_count - (epoch_idx + 1)) * total))))
            # compute loss in evaluation data
            loss = []
            total = 0
            iter_count = 0
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            logging.info("Evaluation data")
            with torch.no_grad():
                while True:
                    start = time.time()
                    batch_x_class, batch_x, batch_h, c_idx, utt_idx, wavfile, h_bs, x_bs, h_ss, x_ss = \
                        next(generator_eval)
                    if c_idx < 0:
                        break

                    tf = batch_h.shape[0]
                    ts = batch_x.shape[0]

                    batch_h = batch_h[h_ss:]
                    batch_x_class = batch_x_class[x_ss:]
                    batch_x = batch_x[x_ss:]
                    if h_bs != -1:
                        batch_h = batch_h[:h_bs]
                        batch_x_class = batch_x_class[1:x_bs]
                        batch_x = batch_x[:x_bs-1]
                    else:
                        batch_x = batch_x[:-1]
                        batch_x_class = batch_x_class[1:]
                    batch_h = batch_h.transpose(0,1).unsqueeze(0)
                    batch_x = batch_x.transpose(0,1).unsqueeze(0)

                    batch_output = model(batch_x, batch_h)[0]

                    if h_ss > 0:
                        batch_loss = criterion(batch_output[model.receptive_field:], \
                                                batch_x_class[model.receptive_field:])
                    else:
                        batch_loss = criterion(batch_output, batch_x_class)

                    loss.append(batch_loss.item())
                    logging.info("batch eval loss %s [%d:%d] %d %d %d %d %d %d = %.3f (%.3f sec)" % (
                        os.path.basename(os.path.dirname(wavfile))+"/"+os.path.basename(wavfile), c_idx+1, \
                            utt_idx+1, tf, ts, h_ss, h_bs, x_ss, x_bs, batch_loss.item(), time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            eval_loss = np.mean(np.array(loss, dtype=np.float64))
            eval_loss_std = np.std(np.array(loss, dtype=np.float64))
            logging.info("(EPOCH:%d) average evaluation loss = %.6f (+- %.6f) (%.3f min., %.3f sec / batch)" %(
                epoch_idx + 1, eval_loss, eval_loss_std, total / 60.0, total / iter_count))
            if (eval_loss+eval_loss_std) <= (min_eval_loss+min_eval_loss_std):
                min_eval_loss = eval_loss
                min_eval_loss_std = eval_loss_std
                min_idx = epoch_idx
            logging.info("min_eval_loss=%.6f (+- %.6f), min_idx=%d" % (\
                            min_eval_loss, min_eval_loss_std, min_idx+1))
            loss = []
            total = 0
            iter_count = 0
            epoch_idx += 1
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            for param in model.scale_in.parameters():
                param.requires_grad = False
            # start next epoch
            if epoch_idx < args.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
                batch_x_class, batch_x, batch_h, c_idx, utt_idx, wavfile, h_bs, x_bs, h_ss, x_ss = \
                    next(generator)
        # feedforward and backpropagate current batch
        if epoch_idx < args.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))

            tf = batch_h.shape[0]
            ts = batch_x.shape[0]

            batch_h = batch_h[h_ss:]
            batch_x_class = batch_x_class[x_ss:]
            batch_x = batch_x[x_ss:]
            if h_bs != -1:
                batch_h = batch_h[:h_bs]
                batch_x_class = batch_x_class[1:x_bs]
                batch_x = batch_x[:x_bs-1]
            else:
                batch_x = batch_x[:-1]
                batch_x_class = batch_x_class[1:]
            batch_h = batch_h.transpose(0,1).unsqueeze(0)
            batch_x = batch_x.transpose(0,1).unsqueeze(0)

            batch_output = model(batch_x, batch_h, do=True)[0]

            if h_ss > 0:
                batch_loss = criterion(batch_output[model.receptive_field:], \
                                                batch_x_class[model.receptive_field:])
            else:
                batch_loss = criterion(batch_output, batch_x_class)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss.append(batch_loss.item())
            logging.info("batch loss %s [%d:%d] %d %d %d %d %d %d = %.3f (%.3f sec)" % (
                os.path.basename(os.path.dirname(wavfile))+"/"+os.path.basename(wavfile), c_idx+1, utt_idx+1,
                    tf, ts, h_ss, h_bs, x_ss, x_bs, batch_loss.item(), time.time() - start))
            iter_idx += 1
            iter_count += 1
            total += time.time() - start

    # save final model
    model.cpu()
    torch.save({"model": model.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
