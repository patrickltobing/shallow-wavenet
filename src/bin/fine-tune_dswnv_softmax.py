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
from utils import read_hdf5
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


def train_generator(wav_file_list, feat_file_list, receptive_field, batch_size=8800, \
                string_path='/feat_org_lf0', wav_transform=None, training=True, upsampling_factor=110):
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
            #h = read_hdf5(featfile, "/feat_org_lf0")
            h = read_hdf5(featfile, string_path)
            #h = read_hdf5(featfile, "/feat_org_p_lf0_grucycle_1e-8")
            #if "/VCC2SF1/" in featfile:
            #    #h = read_hdf5(featfile, "/feat_cvmcep_cyclevqvae-mult-jnt-scpost-1_50-50_10_VCC2SF1")
            #    h = read_hdf5(featfile, "/feat_cvmcep_cyclevae-mult-jnt-scpost-2_32_97_VCC2SF1")
            #elif "/VCC2SM1/" in featfile:
            #    #h = read_hdf5(featfile, "/feat_cvmcep_cyclevqvae-mult-jnt-scpost-1_50-50_10_VCC2SM1")
            #    h = read_hdf5(featfile, "/feat_cvmcep_cyclevae-mult-jnt-scpost-2_32_97_VCC2SM1")
            #elif "/VCC2SF2/" in featfile:
            #    #h = read_hdf5(featfile, "/feat_cvmcep_cyclevqvae-mult-jnt-scpost-1_50-50_10_VCC2SF2")
            #    h = read_hdf5(featfile, "/feat_cvmcep_cyclevae-mult-jnt-scpost-2_32_97_VCC2SF2")
            #elif "/VCC2SM2/" in featfile:
            #    #h = read_hdf5(featfile, "/feat_cvmcep_cyclevqvae-mult-jnt-scpost-1_50-50_10_VCC2SM2")
            #    h = read_hdf5(featfile, "/feat_cvmcep_cyclevae-mult-jnt-scpost-2_32_97_VCC2SM2")
            #elif "/VCC2TF1/" in featfile:
            #    #h = read_hdf5(featfile, "/feat_cvmcep_cyclevqvae-mult-jnt-scpost-1_50-50_10_VCC2TF1")
            #    h = read_hdf5(featfile, "/feat_cvmcep_cyclevae-mult-jnt-scpost-2_32_97_VCC2TF1")
            #elif "/VCC2TM1/" in featfile:
            #    #h = read_hdf5(featfile, "/feat_cvmcep_cyclevqvae-mult-jnt-scpost-1_50-50_10_VCC2TM1")
            #    h = read_hdf5(featfile, "/feat_cvmcep_cyclevae-mult-jnt-scpost-2_32_97_VCC2TM1")
            #elif "/VCC2TF2/" in featfile:
            #    #h = read_hdf5(featfile, "/feat_cvmcep_cyclevqvae-mult-jnt-scpost-1_50-50_10_VCC2TF2")
            #    h = read_hdf5(featfile, "/feat_cvmcep_cyclevae-mult-jnt-scpost-2_32_97_VCC2TF2")
            #elif "/VCC2TM2/" in featfile:
            #    #h = read_hdf5(featfile, "/feat_cvmcep_cyclevqvae-mult-jnt-scpost-1_50-50_10_VCC2TM2")
            #    h = read_hdf5(featfile, "/feat_cvmcep_cyclevae-mult-jnt-scpost-2_32_97_VCC2TM2")

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
            #if training:
            #    x = OneHot(x.unsqueeze(0), training=True).squeeze(0)
            #else:
            #    x = OneHot(x.unsqueeze(0), training=False).squeeze(0)
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
    parser.add_argument("--config", required=True,
                        type=str, help="configure file")
    parser.add_argument("--pretrained", required=True,
                        type=str, help="pretrained model path")
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument("--init_acc", default=False,
                        type=strtobool, help="flag to compute accuracy of initial pretrained model")
    parser.add_argument("--string_path", required=True,
                        type=str, help="flag to compute accuracy of initial pretrained model")
    # other setting
    parser.add_argument("--seed", default=1,
                        type=int, help="seed number")
    parser.add_argument("--resume", default=None,
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

    # load config
    config = torch.load(args.config)
    config.expdir = args.expdir
    config.pretrained = args.pretrained
    config.string_path_ft = args.string_path

    # save args as conf
    if not args.init_acc:
        torch.save(config, args.expdir + "/model.conf")

    # # define network
    model = DSWNV(
        n_quantize=config.n_quantize,
        n_aux=config.n_aux,
        hid_chn=config.hid_chn,
        skip_chn=config.skip_chn,
        dilation_depth=config.dilation_depth,
        dilation_repeat=config.dilation_repeat,
        kernel_size=config.kernel_size,
        aux_kernel_size=config.aux_kernel_size,
        aux_dilation_size=config.aux_dilation_size,
        do_prob=config.do_prob,
        upsampling_factor=config.upsampling_factor)
    logging.info(model)
    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint["model"])
    epoch_idx = checkpoint["iterations"]
    logging.info("pretrained from %d-iter checkpoint." % epoch_idx)
    epoch_idx = 0

    # send to gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    else:
        logging.error("gpu is not available. please check the setting.")
        sys.exit(1)

    if not args.init_acc:
        for param in model.parameters():
            param.requires_grad = True
        for param in model.scale_in.parameters():
            param.requires_grad = False
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        logging.info('Trainable Parameters: %.3f million' % parameters)
        module_list = list(model.conv_aux.parameters())+ list(model.upsampling.parameters())
        module_list += list(model.causal.parameters()) + list(model.in_x.parameters())
        module_list += list(model.dil_h.parameters()) + list(model.out_skip.parameters())
        module_list += list(model.out_1.parameters()) + list(model.out_2.parameters())
        optimizer = torch.optim.Adam(module_list, lr=config.lr)
        model.train()
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    # resume
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch_idx = checkpoint["iterations"]
        logging.info("restored from %d-iter checkpoint." % epoch_idx)

    wav_transform = transforms.Compose([lambda x: encode_mu_law(x, config.n_quantize)])

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

    generator = train_generator(
        wav_list, feat_list,
        receptive_field=model.receptive_field,
        batch_size=config.batch_size,
        wav_transform=wav_transform,
        string_path=config.string_path_ft,
        training=True,
        upsampling_factor=config.upsampling_factor)

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
    generator_eval = train_generator(
        wav_list_eval, feat_list_eval,
        receptive_field=model.receptive_field,
        batch_size=config.batch_size,
        wav_transform=wav_transform,
        string_path=config.string_path_ft,
        training=False,
        upsampling_factor=config.upsampling_factor)

    if args.init_acc:
        epoch_idx = -1

    # train
    loss = []
    total = 0
    iter_idx = 0
    iter_count = 0
    min_eval_loss = 99999999.99
    min_eval_loss_std = 99999999.99
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    while epoch_idx < config.epoch_count:
        start = time.time()
        batch_x_class, batch_x, batch_h, c_idx, utt_idx, wavfile, h_bs, x_bs, h_ss, x_ss = next(generator)
        if c_idx < 0: # summarize epoch
            # save current epoch model
            if not args.init_acc:
                numpy_random_state = np.random.get_state()
                torch_random_state = torch.get_rng_state()
                save_checkpoint(args.expdir, model, optimizer, numpy_random_state, \
                                torch_random_state, epoch_idx + 1)
            # report current epoch
            logging.info("(EPOCH:%d) average training loss = %.6f (+- %.6f) (%.3f min., %.3f sec / batch)" % (
                epoch_idx + 1, np.mean(np.array(loss, dtype=np.float64)), \
                np.std(np.array(loss, dtype=np.float64)), total / 60.0, total / iter_count))
            logging.info("estimated training required time = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((config.epoch_count - (epoch_idx + 1))*total))))
            # compute loss in evaluation data
            loss = []
            total = 0
            iter_count = 0
            if not args.init_acc:
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
            logging.info("Evaluation data")
            while True:
                start = time.time()
                batch_x_class, batch_x, batch_h, c_idx, utt_idx, wavfile, h_bs, x_bs, h_ss, x_ss \
                    = next(generator_eval)
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
                    os.path.basename(os.path.dirname(wavfile))+"/"+os.path.basename(wavfile), \
                    c_idx+1, utt_idx+1, tf, ts, h_ss, h_bs, x_ss, x_bs, batch_loss.item(), time.time()-start))
                iter_count += 1
                total += time.time() - start
            eval_loss = np.mean(np.array(loss, dtype=np.float64))
            eval_loss_std = np.std(np.array(loss, dtype=np.float64))
            logging.info("(EPOCH:%d) average evaluation loss = %.6f (+- %.6f) (%.3f min., %.3f sec / batch)"%(
                epoch_idx + 1, eval_loss, eval_loss_std, total / 60.0, total / iter_count))
            if not args.init_acc:
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
            else:
                exit()
            # start next epoch
            if epoch_idx < config.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
                batch_x_class, batch_x, batch_h, c_idx, utt_idx, wavfile, h_bs, x_bs, h_ss, x_ss \
                        = next(generator)
        # feedforward and backpropagate current batch
        if epoch_idx < config.epoch_count:
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

            if not args.init_acc:
                batch_output = model(batch_x, batch_h, do=True)[0]
            else:
                batch_output = model(batch_x, batch_h)[0]

            if h_ss > 0:
                batch_loss = criterion(batch_output[model.receptive_field:], \
                                        batch_x_class[model.receptive_field:])
            else:
                batch_loss = criterion(batch_output, batch_x_class)

            if not args.init_acc:
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
