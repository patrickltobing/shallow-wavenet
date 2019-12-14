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
import torch.nn.functional as F

from utils import find_files
from utils import read_hdf5, check_hdf5
from utils import read_txt
from cswnv_shift1 import initialize
from cswnv_shift1 import CSWNV, LSDloss, LaplaceLoss

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
                    batch_size=8800, seg=1, training=True, upsampling_factor=110, \
                    spk_trgsrc=None, feat_idx=None):
    """TRAINING BATCH GENERATOR

    Args:
        wav_list (str): list of wav files
        feat_list (str): list of feat files
        receptive_field (int): size of receptive filed
        batch_size (int): batch size
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
    chunk = receptive_field + batch_size + seg
    if batch_size != 0 and upsampling_factor != 0:
        batch_mod = chunk % upsampling_factor
        if batch_mod > 0:
            logging.warning("batch size is decreased due to upsampling (%d -> %d)" % (
                batch_size, batch_size - batch_mod))
            batch_size -= batch_mod

    #if spk_trgsrc is not None:
    #    if spk_trgsrc == 'rec':
    #        if feat_idx is not None:
    #            string_path = '/feat_recmcep_cycvae'
    #        else:
    #            string_path = '/feat_recmcep_cycvae'
    #    else:
    #        #string_path = '/feat_cvmcep-p_cyclevae-mult-jnt-mix-scpost_laplace-2_32_112_'+spk_trgsrc+'-300'
    #        #string_path = '/feat_oversmoothmcep_cycvae-'+spk_trgsrc
    #        string_path = '/feat_oversmoothrecmcep_cycvae-'+spk_trgsrc
    while True:
        # process over all of files
        c_idx = 0
        for wavfile, featfile in zip(wav_list, feat_list):
            # load wavefrom and aux feature
            #x, fs = sf.read(wavfile, dtype=np.float32)
            x, fs = sf.read(wavfile)
            h = read_hdf5(featfile, string_path)

            # check both lengths are same
            logging.debug("before x length = %d" % x.shape[0])
            logging.debug("before h length = %d" % h.shape[0])
            x, h = validate_length(x, h, upsampling_factor)
            logging.debug("after x length = %d" % x.shape[0])
            logging.debug("after h length = %d" % h.shape[0])

            # use mini batch with upsampling
            x = torch.FloatTensor(x)
            h = torch.FloatTensor(h)

            if torch.cuda.is_available():
                x = x.cuda()
                h = h.cuda()

            len_frm = len(h)
            h_ss = 0
            x_ss = 0
            h_bs = chunk // upsampling_factor
            x_bs = h_bs * upsampling_factor
            delta = batch_size // upsampling_factor
            while True:
                if len_frm*upsampling_factor-chunk >= seg:
                    yield x, h, c_idx, idx[c_idx], wavfile, h_bs, x_bs, h_ss, x_ss 

                    h_ss += delta
                    x_ss = h_ss * upsampling_factor
                    len_frm -= delta
                elif len_frm*upsampling_factor-(receptive_field+seg) >= seg:
                    yield x, h, c_idx, idx[c_idx], wavfile, -1, -1, h_ss, x_ss 
                    break
                else:
                    break

            c_idx += 1
            #if c_idx > 0:
            #if c_idx > 1:
            #if c_idx > 2:
            #    break

        yield [], [], -1, -1, [], [], [], [], []

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
    parser.add_argument("--expdir", required=True,
                        type=str, help="directory to save the model")
    parser.add_argument("--config", required=True,
                        type=str, help="model path to restart training")
    parser.add_argument("--pretrained", required=True,
                        type=str, help="model path to restart training")
    parser.add_argument("--spk_trgsrc", default=None,
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

    # save args as conf
    config = torch.load(args.config)
    config.expdir = args.expdir
    config.pretrained = args.pretrained
    config.string_path_ft = args.string_path

    if not args.init_acc:
        torch.save(config, args.expdir + "/model.conf")

    # define network
    model = CSWNV(
        n_aux=config.n_aux,
        skip_chn=config.skip_chn,
        hid_chn=config.hid_chn,
        dilation_depth=config.dilation_depth,
        dilation_repeat=config.dilation_repeat,
        kernel_size=config.kernel_size,
        aux_kernel_size=config.aux_kernel_size,
        aux_dilation_size=config.aux_dilation_size,
        do_prob=config.do_prob,
        seg=config.seg,
        lpc=config.lpc,
        aux_conv2d_flag=config.aux_conv2d_flag,
        wav_conv_flag=config.wav_conv_flag,
        upsampling_factor=config.upsampling_factor)
    logging.info(model)
    criterion_lsd = LSDloss()
    criterion_laplace = LaplaceLoss()

    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint["model"])
    epoch_idx = checkpoint["iterations"]
    logging.info("pretrained from %d-iter checkpoint." % epoch_idx)
    epoch_idx = 0

    # send to gpu
    if torch.cuda.is_available():
        model.cuda()
        criterion_lsd.cuda()
        criterion_laplace.cuda()
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
        module_list = list(model.conv_aux.parameters())
        module_list += list(model.upsampling.parameters())
        if model.aux_conv2d_flag and model.seg > 1:
            module_list += list(model.aux_conv2d.parameters())
        if model.wav_conv_flag:
            module_list += list(model.wav_conv.parameters())
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
            string_path=config.string_path_ft,
            spk_trgsrc=args.spk_trgsrc,
            seg=model.seg,
            batch_size=config.batch_size,
            training=True,
            upsampling_factor=config.upsampling_factor)
    else:
        generator = train_generator(
            wav_list, feat_list,
            model.receptive_field,
            string_path=config.string_path_ft,
            spk_trgsrc=args.spk_trgsrc,
            seg=model.seg,
            batch_size=config.batch_size,
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
    if args.pretrained is None:
        generator_eval = train_generator(
            wav_list_eval, feat_list_eval,
            model.receptive_field,
            string_path=config.string_path_ft,
            spk_trgsrc=args.spk_trgsrc,
            seg=model.seg,
            batch_size=config.batch_size,
            training=False,
            upsampling_factor=config.upsampling_factor)
    else:
        generator_eval = train_generator(
            wav_list_eval, feat_list_eval,
            model.receptive_field,
            string_path=config.string_path_ft,
            spk_trgsrc=args.spk_trgsrc,
            seg=model.seg,
            batch_size=config.batch_size,
            training=False,
            upsampling_factor=config.upsampling_factor)

    if args.init_acc:
        epoch_idx = -1

    # train
    logging.info(config.string_path_ft)
    loss_laplace = []
    loss_err = []
    loss_lsd = []
    fft_facts = []
    init_fft = 64
    hann_win = [None]*config.n_fft_facts
    if config.n_fft_facts == 5:
        fft_facts = [128, 256, 512, 1024, 2048]
        for i in range(config.n_fft_facts):
            hann_win[i] = torch.hann_window(fft_facts[i]).cuda()
    elif config.n_fft_facts == 9:
        fft_facts = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
        for i in range(config.n_fft_facts):
            hann_win[i] = torch.hann_window(fft_facts[i]).cuda()
    elif config.n_fft_facts == 17:
        fft_facts = [128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]
        for i in range(config.n_fft_facts):
            hann_win[i] = torch.hann_window(fft_facts[i]).cuda()
    else:
        for i in range(config.n_fft_facts):
            if i % 2 == 0:
                init_fft *= 2
                fft_facts.append(init_fft)
            else:
                fft_facts.append(init_fft+int(init_fft/2))
            hann_win[i] = torch.hann_window(fft_facts[i]).cuda()
    logging.info(fft_facts)
    batch_stft_loss = [None]*config.n_fft_facts
    stft_out = [None]*config.n_fft_facts
    stft_trg = [None]*config.n_fft_facts
    total = 0
    iter_idx = 0
    iter_count = 0
    min_eval_loss_lsd = 99999999.99
    min_eval_loss_laplace = 99999999.99
    min_eval_loss_err = 99999999.99
    min_eval_loss_lsd_std = 99999999.99
    min_eval_loss_laplace_std = 99999999.99
    min_eval_loss_err_std = 99999999.99
    min_idx = -1
    if args.resume is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
    logging.info("==%d EPOCH==" % (epoch_idx+1))
    logging.info("Training data")
    #config.epoch_count = 5300
    while epoch_idx < config.epoch_count:
        start = time.time()
        batch_x_float, batch_h, c_idx, utt_idx, wavfile, h_bs, x_bs, h_ss, x_ss = next(generator)
        if c_idx < 0: # summarize epoch
            # save current epoch model
            if not args.init_acc:
                numpy_random_state = np.random.get_state()
                torch_random_state = torch.get_rng_state()
                save_checkpoint(args.expdir, model, optimizer, numpy_random_state, torch_random_state, \
                                    epoch_idx + 1)
            # report current epoch
            logging.info("(EPOCH:%d) average training loss = %.6f (+- %.6f) %.6f dB (+- %.6f dB) %.6f "\
                "(+- %.6f) (%.3f min., %.3f sec / batch)" % (epoch_idx + 1, np.mean(loss_laplace), \
                np.std(loss_laplace), np.mean(loss_lsd), np.std(loss_lsd), np.mean(loss_err), \
                np.std(loss_err), total / 60.0, total / iter_count))
            logging.info("estimated training required time = {0.days:02}:{0.hours:02}:{0.minutes:02}:"\
            "{0.seconds:02}".format(relativedelta(seconds=int((config.epoch_count - (epoch_idx + 1))*total))))
            # compute loss in evaluation data
            loss_lsd = []
            loss_err = []
            loss_laplace = []
            total = 0
            iter_count = 0
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            logging.info("Evaluation data")
            while True:
                with torch.no_grad():
                    start = time.time()
                    batch_x_float, batch_h, c_idx, utt_idx, wavfile, h_bs, x_bs, h_ss, x_ss \
                        = next(generator_eval)
                    if c_idx < 0:
                        break

                    tf = batch_h.shape[0]
                    ts = batch_x_float.shape[0]

                    batch_h = batch_h[h_ss:]
                    batch_x_ = batch_x_float[x_ss:]
                    if model.lpc > 0:
                        if x_ss+model.lpc_offset >= 0:
                            batch_x_lpc = batch_x_float[x_ss+model.lpc_offset:]
                        else:
                            batch_x_lpc = batch_x_float[x_ss:]
                    if h_bs != -1:
                        batch_h = batch_h[:h_bs]
                        if model.lpc > 0:
                            if x_ss+model.lpc_offset >= 0:
                                batch_x_prob = batch_x_lpc[:x_bs-model.lpc_offset].unsqueeze(0)
                            else:
                                batch_x_prob = F.pad(batch_x_lpc[:x_bs], (-(x_ss+model.lpc_offset), 0), \
                                                    'constant', 0).unsqueeze(0)
                        batch_x = batch_x_[:x_bs-model.seg]
                        batch_x_float = batch_x_[model.seg:x_bs]
                    else:
                        if model.lpc > 0:
                            if x_ss+model.lpc_offset > 0:
                                batch_x_prob = batch_x_lpc.unsqueeze(0)
                            else:
                                batch_x_prob = F.pad(batch_x_lpc, (-(x_ss+model.lpc_offset)), \
                                                    'constant', 0).unsqueeze(0)
                        batch_x = batch_x_[:-model.seg]
                        batch_x_float = batch_x_[model.seg:]
                    batch_h = batch_h.transpose(0,1).unsqueeze(0)
                    batch_x = batch_x.unsqueeze(0).unsqueeze(1)
                    if h_ss > 0:
                        feat_len = batch_x_float[model.receptive_field:].shape[0]
                    else:
                        feat_len = batch_x_float.shape[0]

                    if model.lpc > 0:
                        mus, bs, log_bs, ass = model(batch_h, batch_x)
                        # jump off s samples as in synthesis
                        mus = mus[:,::model.seg,:]
                        bs = bs[:,::model.seg,:]
                        log_bs = log_bs[:,::model.seg,:]
                        ass = ass[:,::model.seg,:].flip(-1)
                        init_mus = mus
                        for j in range(model.seg):
                            tmp_smpls = batch_x_prob[:,j:-(model.seg-j)].unfold(1, model.lpc, model.seg)
                            lpc = torch.sum(ass*tmp_smpls,-1,keepdim=True)
                            if j > 0:
                                mus = torch.cat((mus, lpc+init_mus[:,:,j:j+1]),2)
                            else:
                                mus = lpc+init_mus[:,:,j:j+1]
                        mus = mus.reshape(mus.shape[0],-1)
                        bs = bs.reshape(bs.shape[0],-1)
                        log_bs = log_bs.reshape(log_bs.shape[0],-1)
                    else:
                        mus, bs, log_bs = model(batch_h, batch_x)

                    if h_ss > 0:
                        mus = mus[0,model.receptive_field:]
                        bs = bs[0,model.receptive_field:]
                        log_bs = log_bs[0,model.receptive_field:]
                        batch_x_float = batch_x_float[model.receptive_field:]
                    else:
                        mus = mus[0]
                        bs = bs[0]
                        log_bs = log_bs[0]

                    m_sum = 0
                    batch_loss_laplace = criterion_laplace(mus, bs, batch_x_float, log_b=log_bs)
                    eps = torch.empty(mus.shape).cuda().uniform_(-0.4999,0.5)
                    batch_output = mus-bs*eps.sign()*torch.log1p(-2*eps.abs())
                    batch_loss_err = torch.mean(torch.abs(batch_output-batch_x_float))
                    logging.info("%lf %E %lf %E" % (torch.min(batch_x_float), torch.mean(batch_x_float), \
                                                    torch.max(batch_x_float), torch.var(batch_x_float)))
                    logging.info("%lf %E %lf %E" % (torch.min(batch_output), torch.mean(batch_output), \
                                                    torch.max(batch_output), torch.var(batch_output)))
                    m = 0
                    for i in range(config.n_fft_facts):
                        if feat_len > int(fft_facts[i]/2):
                            stft_out[i] = torch.stft(batch_output, fft_facts[i], window=hann_win[i])
                            stft_trg[i] = torch.stft(batch_x_float, fft_facts[i], window=hann_win[i])
                            tmp_batch_stft_loss = criterion_lsd(stft_out[i], stft_trg[i])
                            if not torch.isinf(tmp_batch_stft_loss) and not torch.isnan(tmp_batch_stft_loss):
                                if m > 0:
                                    batch_loss_lsd = torch.cat((batch_loss_lsd, \
                                                                tmp_batch_stft_loss.unsqueeze(0)))
                                else:
                                    batch_loss_lsd = tmp_batch_stft_loss.unsqueeze(0)
                                m += 1

                    loss_err.append(batch_loss_err.item())
                    loss_laplace.append(batch_loss_laplace.item())
                    if m > 0:
                        batch_loss_lsd = torch.mean(batch_loss_lsd)
                        loss_lsd.append(batch_loss_lsd.item())
                        logging.info("batch eval loss %s [%d:%d] %d %d %d %d %d %d = %.3f %.3f dB %.6f "\
                            "(%.3f sec)" % (os.path.join(os.path.basename(os.path.dirname(wavfile)),\
                            os.path.basename(wavfile)), c_idx+1, utt_idx+1, tf, ts, h_ss, h_bs, x_ss, x_bs, \
                            batch_loss_laplace.item(), batch_loss_lsd.item(), batch_loss_err.item(), \
                            time.time() - start))
                    else:
                        logging.info("batch eval loss %s [%d:%d] %d %d %d %d %d %d = %.3f n/a %.6f "\
                            "(%.3f sec)" % (os.path.join(os.path.basename(os.path.dirname(wavfile)),\
                            os.path.basename(wavfile)), c_idx+1, utt_idx+1, tf, ts, h_ss, h_bs, x_ss, x_bs, \
                            batch_loss_laplace.item(), batch_loss_err.item(), time.time() - start))
                    iter_count += 1
                    total += time.time() - start
            eval_loss_lsd = np.mean(loss_lsd)
            eval_loss_lsd_std = np.std(loss_lsd)
            eval_loss_err = np.mean(loss_err)
            eval_loss_err_std = np.std(loss_err)
            eval_loss_laplace = np.mean(loss_laplace)
            eval_loss_laplace_std = np.std(loss_laplace)
            logging.info("(EPOCH:%d) average evaluation loss = %.6f (+- %.6f) %.6f dB (+- %.6f dB) %.6f "\
                "(+- %.6f) (%.3f min., %.3f sec / batch)" % (epoch_idx + 1, eval_loss_laplace, \
                eval_loss_laplace_std, eval_loss_lsd, eval_loss_lsd_std, eval_loss_err, \
                eval_loss_err_std, total / 60.0, total / iter_count))
            if not args.init_acc:
                if (eval_loss_laplace+eval_loss_laplace_std+eval_loss_lsd+eval_loss_lsd_std+eval_loss_err\
                    +eval_loss_err_std) <= (min_eval_loss_laplace+min_eval_loss_laplace_std\
                    +min_eval_loss_lsd+min_eval_loss_lsd_std+min_eval_loss_err+min_eval_loss_err_std):
                    min_eval_loss_lsd = eval_loss_lsd
                    min_eval_loss_lsd_std = eval_loss_lsd_std
                    min_eval_loss_err = eval_loss_err
                    min_eval_loss_err_std = eval_loss_err_std
                    min_eval_loss_laplace = eval_loss_laplace
                    min_eval_loss_laplace_std = eval_loss_laplace_std
                    min_idx = epoch_idx
                logging.info("min_eval_loss = %.6f (+- %.6f) %.6f dB (+- %.6f dB) %.6f (+- %.6f) "\
                    "min_idx=%d" % (min_eval_loss_laplace, min_eval_loss_laplace_std, min_eval_loss_lsd, \
                    min_eval_loss_lsd_std, min_eval_loss_err, min_eval_loss_err_std, min_idx+1))
            else:
                exit()
            loss_lsd = []
            loss_laplace = []
            loss_err = []
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
            if epoch_idx < config.epoch_count:
                start = time.time()
                logging.info("==%d EPOCH==" % (epoch_idx+1))
                logging.info("Training data")
                batch_x_float, batch_h, c_idx, utt_idx, wavfile, h_bs, x_bs, h_ss, x_ss = next(generator)
        # feedforward and backpropagate current batch
        if epoch_idx < config.epoch_count:
            logging.info("%d iteration [%d]" % (iter_idx+1, epoch_idx+1))

            tf = batch_h.shape[0]
            ts = batch_x_float.shape[0]

            batch_h = batch_h[h_ss:]
            batch_x_ = batch_x_float[x_ss:]
            if model.lpc > 0:
                if x_ss+model.lpc_offset >= 0:
                    batch_x_lpc = batch_x_float[x_ss+model.lpc_offset:]
                else:
                    batch_x_lpc = batch_x_float[x_ss:]
            if h_bs != -1:
                batch_h = batch_h[:h_bs]
                if model.lpc > 0:
                    if x_ss+model.lpc_offset >= 0:
                        batch_x_prob = batch_x_lpc[:x_bs-model.lpc_offset].unsqueeze(0)
                    else:
                        batch_x_prob = F.pad(batch_x_lpc[:x_bs], (-(x_ss+model.lpc_offset), 0), \
                                            'constant', 0).unsqueeze(0)
                batch_x = batch_x_[:x_bs-model.seg]
                batch_x_float = batch_x_[model.seg:x_bs]
            else:
                if model.lpc > 0:
                    if x_ss+model.lpc_offset > 0:
                        batch_x_prob = batch_x_lpc.unsqueeze(0)
                    else:
                        batch_x_prob = F.pad(batch_x_lpc, (-(x_ss+model.lpc_offset)), \
                                            'constant', 0).unsqueeze(0)
                batch_x = batch_x_[:-model.seg]
                batch_x_float = batch_x_[model.seg:]
            batch_h = batch_h.transpose(0,1).unsqueeze(0)
            batch_x = batch_x.unsqueeze(0).unsqueeze(1)
            if h_ss > 0:
                if model.seg > 1:
                    feat_len = batch_x_float[model.receptive_field:-(model.seg-1)].shape[0]
                else:
                    feat_len = batch_x_float[model.receptive_field:].shape[0]
            else:
                if model.seg > 1:
                    feat_len = batch_x_float[:-(model.seg-1)].shape[0]
                else:
                    feat_len = batch_x_float.shape[0]

            if model.lpc > 0:
                if not args.init_acc:
                    mus, bs_noclip, bs, log_bs, ass = model(batch_h, batch_x, do=True, clip=True)
                else:
                    mus, bs_noclip, bs, log_bs, ass = model(batch_h, batch_x, clip=True)
                ass = ass.flip(-1)
                init_mus = mus
                for j in range(model.seg):
                    tmp_smpls = batch_x_prob[:,j:-(model.seg-j)].unfold(1, model.lpc, 1)
                    lpc = torch.sum(ass*tmp_smpls,-1,keepdim=True)
                    if j > 0:
                        mus = torch.cat((mus, lpc+init_mus[:,:,j:j+1]),2)
                    else:
                        mus = lpc+init_mus[:,:,j:j+1]
                if model.seg == 1:
                    mus = mus.reshape(mus.shape[0], -1)
                    bs_noclip = bs_noclip.reshape(mus.shape[0], -1)
                    bs = bs.reshape(mus.shape[0], -1)
                    log_bs = log_bs.reshape(mus.shape[0], -1)
            else:
                if not args.init_acc:
                    mus, bs_noclip, bs, log_bs = model(batch_h, batch_x, do=True, clip=True)
                else:
                    mus, bs_noclip, bs, log_bs = model(batch_h, batch_x, clip=True)

            if h_ss > 0:
                mus = mus[0,model.receptive_field:]
                bs_noclip = bs_noclip[0,model.receptive_field:]
                bs = bs[0,model.receptive_field:]
                log_bs = log_bs[0,model.receptive_field:]
                batch_x_float = batch_x_float[model.receptive_field:]
            else:
                mus = mus[0]
                bs_noclip = bs_noclip[0]
                bs = bs[0]
                log_bs = log_bs[0]

            m_sum = 0
            if model.seg > 1:
                n_sum = 0
                for i in range(model.seg):
                    if i > 0:
                        i_n = i+1
                        mus_i = mus[:,i:i_n].squeeze(-1)
                        bs_noclip_i = bs_noclip[:,i:i_n].squeeze(-1)
                        if i_n < model.seg:
                            batch_x_float_i = batch_x_float[i:-(model.seg-(i_n))]
                        else:
                            batch_x_float_i = batch_x_float[i:]
                        tmp_batch_loss_laplace = criterion_laplace(mus_i, bs[:,i:i_n].squeeze(-1), \
                                                batch_x_float_i, log_b=log_bs[:,i:i_n].squeeze(-1), log=False)
                        batch_loss_laplace = torch.cat((batch_loss_laplace, \
                                                tmp_batch_loss_laplace.unsqueeze(0)))
                    else:
                        mus_i = mus[:,:1].squeeze(-1)
                        bs_noclip_i = bs_noclip[:,:1].squeeze(-1)
                        batch_x_float_i = batch_x_float[:-(model.seg-1)]
                        tmp_batch_loss_laplace = criterion_laplace(mus_i, bs[:,:1].squeeze(-1), \
                                                    batch_x_float_i, log_b=log_bs[:,:1].squeeze(-1))
                        batch_loss_laplace = tmp_batch_loss_laplace.unsqueeze(0)
                    eps = torch.empty(mus_i.shape).cuda().uniform_(-0.4999,0.5)
                    batch_output = mus_i-bs_noclip_i*eps.sign()*torch.log1p(-2*eps.abs())
                    tmp_batch_loss_err = torch.mean(torch.abs(batch_output-batch_x_float_i))
                    if i > 0:
                        batch_loss_err = torch.cat((batch_loss_err, tmp_batch_loss_err.unsqueeze(0)))
                    else:
                        batch_loss_err = tmp_batch_loss_err.unsqueeze(0)
                    if i == 0:
                        logging.info("%lf %E %lf %E" % (torch.min(batch_x_float_i), \
                        torch.mean(batch_x_float_i), torch.max(batch_x_float_i), torch.var(batch_x_float_i)))
                        logging.info("%lf %E %lf %E" % (torch.min(batch_output), torch.mean(batch_output), \
                        torch.max(batch_output), torch.var(batch_output)))
                    n = 0
                    for i in range(config.n_fft_facts):
                        if feat_len > int(fft_facts[i]/2):
                            stft_out[i] = torch.stft(batch_output, fft_facts[i], window=hann_win[i])
                            stft_trg[i] = torch.stft(batch_x_float_i, fft_facts[i], window=hann_win[i])
                            tmp_batch_stft_loss = criterion_lsd(stft_out[i], stft_trg[i], LSD=False, L2=False)
                            if not torch.isinf(tmp_batch_stft_loss) and not torch.isnan(tmp_batch_stft_loss):
                                if n > 0:
                                    tmp_batch_loss_stft_l1 = torch.cat((tmp_batch_loss_stft_l1, \
                                                                        tmp_batch_stft_loss.unsqueeze(0)))
                                else:
                                    tmp_batch_loss_stft_l1 = tmp_batch_stft_loss.unsqueeze(0)
                                n += 1
                    if n > 0:
                        if n_sum > 0:
                            batch_loss_stft_l1 = torch.cat((batch_loss_stft_l1, \
                                                            torch.mean(tmp_batch_loss_stft_l1).unsqueeze(0)))
                        else:
                            batch_loss_stft_l1 = torch.mean(tmp_batch_loss_stft_l1).unsqueeze(0)
                    n_sum += n
                    m = 0
                    for i in range(config.n_fft_facts):
                        if feat_len > int(fft_facts[i]/2):
                            tmp_batch_stft_loss = criterion_lsd(stft_out[i], stft_trg[i])
                            if not torch.isinf(tmp_batch_stft_loss) and not torch.isnan(tmp_batch_stft_loss):
                                if m > 0:
                                    tmp_batch_loss_lsd = torch.cat((tmp_batch_loss_lsd, \
                                                                    tmp_batch_stft_loss.unsqueeze(0)))
                                else:
                                    tmp_batch_loss_lsd = tmp_batch_stft_loss.unsqueeze(0)
                                m += 1
                    if m > 0:
                        if m_sum > 0:
                            batch_loss_lsd = torch.cat((batch_loss_lsd, \
                                                        torch.mean(tmp_batch_loss_lsd).unsqueeze(0)))
                        else:
                            batch_loss_lsd = torch.mean(tmp_batch_loss_lsd).unsqueeze(0)
                    m_sum += m
                batch_loss_laplace = torch.mean(batch_loss_laplace)
                batch_loss = batch_loss_laplace
                if n_sum > 0:
                    batch_loss += torch.mean(batch_loss_stft_l1)
                if m_sum > 0:
                    batch_loss_lsd = torch.mean(batch_loss_lsd)
                batch_loss_err = torch.mean(batch_loss_err)
            else:
                batch_loss_laplace = criterion_laplace(mus, bs, batch_x_float, log_b=log_bs)
                batch_loss = batch_loss_laplace
                eps = torch.empty(mus.shape).cuda().uniform_(-0.4999,0.5)
                batch_output = mus-bs_noclip*eps.sign()*torch.log1p(-2*eps.abs())
                batch_loss_err = torch.mean(torch.abs(batch_output-batch_x_float))
                logging.info("%lf %E %lf %E" % (torch.min(batch_x_float), torch.mean(batch_x_float), \
                                                torch.max(batch_x_float), torch.var(batch_x_float)))
                logging.info("%lf %E %lf %E" % (torch.min(batch_output), torch.mean(batch_output), \
                                                torch.max(batch_output), torch.var(batch_output)))
                n = 0
                for i in range(config.n_fft_facts):
                    if feat_len > int(fft_facts[i]/2):
                        stft_out[i] = torch.stft(batch_output, fft_facts[i], window=hann_win[i])
                        stft_trg[i] = torch.stft(batch_x_float, fft_facts[i], window=hann_win[i])
                        tmp_batch_stft_loss = criterion_lsd(stft_out[i], stft_trg[i], LSD=False, L2=False)
                        if not torch.isinf(tmp_batch_stft_loss) and not torch.isnan(tmp_batch_stft_loss):
                            if n > 0:
                                batch_loss_stft_l1 = torch.cat((batch_loss_stft_l1, \
                                                                tmp_batch_stft_loss.unsqueeze(0)))
                            else:
                                batch_loss_stft_l1 = tmp_batch_stft_loss.unsqueeze(0)
                            n += 1
                if n > 0:
                    batch_loss += torch.mean(batch_loss_stft_l1)
                m = 0
                for i in range(config.n_fft_facts):
                    if feat_len > int(fft_facts[i]/2):
                        tmp_batch_stft_loss = criterion_lsd(stft_out[i], stft_trg[i])
                        if not torch.isinf(tmp_batch_stft_loss) and not torch.isnan(tmp_batch_stft_loss):
                            if m > 0:
                                batch_loss_lsd = torch.cat((batch_loss_lsd, tmp_batch_stft_loss.unsqueeze(0)))
                            else:
                                batch_loss_lsd = tmp_batch_stft_loss.unsqueeze(0)
                            m += 1
                if m > 0:
                    batch_loss_lsd = torch.mean(batch_loss_lsd)

            if not args.init_acc:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            loss_err.append(batch_loss_err.item())
            loss_laplace.append(batch_loss_laplace.item())
            if (model.seg > 1 and m_sum > 0) or (model.seg == 1 and m > 0):
                loss_lsd.append(batch_loss_lsd.item())
                logging.info("batch loss %s [%d:%d] %d %d %d %d %d %d = %.3f %.3f dB %.6f (%.3f sec)" % (
                    os.path.basename(os.path.dirname(wavfile))+"/"+os.path.basename(wavfile), \
                    c_idx+1, utt_idx+1, tf, ts, h_ss, h_bs, x_ss, x_bs, batch_loss_laplace.item(), \
                    batch_loss_lsd.item(), batch_loss_err.item(), time.time() - start))
            else:
                logging.info("batch loss %s [%d:%d] %d %d %d %d %d %d = %.3f n/a %.6f (%.3f sec)" % (
                    os.path.basename(os.path.dirname(wavfile))+"/"+os.path.basename(wavfile), \
                    c_idx+1, utt_idx+1, tf, ts, h_ss, h_bs, x_ss, x_bs, batch_loss_laplace.item(), \
                    batch_loss_err.item(), time.time() - start))
            iter_idx += 1
            iter_count += 1
            total += time.time() - start

    # save final model
    model.cpu()
    torch.save({"model": model.state_dict()}, args.expdir + "/checkpoint-final.pkl")
    logging.info("final checkpoint created.")


if __name__ == "__main__":
    main()
