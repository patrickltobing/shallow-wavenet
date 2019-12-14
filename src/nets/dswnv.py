# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import logging
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def encode_mu_law(x, mu=256):
    """FUNCTION TO PERFORM MU-LAW ENCODING

    Args:
        x (ndarray): audio signal with the range from -1 to 1
        mu (int): quantized level

    Return:
        (ndarray): quantized audio signal with the range from 0 to mu - 1
    """
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.int64)


def decode_mu_law(y, mu=256):
    """FUNCTION TO PERFORM MU-LAW DECODING

    Args:
        x (ndarray): quantized audio signal with the range from 0 to mu - 1
        mu (int): quantized level

    Return:
        (ndarray): audio signal with the range from -1 to 1
    """
    mu = mu - 1
    fx = (y - 0.5) / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
    return x


def initialize(m):
    """FUNCTION TO INITILIZE CONV WITH XAVIER

    Arg:
        m (torch.nn.Module): torch nn module instance
    """
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


#def OneHot(x, depth=256, training=False):
def OneHot(x, depth=256):
    """Forward calculation

    Arg:
        x (Variable): long tensor variable with the shape  (B x T)

    Return:
        (Variable): float tensor variable with the shape (B x depth x T)
    """
    x = x % depth
    x = torch.unsqueeze(x, 2)
    #x = torch.unsqueeze(x, 1)
    #x_onehot = Variable(torch.FloatTensor(x.size(0), x.size(1), depth).zero_())
    x_onehot = torch.FloatTensor(x.size(0), x.size(1), depth).zero_()
    #x_onehot = torch.FloatTensor(x.size(0), depth).zero_()
    #if training:
    #    x_onehot = Variable(torch.FloatTensor(x.size(0), x.size(1), depth).zero_())
    #else:
    #    with torch.no_grad():
    #        x_onehot = Variable(torch.FloatTensor(x.size(0), x.size(1), depth).zero_())

    if torch.cuda.is_available():
        x_onehot = x_onehot.cuda()

    #return x_onehot.scatter_(2, x, 1).transpose(1,2)
    return x_onehot.scatter_(2, x, 1)
    #return x_onehot.scatter_(1, x, 1)


class UpSampling(nn.Module):
    """UPSAMPLING LAYER WITH DECONVOLUTION

    Arg:
        upsampling_factor (int): upsampling factor
    """

    def __init__(self, upsampling_factor, bias=True):
        super(UpSampling, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.bias = bias
        self.conv = nn.ConvTranspose2d(1, 1,
                                       kernel_size=(1, self.upsampling_factor),
                                       stride=(1, self.upsampling_factor),
                                       bias=self.bias)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T')
                        where T' = T * upsampling_factor
        """
        x = x.unsqueeze(1)  # B x 1 x C x T
        x = self.conv(x)  # B x 1 x C x T'
        return x.squeeze(1)


class CausalConv1d(nn.Module):
    """1D DILATED CAUSAL CONVOLUTION"""

    def __init__(self, in_channels, out_channels, kernel_size, dil_fact=0, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dil_fact = dil_fact
        self.dilation = self.kernel_size**self.dil_fact
        self.padding = self.kernel_size**(self.dil_fact+1) - self.dilation
        self.bias = bias
        self.conv = nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, padding=self.padding, \
                                dilation=self.dilation, bias=self.bias)

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """
        return self.conv(x)[:,:,:x.shape[2]]


class TwoSidedDilConv1d(nn.Module):
    """1D TWO-SIDED DILATED CONVOLUTION with DYNAMIC DIMENSION"""

    def __init__(self, in_dim=39, kernel_size=3, layers=2):
        super(TwoSidedDilConv1d, self).__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.layers = layers
        self.rec_field = self.kernel_size**self.layers
        self.conv = nn.ModuleList()
        for i in range(self.layers):
            if i > 0:
                self.conv += [nn.Conv1d(self.in_dim*(self.kernel_size**(i)), \
                                self.in_dim*(self.kernel_size**(i+1)), self.kernel_size, stride=1, \
                                dilation=self.kernel_size**i, \
                                padding=int((self.kernel_size**(i+1)-self.kernel_size**(i))/2))]
            else:
                self.conv += [nn.Conv1d(self.in_dim, self.in_dim*self.kernel_size, self.kernel_size, \
                                stride=1, dilation=1, padding=int((self.kernel_size-1)/2))]

    def forward(self, x):
        """Forward calculation

        Arg:
            x (Variable): float tensor variable with the shape  (B x C x T)

        Return:
            (Variable): float tensor variable with the shape (B x C x T)
        """
        for i in range(self.layers):
            x = self.conv[i](x)

        return x


class DSWNV(nn.Module):
    def __init__(self, n_quantize=256, n_aux=54, hid_chn=192, skip_chn=256, aux_kernel_size=3, \
                aux_dilation_size=2, dilation_depth=3, dilation_repeat=3, kernel_size=6, \
                upsampling_factor=110, audio_in_flag=False, wav_conv_flag=False, do_prob=0):
        super(DSWNV, self).__init__()
        self.n_aux = n_aux
        self.n_quantize = n_quantize
        self.upsampling_factor = upsampling_factor
        self.in_audio_dim = self.n_quantize
        self.n_hidch = hid_chn
        self.n_skipch = skip_chn
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.aux_kernel_size = aux_kernel_size
        self.aux_dilation_size = aux_dilation_size
        self.do_prob = do_prob
        self.audio_in_flag = audio_in_flag
        self.wav_conv_flag = wav_conv_flag

        # Input Layers
        self.scale_in = nn.Conv1d(self.n_aux, self.n_aux, 1)
        self.conv_aux = TwoSidedDilConv1d(in_dim=self.n_aux, kernel_size=aux_kernel_size, \
                                            layers=aux_dilation_size)
        self.in_aux_dim = self.n_aux*self.conv_aux.rec_field
        self.upsampling = UpSampling(self.upsampling_factor)
        if self.do_prob > 0:
            self.aux_drop = nn.Dropout(p=self.do_prob)
        if not self.audio_in_flag:
            self.in_tot_dim = self.in_aux_dim
        else:
            self.in_tot_dim = self.in_aux_dim+self.in_audio_dim
        if self.wav_conv_flag:
            self.wav_conv = nn.Conv1d(self.in_audio_dim, self.n_hidch, 1)
            self.causal = CausalConv1d(self.n_hidch, self.n_hidch, self.kernel_size, dil_fact=0)
        else:
            self.causal = CausalConv1d(self.in_audio_dim, self.n_hidch, self.kernel_size, dil_fact=0)

        # Dilated Convolutional Recurrent Neural Network (DCRNN)
        self.padding = []
        self.dil_facts = [i for i in range(self.dilation_depth)] * self.dilation_repeat
        logging.info(self.dil_facts)
        self.in_x = nn.ModuleList()
        self.dil_h = nn.ModuleList()
        self.out_skip = nn.ModuleList()
        for i, d in enumerate(self.dil_facts):
            self.in_x += [nn.Conv1d(self.in_tot_dim, self.n_hidch*2, 1)]
            self.dil_h += [CausalConv1d(self.n_hidch, self.n_hidch*2, self.kernel_size, dil_fact=d)]
            self.padding.append(self.dil_h[i].padding)
            self.out_skip += [nn.Conv1d(self.n_hidch, self.n_skipch, 1)]
        logging.info(self.padding)
        self.receptive_field = sum(self.padding) + self.kernel_size-1
        logging.info(self.receptive_field)
        if self.do_prob > 0:
            self.dcrnn_drop = nn.Dropout(p=self.do_prob)

        # Output Layers
        self.out_1 = nn.Conv1d(self.n_skipch, self.n_quantize, 1)
        self.out_2 = nn.Conv1d(self.n_quantize, self.n_quantize, 1)

    def forward(self, audio, aux, do=False, last=False):
        # Input	Features
        x = self.upsampling(self.conv_aux(self.scale_in(aux)))[:,:,1:] # B x C x T
        if self.do_prob > 0 and do:
            x = self.aux_drop(x)
        if self.audio_in_flag:
            x = torch.cat((x,audio),1) # B x C x T
        # Initial Hidden Units
        if not self.wav_conv_flag:
            h = F.softsign(self.causal(audio)) # B x C x T
        else:
            h = F.softsign(self.causal(self.wav_conv(audio))) # B x C x T
        # DCRNN blocks
        sum_out, h = self._dcrnn_forward(x, h, self.in_x[0], self.dil_h[0], self.out_skip[0])
        if self.do_prob > 0 and do:
            for l in range(1,len(self.dil_facts)):
                if (l+1)%self.dilation_depth == 0:
                    out, h = self._dcrnn_forward_drop(x, h, self.in_x[l], self.dil_h[l], self.out_skip[l])
                else:
                    out, h = self._dcrnn_forward(x, h, self.in_x[l], self.dil_h[l], self.out_skip[l])
                sum_out += out
        else:
            for l in range(1,len(self.dil_facts)):
                out, h = self._dcrnn_forward(x, h, self.in_x[l], self.dil_h[l], self.out_skip[l])
                sum_out += out
        # Output
        return self.out_2(F.relu(self.out_1(F.relu(sum_out)))).transpose(1,2)

    def _dcrnn_forward_drop(self, x, h, in_x, dil_h, out_skip):
        x_h_ = in_x(x)*dil_h(h)
        z = torch.sigmoid(x_h_[:,:self.n_hidch,:])
        h = (1-z)*torch.tanh(x_h_[:,self.n_hidch:,:]) + z*h
        return out_skip(h), self.dcrnn_drop(h)

    def _dcrnn_forward(self, x, h, in_x, dil_h, out_skip):
        x_h_ = in_x(x)*dil_h(h)
        z = torch.sigmoid(x_h_[:,:self.n_hidch,:])
        h = (1-z)*torch.tanh(x_h_[:,self.n_hidch:,:]) + z*h
        return out_skip(h), h

    def _generate_dcrnn_forward(self, x, h, in_x, dil_h, out_skip):
        x_h_ = in_x(x)*dil_h(h)[:,:,-1:]
        z = torch.sigmoid(x_h_[:,:self.n_hidch,:])
        h = (1-z)*torch.tanh(x_h_[:,self.n_hidch:,:]) + z*h[:,:,-1:]
        return out_skip(h), h

    def batch_fast_generate(self, audio, aux, n_samples_list, intervals=4410):
        with torch.no_grad():
            # set max length
            max_samples = max(n_samples_list)
    
            # upsampling
            x = self.upsampling(self.conv_aux(self.scale_in(aux))) # B x C x T
    
            logging.info(x.shape)
            # padding if the length less than
            n_pad = self.receptive_field
            if n_pad > 0:
                audio = F.pad(audio, (n_pad, 0), "constant", self.n_quantize // 2)
                x = F.pad(x, (n_pad, 0), "replicate")

            logging.info(x.shape)
            audio = OneHot(audio).transpose(1,2)
            #audio = OneHot(audio)
            if not self.audio_in_flag:
                x_ = x[:, :, :audio.size(2)]
            else:
                x_ = torch.cat((x[:, :, :audio.size(2)],audio),1)
            if self.wav_conv_flag:
                audio = self.wav_conv(audio) # B x C x T
            output = F.softsign(self.causal(audio)) # B x C x T
            output_buffer = []
            buffer_size = []
            for l in range(len(self.dil_facts)):
                _, output = self._dcrnn_forward(
                    x_, output, self.in_x[l], self.dil_h[l],
                    self.out_skip[l])
                if l < len(self.dil_facts)-1:
                    buffer_size.append(self.padding[l+1])
                else:
                    buffer_size.append(self.kernel_size - 1)
                output_buffer.append(output[:, :, -buffer_size[l] - 1: -1])
    
            # generate
            samples = audio.data  # B x T
            time_sample = []
            start = time.time()
            out_idx = self.kernel_size*2-1
            for i in range(max_samples):
                start_sample = time.time()
                samples_size = samples.size(-1)
                if not self.audio_in_flag:
                    x_ = x[:, :, (samples_size-1):samples_size]
                else:
                    x_ = torch.cat((x[:, :, (samples_size-1):samples_size],samples[:,:,-1:]),1)
                output = F.softsign(self.causal(samples[:,:,-out_idx:])[:,:,-self.kernel_size:]) # B x C x T
                output_buffer_next = []
                skip_connections = []
                for l in range(len(self.dil_facts)):
                    #start_ = time.time()
                    skip, output = self._generate_dcrnn_forward(
                        x_, output, self.in_x[l], self.dil_h[l],
                        self.out_skip[l])
                    output = torch.cat((output_buffer[l], output), 2)
                    output_buffer_next.append(output[:, :, -buffer_size[l]:])
                    skip_connections.append(skip)
    
                # update buffer
                output_buffer = output_buffer_next
    
                # get predicted sample
                output = self.out_2(F.relu(self.out_1(F.relu(sum(skip_connections))))).transpose(1,2)[:,-1]

                posterior = F.softmax(output, dim=-1)
                dist = torch.distributions.OneHotCategorical(posterior)
                sample = dist.sample().data  # B
                if i > 0:
                    out_samples = torch.cat((out_samples, torch.argmax(sample, dim=--1).unsqueeze(1)), 1)
                else:
                    out_samples = torch.argmax(sample, dim=--1).unsqueeze(1)

                if self.wav_conv_flag:
                    samples = torch.cat((samples, self.wav_conv(sample.unsqueeze(2))), 2)
                else:
                    samples = torch.cat((samples, sample.unsqueeze(2)), 2)
    
                # show progress
                time_sample.append(time.time()-start_sample)
                #if intervals is not None and (i + 1) % intervals == 0:
                if (i + 1) % intervals == 0:
                    logging.info("%d/%d estimated time = %.6f sec (%.6f sec / sample)" % (
                        (i + 1), max_samples,
                        (max_samples - i - 1) * ((time.time() - start) / intervals),
                        (time.time() - start) / intervals))
                    start = time.time()
                    #break
            logging.info("average time / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % (\
                        np.mean(np.array(time_sample)), len(time_sample), \
                        1.0/(1000*np.mean(np.array(time_sample)))))
            logging.info("average throughput / sample = %.6f sec (%ld samples * %ld) [%.3f kHz/s]" % (\
                        sum(time_sample)/(len(time_sample)*len(n_samples_list)), len(time_sample), \
                        len(n_samples_list), len(time_sample)*len(n_samples_list)/(1000*sum(time_sample))))
            samples = out_samples
    
            # devide into each waveform
            samples = samples[:, -max_samples:].cpu().numpy()
            samples_list = np.split(samples, samples.shape[0], axis=0)
            samples_list = [s[0, :n_s] for s, n_s in zip(samples_list, n_samples_list)]
    
            return samples_list
