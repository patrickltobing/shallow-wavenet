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
from torch.distributions.normal import Normal
import torch.nn.functional as F


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


class CSWNV(nn.Module):
    def __init__(self, n_aux=54, hid_chn=192, skip_chn=256, aux_kernel_size=3, aux_dilation_size=2, \
            dilation_depth=3, dilation_repeat=2, kernel_size=7, upsampling_factor=110, seg=5, lpc=4, \
            do_prob=0, aux_conv2d_flag=False, wav_conv_flag=False):
        super(CSWNV, self).__init__()
        self.n_aux = n_aux
        self.n_hidch = hid_chn
        self.n_skipch = skip_chn
        self.aux_kernel_size = aux_kernel_size
        self.aux_dilation_size = aux_dilation_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.kernel_size = kernel_size
        self.upsampling_factor = upsampling_factor
        self.seg = seg
        self.lpc = lpc
        self.lpc_offset = self.seg-self.lpc
        self.do_prob = do_prob
        self.aux_conv2d_flag = aux_conv2d_flag
        self.wav_conv_flag = wav_conv_flag

        # Input Layers
        self.scale_in = nn.Conv1d(self.n_aux, self.n_aux, 1)
        self.conv_aux = TwoSidedDilConv1d(in_dim=self.n_aux, kernel_size=aux_kernel_size, \
                                            layers=aux_dilation_size)
        self.upsampling = UpSampling(self.upsampling_factor)
        if self.do_prob > 0:
            self.aux_drop = nn.Dropout(p=self.do_prob)
        self.in_aux_dim = self.n_aux*self.conv_aux.rec_field
        if self.aux_conv2d_flag and self.seg > 1:
            self.aux_conv2d = nn.Conv2d(self.in_aux_dim, self.in_aux_dim, (self.seg, 1))
        elif self.seg > 1:
            self.in_aux_dim *= self.seg
        if self.wav_conv_flag:
            self.wav_conv = nn.Conv1d(1, self.n_hidch, 1)
            self.causal = CausalConv1d(self.n_hidch, self.n_hidch, self.kernel_size, dil_fact=0)
        else:
            self.causal = CausalConv1d(1, self.n_hidch, self.kernel_size, dil_fact=0)

        # Dilated Convolutional Recurrent Neural Network (DCRNN)
        self.padding = []
        self.dil_facts = [i for i in range(self.dilation_depth)] * self.dilation_repeat
        logging.info(self.dil_facts)
        self.in_x = nn.ModuleList()
        self.dil_h = nn.ModuleList()
        self.out_skip = nn.ModuleList()
        for i, d in enumerate(self.dil_facts):
            self.in_x += [nn.Conv1d(self.in_aux_dim, self.n_hidch*2, 1)]
            self.dil_h += [CausalConv1d(self.n_hidch, self.n_hidch*2, self.kernel_size, dil_fact=d)]
            self.padding.append(self.dil_h[i].padding)
            self.out_skip += [nn.Conv1d(self.n_hidch, self.n_skipch, 1)]
        logging.info(self.padding)
        self.receptive_field = sum(self.padding) + self.kernel_size-1
        logging.info(self.receptive_field)
        if self.do_prob > 0:
            self.dcrnn_drop = nn.Dropout(p=self.do_prob)

        # Output Layers
        self.out_1 = nn.Conv1d(self.n_skipch, self.n_skipch, 1)
        self.out_2 = nn.Conv1d(self.n_skipch, 2*self.seg+self.lpc, 1)

    def forward(self, aux, audio, do=False, clip=False):
        # Input layers
        x = self.upsampling(self.conv_aux(self.scale_in(aux)))[:,:,self.seg:] #self.seg:self.seg-1 nogrp
        if self.do_prob > 0 and do:
            x = self.aux_drop(x)
        if self.aux_conv2d_flag and self.seg > 1:
            # B x C x T --> B x C x seg x T --> B x hid_chn x T
            x = self.aux_conv2d(x.unfold(2, self.seg, 1).permute(0,1,3,2)).squeeze(2)
        elif self.seg > 1:
            x = x.unfold(2, self.seg, 1).permute(0, 2, 1, 3) # B x C x T --> B x T x C_seg
            x = x.reshape(x.size(0), x.size(1), -1).permute(0, 2, 1) # B x C_seg x T

        # Initial Hidden Units
        if not self.wav_conv_flag:
            h = F.softsign(self.causal(audio)[:,:,self.seg-1:]) # B x C x T
        else:
            h = F.softsign(self.causal(self.wav_conv(audio))[:,:,self.seg-1:]) # B x C x T
        # DCRNN blocks
        if len(self.dil_facts) > 2:
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
        else:
            sum_out, h = self._dcrnn_forward_drop(x, h, self.in_x[0], self.dil_h[0], self.out_skip[0])
            out, h = self._dcrnn_forward(x, h, self.in_x[1], self.dil_h[1], self.out_skip[1])
            sum_out += out

        # Output layers
        out = self.out_2(F.relu(self.out_1(F.relu(sum_out)))).transpose(1,2)
        if self.lpc > 0:
            mu = out[:,:,:self.seg]
            log_b = F.logsigmoid(out[:,:,self.seg:2*self.seg])
            a = out[:,:,2*self.seg:]
            if clip:
                if torch.min(log_b) < -14.162084148244246758816564788835:
                    b = torch.exp(log_b) #no_clip
                    log_b = torch.clamp(log_b, min=-14.162084148244246758816564788835)
                    return mu, b, torch.exp(log_b), log_b, a
                else:
                    b = torch.exp(log_b)
                    return mu, b, b, log_b, a
            else:
                return mu, torch.exp(log_b), log_b, a
        else:
            if self.seg > 1:
                log_b = F.logsigmoid(out[:,:,self.seg:])
                if clip:
                    if torch.min(log_b) < -14.162084148244246758816564788835:
                        b = torch.exp(log_b) #no_clip
                        log_b = torch.clamp(log_b, min=-14.162084148244246758816564788835)
                        return out[:,:,:self.seg], b, torch.exp(log_b), log_b
                    else:
                        b = torch.exp(log_b)
                        return out[:,:,:self.seg], b, b, log_b
                else:
                    return out[:,:,:self.seg], torch.exp(log_b), log_b
            else:
                log_b = F.logsigmoid(out[:,:,self.seg:]).reshape(out.shape[0],-1)
                if clip:
                    if torch.min(log_b) < -14.162084148244246758816564788835:
                        b = torch.exp(log_b) #no_clip
                        log_b = torch.clamp(log_b, min=-14.162084148244246758816564788835)
                        return out[:,:,:self.seg].reshape(out.shape[0],-1), b, torch.exp(log_b), log_b
                    else:
                        b = torch.exp(log_b)
                        return out[:,:,:self.seg].reshape(out.shape[0],-1), b, b, log_b
                else:
                    return out[:,:,:self.seg].reshape(out.shape[0],-1), torch.exp(log_b), log_b

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

    def _generate_dcrnn_forward(self, x, h, in_x, dil_h, out_skip, out_sample=1):
        x_h_ = in_x(x)*dil_h(h)[:,:,-out_sample:]
        z = torch.sigmoid(x_h_[:,:self.n_hidch,:])
        h = (1-z)*torch.tanh(x_h_[:,self.n_hidch:,:]) + z*h[:,:,-out_sample:]
        return out_skip(h), h

    def batch_fast_generate(self, audio, aux, n_samples_list, intervals=4410, Laplace=True):
        with torch.no_grad():
            # set max length
            intervals = intervals//self.seg
            max_samples = max(n_samples_list)
            if self.seg > 1:
                max_samples_scale = int(max_samples/self.seg)
            else:
                max_samples_scale = max_samples
    
            x = self.upsampling(self.conv_aux(self.scale_in(aux)))
    
            logging.info(x.shape)
            n_pad = self.receptive_field
            if n_pad > 0:
                audio = F.pad(audio, (n_pad, 0), "constant", 0)
                x = F.pad(x, (n_pad, 0), "replicate")

            logging.info(x.shape)
            if self.aux_conv2d_flag and self.seg > 1:
                # B x C x T --> B x C x seg x T --> B x hid_chn x T
                x = self.aux_conv2d(x.unfold(2, self.seg, 1).permute(0,1,3,2)).squeeze(2)
            elif self.seg > 1:
                x = x.unfold(2, self.seg, 1).permute(0, 2, 1, 3) # B x C x T --> B x T x C_seg
                x = x.reshape(x.size(0), x.size(1), -1).permute(0, 2, 1) # B x C_seg x T
            logging.info(x.shape)

            audio = audio.unsqueeze(1)
            x_ = x[:,:,:(audio.size(-1)-(self.seg-1))]

            if self.lpc > 0:
                lp_x_buffer = audio[:,:,self.seg-1:]
                lp_x_buffer = lp_x_buffer[:,:,-self.lpc:]

            if self.wav_conv_flag:
                audio = self.wav_conv(audio) # B x C x T
            output = F.softsign(self.causal(audio)[:,:,self.seg-1:]) # B x C x T
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
                output_buffer.append(output[:, :, -buffer_size[l] - self.seg: -self.seg])

            # generate
            samples = audio[:,:,self.seg-1:].data  # B x T
            time_sample = []
            start = time.time()
            causal_out_idx = self.kernel_size+self.seg-1
            out_idx = causal_out_idx+self.kernel_size-1
            batch_size = x.shape[0]
            if Laplace: #Laplacian dist.
                if self.lpc > 0:
                    eps = torch.empty(batch_size,1,1).cuda()
                else:
                    eps = torch.empty(batch_size,1,self.seg).cuda()
                for i in range(max_samples_scale):
                    start_sample = time.time()
                    samples_size = samples.size(-1)
                    x_ = x[:,:,(samples_size-self.seg):samples_size]
                    output = F.softsign(self.causal(samples[:,:,-out_idx:])[:,:,-causal_out_idx:]) # B x C x T
                    output_buffer_next = []
                    skip_connections = []
                    for l in range(len(self.dil_facts)):
                        skip, output = self._generate_dcrnn_forward(
                            x_, output, self.in_x[l], self.dil_h[l],
                            self.out_skip[l], out_sample=self.seg)
                        output = torch.cat((output_buffer[l], output), 2)
                        output_buffer_next.append(output[:,:,-buffer_size[l]:])
                        skip_connections.append(skip)
    
                    # update buffer
                    output_buffer = output_buffer_next
    
                    # get predicted sample
                    out = self.out_2(F.relu(self.out_1(F.relu(sum(skip_connections))))).transpose(1,2)[:,-1:,:]
                    if self.lpc > 0:
                        mus = out[:,:,:self.seg]
                        bs = torch.exp(F.logsigmoid(out[:,:,self.seg:2*self.seg]))
                        ass = out[:,:,2*self.seg:].flip(-1)
                        lpc = torch.sum(ass*lp_x_buffer,-1,keepdim=True)
                        eps.uniform_(-0.4999,0.5)
                        #output = lpc+mus[:,:,:1] - bs[:,:,:1]*eps.sign()*torch.log1p(-2*eps.abs())
                        output = torch.clamp(lpc+mus[:,:,:1] \
                                    - bs[:,:,:1]*eps.sign()*torch.log1p(-2*eps.abs()), min=-1, max=1)
                        lp_x_buffer = torch.cat((lp_x_buffer[:,:,1:],output[:,:,-1:]),2)
                        for j in range(1,self.seg):
                            lpc = torch.sum(ass*lp_x_buffer,-1,keepdim=True)
                            eps.uniform_(-0.4999,0.5)
                            #output = torch.cat((output, lpc+mus[:,:,j:j+1] \
                            #                - bs[:,:,j:j+1]*eps.sign()*torch.log1p(-2*eps.abs())),2)
                            output = torch.cat((output, torch.clamp(lpc+mus[:,:,j:j+1] \
                                    - bs[:,:,j:j+1]*eps.sign()*torch.log1p(-2*eps.abs()), min=-1, max=1)),2)
                            lp_x_buffer = torch.cat((lp_x_buffer[:,:,1:],output[:,:,-1:]),2)
                    else:
                        eps.uniform_(-0.4999,0.5)
                        #output = out[:,:,:self.seg] \
                        #    - torch.exp(F.logsigmoid(out[:,:,self.seg:]))*eps.sign()*torch.log1p(-2*eps.abs())
                        output = torch.clamp(out[:,:,:self.seg] - torch.exp(F.logsigmoid(out[:,:,self.seg:]))\
                                    *eps.sign()*torch.log1p(-2*eps.abs()), min=-1, max=1)
                    output = output.reshape(batch_size,1,-1)

                    if i > 0:
                        out_samples = torch.cat((out_samples, output.squeeze(1)), 1)
                    else:
                        out_samples = output.squeeze(1)

                    if self.wav_conv_flag:
                        samples = torch.cat((samples, self.wav_conv(output)), 2)
                    else:
                        samples = torch.cat((samples, output), 2)
    
                    # show progress
                    time_sample.append(time.time()-start_sample)
                    if intervals is not None and (i + 1) % intervals == 0:
                    #if (i + 1)*self.sampling_scale % intervals == 0:
                        logging.info("%d/%d estimated time = %.6f sec (%.6f sec / sample)" % (
                            (i + 1)*self.seg, max_samples,
                            (((max_samples - (i - 1)*self.seg) * ((time.time() - start) \
                                / (intervals)))/self.seg),
                            (time.time() - start) / (intervals*self.seg)))
                        start = time.time()
                        #break

            time_sample = np.array(time_sample)/self.seg
            logging.info("average time / sample = %.6f sec (%ld samples) [%.3f kHz/s]" % (\
                            np.mean(time_sample), len(time_sample)*self.seg, 1.0/(1000*np.mean(time_sample))))
            logging.info("average throughput / sample = %.6f sec (%ld samples * %ld) [%.3f kHz/s]" % (\
                            np.sum(time_sample)/(len(time_sample)*len(n_samples_list)), \
                            len(time_sample)*self.seg, len(n_samples_list), \
                            len(time_sample)*len(n_samples_list)/(1000*np.sum(time_sample))))
            samples = out_samples

            # devide into each waveform
            samples = samples[:, -max_samples:].cpu().numpy()
            samples_list = np.split(samples, samples.shape[0], axis=0)
            samples_list = [s[0, :n_s] for s, n_s in zip(samples_list, n_samples_list)]
    
            return samples_list


class LaplaceLoss(nn.Module):
    def __init__(self):
        super(LaplaceLoss, self).__init__()
        self.c = 0.69314718055994530941723212145818 # ln(2)

    def forward(self, mu, b, target, log_b=None, clip=False, log=True):
        if log_b is None:
            if clip and torch.min(b) < 7.0710678118654752440084436210504e-7:
                b = torch.clamp(b, min=7.0710678118654752440084436210504e-7)
            log_b = torch.log(b)
        elif clip:
            if torch.min(log_b) < -14.162084148244246758816564788835:
                log_b = torch.clamp(log_b, min=-14.162084148244246758816564788835)
                b = torch.exp(log_b)

        if log:
            var = 2*(b**2)
            logging.info("%lf %E %lf %E %E %E %E" % (torch.min(mu), torch.mean(mu), torch.max(mu), \
                            torch.var(mu), torch.min(var), torch.mean(var), torch.max(var)))
        nll = self.c + log_b + torch.abs(target-mu)/b # neg_log_like (Laplace)
        return torch.mean(nll)


class LSDloss(nn.Module):
    def __init__(self):
        super(LSDloss, self).__init__()

    def forward(self, x, y, LSD=True, L2=True):
        if LSD:
            if L2:
                pow_x = torch.sum(x**2, -1)
                pow_y = torch.sum(y**2, -1)
                lsd = torch.sqrt(torch.mean((10*(torch.log10(pow_x)-torch.log10(pow_y)))**2, 0))
            else:
                pow_x = torch.sum(x**2, -1)
                pow_y = torch.sum(y**2, -1)
                lsd = (pow_x-pow_y)**2
        else:
            if L2:
                lsd = (x-y)**2
            else:
                lsd = torch.abs(x-y)

        return torch.mean(lsd)
