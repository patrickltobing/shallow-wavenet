#!/bin/sh


echo all_dswnv-softmax_6lyrs.txt
#awk -f proc_loss_log_resume.awk \
#    exp/tr50_dswnv-softmax_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse/train.log \
awk -f proc_loss_log.awk \
    exp/tr50_dswnv-softmax_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse/log/train.log \
        > all_dswnv-softmax_6lyrs.txt


echo pair_dswnv-softmax_6lyrs.txt
#awk -f proc_loss_log_resume.awk \
#    exp/tr50_dswnv-softmax_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse/train.log \
awk -f proc_loss_log.awk \
    exp/tr50_dswnv-softmax_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse/log/train.log \
        > pair_dswnv-softmax_6lyrs.txt


echo ft_all_tf1_dswnv-softmax_6lyrs.txt
#awk -f proc_loss_log_resume.awk \
#    exp/ft_75_dswnv-softmax_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse-VCC2TF1/train.log \
awk -f proc_loss_log.awk \
    exp/ft_75_dswnv-softmax_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse-VCC2TF1/log/fine_tune_VCC2TF1.log \
        > ft_all_tf1_dswnv-softmax_6lyrs.txt


echo ft_all_tm1_dswnv-softmax_6lyrs.txt
#awk -f proc_loss_log_resume.awk \
#    exp/ft_75_dswnv-softmax_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse-VCC2TM1/train.log \
awk -f proc_loss_log.awk \
    exp/ft_75_dswnv-softmax_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse-VCC2TM1/log/fine_tune_VCC2TM1.log \
        > ft_all_tm1_dswnv-softmax_6lyrs.txt


echo ft_pair_tf1_dswnv-softmax_6lyrs.txt
#awk -f proc_loss_log_resume.awk \
#    exp/ft_46_dswnv-softmax_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse-VCC2TF1/train.log \
awk -f proc_loss_log.awk \
    exp/ft_46_dswnv-softmax_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse-VCC2TF1/log/fine_tune_VCC2TF1.log \
        > ft_pair_tf1_dswnv-softmax_6lyrs.txt


echo ft_pair_tm1_dswnv-softmax_6lyrs.txt
#awk -f proc_loss_log_resume.awk \
#    exp/ft_46_dswnv-softmax_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse-VCC2TM1/train.log \
awk -f proc_loss_log.awk \
    exp/ft_46_dswnv-softmax_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse-VCC2TM1/log/fine_tune_VCC2TM1.log \
        > ft_pair_tm1_dswnv-softmax_6lyrs.txt


echo tf1_dswnv-softmax_6lyrs.txt
#awk -f proc_loss_log_resume.awk \
#    exp/tr50_dswnv-softmax_TF1vcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse/train.log \
awk -f proc_loss_log.awk \
    exp/tr50_dswnv-softmax_TF1vcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse/log/train.log \
        > tf1_dswnv-softmax_6lyrs.txt


echo tm1_dswnv-softmax_6lyrs.txt
#awk -f proc_loss_log_resume.awk \
#    exp/tr50_dswnv-softmax_TM1vcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse/train.log \
awk -f proc_loss_log.awk \
    exp/tr50_dswnv-softmax_TM1vcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch256_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_ep4000_cwffalse/log/train.log \
        > tm1_dswnv-softmax_6lyrs.txt


echo all_cswnv-laplacex_6lyrs.txt
#awk -f proc_loss_log_cont_resume.awk \
#    exp/tr50_cswnv-stftcmplx-laplace_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue/train.log \
awk -f proc_loss_log_cont.awk \
    exp/tr50_cswnv-stftcmplx-laplace_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue/log/train.log \
        > all_cswnv-laplace_6lyrs.txt


echo pair_cswnv-laplacex_6lyrs.txt
#awk -f proc_loss_log_cont_resume.awk \
#    exp/tr50_cswnv-stftcmplx-laplace_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue/train.log \
awk -f proc_loss_log_cont.awk \
    exp/tr50_cswnv-stftcmplx-laplace_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue/log/train.log \
        > pair_cswnv-laplace_6lyrs.txt


echo tf1_cswnv-laplacex_6lyrs.txt
#awk -f proc_loss_log_cont_resume.awk \
#    exp/tr50_cswnv-stftcmplx-laplace_TF1vcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue/train.log \
awk -f proc_loss_log_cont.awk \
    exp/tr50_cswnv-stftcmplx-laplace_TF1vcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue/log/train.log \
        > tf1_cswnv-laplace_6lyrs.txt


echo tm1_cswnv-laplacex_6lyrs.txt
#awk -f proc_loss_log_cont_resume.awk \
#    exp/tr50_cswnv-stftcmplx-laplace_TM1vcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue/train.log \
awk -f proc_loss_log_cont.awk \
    exp/tr50_cswnv-stftcmplx-laplace_TM1vcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue/log/train.log \
        > tm1_cswnv-laplace_6lyrs.txt


echo ft_all_tf1_cswnv-laplacex_6lyrs.txt
#awk -f proc_loss_log_cont_resume.awk \
#    exp/ft_55_cswnv-stftcmplx-laplace_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue-VCC2TF1/train.log \
awk -f proc_loss_log_cont.awk \
    exp/ft_55_cswnv-stftcmplx-laplace_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue-VCC2TF1/log/fine_tune_VCC2TF1.log \
        > ft_all_tf1_cswnv-laplace_6lyrs.txt


echo ft_all_tm1_cswnv-laplacex_6lyrs.txt
#awk -f proc_loss_log_cont_resume.awk \
#    exp/ft_55_cswnv-stftcmplx-laplace_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue-VCC2TM1/train.log \
awk -f proc_loss_log_cont.awk \
    exp/ft_55_cswnv-stftcmplx-laplace_14spksvcc2018arctic_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue-VCC2TM1/log/fine_tune_VCC2TM1.log \
        > ft_all_tm1_cswnv-laplace_6lyrs.txt


echo ft_pair_tf1_cswnv-laplacex_6lyrs.txt
#awk -f proc_loss_log_cont_resume.awk \
#    exp/ft_1001_cswnv-stftcmplx-laplace_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue-VCC2TF1/train.log \
awk -f proc_loss_log_cont.awk \
    exp/ft_1001_cswnv-stftcmplx-laplace_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue-VCC2TF1/log/fine_tune_VCC2TF1.log \
        > ft_pair_tf1_cswnv-laplace_6lyrs.txt


echo ft_pair_tm1_cswnv-laplacex_6lyrs.txt
#awk -f proc_loss_log_cont_resume.awk \
#    exp/ft_1001_cswnv-stftcmplx-laplace_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue-VCC2TM1/train.log \
awk -f proc_loss_log_cont.awk \
    exp/ft_1001_cswnv-stftcmplx-laplace_2spksvcc2018_22.05kHz_lr1e-4_bs8800_aux54_resch192_skipch256_dd3_dr2_ks7_auxks3_auxds2_do0.5_seg1_ep4000_fft17_lpc4_caffalse_cwftrue-VCC2TM1/log/fine_tune_VCC2TM1.log \
        > ft_pair_tm1_cswnv-laplace_6lyrs.txt


