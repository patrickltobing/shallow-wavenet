#!/bin/bash
#######################################################################################
#   SCRIPT FOR SHALLOW WAVENET VOCODER (SWNV) (Discrete-Softmax/Continuous-Laplace)   #
#######################################################################################

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# USER SETTINGS {{{
#######################################
#           STAGE SETTING             #
#######################################
# {{{
# 0: initialize speaker data list
# init: calculate F0 histograms of speakers
# 1: feature extraction step
# 2: statistics calculation step
# 3: apply noise shaping step
# 4: training step
# 5: decoding step
# 6: restore noise shaping step
# 7: fine-tune step
# 8: decoding fine-tune step
# 9: restore noise shaping fine-tune step
# }}}
#stage=0
#stage=init
#stage=0init
#stage=1
#stage=2
#stage=3
#stage=0123
#stage=23
#stage=01234
#stage=4
#stage=56
#stage=5
#stage=6
#stage=7
#stage=89
#stage=8
#stage=9

#######################################
#          FEATURE SETTING            #
#######################################
# {{{
# shiftms: shift length in msec (default=5)
# fftl: fft length (default=1024)
# highpass_cutoff: highpass filter cutoff frequency (if 0, will not apply)
# mcep_dim: dimension of mel-cepstrum
# mcep_alpha: alpha value of mel-cepstrum
# mag: coefficient of noise shaping (default=0.5)
# n_jobs: number of parallel jobs
# }}}
shiftms=5
#fftl=512
fftl=1024
#fftl=2048
highpass_cutoff=70
#fs=8000 ## 8kHz not yet supported, WORLD fails to analyze and synthesize properly
#fs=16000
fs=22050
#fs=24000
#fs=44100
#fs=48000
upsampling_factor=`echo "${shiftms} * ${fs} / 1000" | bc`
## total dimension of mcep = 1+mcep_dim (including 0th power)
#mcep_dim=24
#mcep_dim=34
mcep_dim=49
mag=0.5
##number of parallel jobs for feature extraction
#n_jobs=1
#n_jobs=10
#n_jobs=25
n_jobs=36
#n_jobs=40
#n_jobs=41
#n_jobs=45
n_jobs=50

#######################################
#          TRAINING SETTING           #
#######################################
# {{{
# spks: target spekaers in vcc2018
# n_quantize: number of quantization
# n_aux: number of aux features
# n_hidch: number of hidden (highway) channels
# n_skipch: number of skip channels
# dilation_depth: dilation depth (e.g. if set 10, max dilation = 2^(10-1))
# dilation_repeat: number of dilation repeats
# kernel_size: kernel size of dilated convolution
# lr: learning rate
# iters: number of iterations
# batch_size: batch size
# }}}

spks=(VCC2SF1 VCC2SM1 VCC2SF2 VCC2SM2 bdl VCC2SF3 VCC2SM3 VCC2TM1 VCC2TF1 VCC2TM2 VCC2TF2 slt VCC2SF4 VCC2SM4)
spks=(VCC2TF1 VCC2TM1)
#spks=(VCC2TF1)
#spks=(VCC2TM1)

data_name=14spksvcc2018arctic
data_name=2spksvcc2018
#data_name=TF1vcc2018
#data_name=TM1vcc2018

# uv and log-f0 occupied the first two dimensions
# then codeap and mcep
## [uv,log-f0,codeap,mcep]
if [ $fs -eq 22050 ]; then
    stdim=4 # 2-dim code-aperiodicity
    wav_org_dir=wav_22.05kHz
    data_name=${data_name}_22.05kHz
    mcep_alpha=0.455 #22.05k ## frequency warping based on pysptk.util.mcepalpha
elif [ $fs -eq 24000 ]; then
    stdim=5 # 3-dim code-aperiodicity 
    wav_org_dir=wav_24kHz
    data_name=${data_name}_24kHz
    mcep_alpha=0.466 #24k
elif [ $fs -eq 48000 ]; then
    stdim=7 # 5-dim code-aperiodicity
    wav_org_dir=wav_48kHz
    data_name=${data_name}_48kHz
    mcep_alpha=0.554 #48k
elif [ $fs -eq 44100 ]; then
    stdim=7 # 5-dim code-aperiodicity
    wav_org_dir=wav_44.1kHz
    data_name=${data_name}_44.1kHz
    mcep_alpha=0.544 #44.1k
elif [ $fs -eq 8000 ]; then ## 8kHz not yet supported, WORLD fails to analyze and synthesize properly
    stdim=2
    wav_org_dir=wav_8kHz
    data_name=${data_name}_8kHz
    mcep_alpha=0.312 #8k
else
    stdim=3 #16k: 1-dim code-aperiodicity
    wav_org_dir=wav_16kHz
    data_name=${data_name}_16kHz
    mcep_alpha=0.41000000000000003 #16k
fi
## from WORLD: number of code-aperiodicities = min(15000,fs/2-3000)/3000
## [https://github.com/mmorise/World/blob/master/src/codec.cpp] line 212

powmcep_dim=`expr ${mcep_dim} + 1`

trn=tr${powmcep_dim}_${data_name}
dev=dv${powmcep_dim}_${data_name}
tst=ts${powmcep_dim}_${data_name}

n_aux=`expr ${stdim} + ${powmcep_dim}`

model="laplace" #continuous (cswnv)
#model="softmax" #discrete (dswnv)

n_quantize=256 #mu-law dim. for softmax

if [ $model == "laplace" ]; then
    n_hidch=192 #laplace 6 layers
elif [ $model == "softmax" ]; then
    n_hidch=256 #softmax 6 layers
fi

n_skipch=256

# total layers = dilation_depth x dilation_repeat
dilation_depth=3
dilation_repeat=2

# kernel_size for causal dilated convolutions
# receptive_field = (sum of padding in one dilation_depth)*dilation_repeat + kernel_size
# for kernel_size=7, dilation_depth=3, and dilation_repeat=2
# receptive_field = (6+42+294)*2 + 7 = 342*2 + 7 = 691
kernel_size=7

## kernel and dilation settings for auxiliary/conditioning input convolution layers.
## -4/+4 frames for kernel=3 and dilation=2 (receptive field is kernel**dilation),
## with out_dim of 1st dilation layer = 3*in_dim,
## and out_dim of 2nd dilation layer = 9*in_dim
aux_kernel_size=3
aux_dilation_size=2

## number of samples in an output_segment for multiple samples output [ICASSP 2020]
seg=1
#seg=2
#seg=5
#seg=10

lr=1e-4

#do_prob=0
do_prob=0.5

# maximum number of epoch
epoch_count=4000

# number of samples in a batch
# these defaults are equal to 80 frames
# 16kHz --> 80*80 = 6400; 22.05kHz --> 80*110 = 8800; 24kHz --> 80*120 = 9600
#batch_size=6400 #16k
batch_size=8800 #22.05k
#batch_size=9600 #24k

# number of FFT windowing configurations in continuous model
# each configurations details are actually written in the training script of continuous model (cswnv)
#n_fft_facts=5
#n_fft_facts=9
n_fft_facts=17

# number of LP coefficients used in continuous model [ICASSP 2020]
# time-varying LP coefficients are estimated along with distribution parameters, i.e., data-driven
#lpc=0
lpc=4

# use 2d-convolution for pre-processing aux. features
#caf=true
caf=false

# use 1d-convolution for pre-processing audio sample
if [ $model == "laplace" ]; then
    cwf=true # for continuous, set true
elif [ $model == "softmax" ]; then
    cwf=false # for discrete, set false
fi

GPU_device=0
#GPU_device=1
#GPU_device=2

string_path="/feat_org_lf0"

#n_gpus=1
#n_gpus=2
n_gpus=3

#######################################
#     DECODING/FINE-TUNING SETTING    #
#######################################
min_idx=55
min_idx=1001
#min_idx=1865
#min_idx=2168
#min_idx=75
#min_idx=46
#min_idx=50
#min_idx=45

min_idx_ft=398
min_idx_ft=472
min_idx_ft=398
min_idx_ft=473
#min_idx_ft=55
#min_idx_ft=48
#min_idx_ft=20
#min_idx_ft=17

#idx_resume=

spk_trg=VCC2TF1
spk_trg=VCC2TM1

#spks_trg=(VCC2TF1 VCC2TM1)
spks_trg_ft=(VCC2TF1)
spks_trg_ft=(VCC2TM1)
#spks_trg_ft=(VCC2TF1 VCC2TM1)

init_acc=false
#init_acc=true

#decode_batch_size=1
#decode_batch_size=2
#decode_batch_size=4
#decode_batch_size=5
decode_batch_size=7
#decode_batch_size=9
#decode_batch_size=18

#GPU_device_str="0,1"
GPU_device_str="0,1,2"
GPU_device_str="2,1,0"
GPU_device_str="1,2,0"
GPU_device_str="1,0,2"
#GPU_device_str="2"
#GPU_device_str="1,2"

# parse options
. parse_options.sh

echo $model $data_name $string_path $GPU_device

# stop when error occured
set -e
# }}}


# STAGE 0 {{{
if [ `echo ${stage} | grep 0` ];then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    mkdir -p data/${trn}
    mkdir -p data/${dev}
    mkdir -p data/${tst}
    [ -e data/${trn}/wav.scp ] && rm data/${trn}/wav.scp
    [ -e data/${dev}/wav.scp ] && rm data/${dev}/wav.scp
    [ -e data/${tst}/wav.scp ] && rm data/${tst}/wav.scp
    for spk in ${spks[@]};do
        if [ -n "$(echo $spk | sed -n 's/\(VCC2\)/\1/p')" ]; then
            echo a $spk
            # first 10 utts. to dev set
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n +11 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 10 >> data/${dev}/wav.scp
            find ${wav_org_dir}/eval/${spk} -name "*.wav" | sort >> data/${tst}/wav.scp
        elif [ -n "$(echo $spk | sed -n 's/.*\(bdl.*\)/\1/p')" ] \
                || [ -n "$(echo $spk | sed -n 's/.*\(slt.*\)/\1/p')" ]; then
            echo b $spk
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | head -n 791 >> data/${trn}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 340 | head -n 170 >> data/${dev}/wav.scp
            find ${wav_org_dir}/${spk} -name "*.wav" \
                | sort | tail -n 170 >> data/${tst}/wav.scp
        fi
        spk_f0_conf_file="conf/${spk}.f0"
        if [ ! -r ${spk_f0_conf_file} ]; then
            echo 40 700 > ${spk_f0_conf_file}
            echo "F0 config of ${spk} is initialized. Please run stage init, then change accordingly."
        fi
    done
fi
# }}}


# STAGE init {{{
if [ `echo ${stage} | grep "init"` ];then
    echo "###########################################################"
    echo "#               INIT FEATURE EXTRACTION STEP              #"
    echo "###########################################################"
    if true; then
    #if false; then
        # extract feat and wav_anasyn src_speaker
        nj=0
        expdir=exp/feature_extract/${trn}_init
        mkdir -p $expdir
        for spk in ${spks[@]}; do
            echo $spk
            scp=${expdir}/wav_${spk}.scp
            n_wavs_trn=`cat data/${trn}/wav.scp | grep "\/${spk}\/" | wc -l`
            n_wavs_dev=`cat data/${dev}/wav.scp | grep "\/${spk}\/" | wc -l`
            n_wavs=`expr ${n_wavs_trn} + ${n_wavs_dev}`
            echo $n_wavs $n_wavs_trn $n_wavs_dev
            if [ $n_wavs -gt 0 ]; then
                if [ $n_wavs_trn -gt 0 ]; then
                    cat data/${trn}/wav.scp | grep "\/${spk}\/" > ${scp}
                    if [ $n_wavs_dev -gt 0 ]; then
                        cat data/${dev}/wav.scp | grep "\/${spk}\/" >> ${scp}
                    fi
                else
                    cat data/${dev}/wav.scp | grep "\/${spk}\/" > ${scp}
                fi
                ${train_cmd} --num-threads ${n_jobs} ${expdir}/feature_extract_${spk}.log \
                    feature_extract.py \
                        --expdir exp/feature_extract \
                        --waveforms ${scp} \
                        --hdf5dir hdf5/${trn}_init/${spk} \
                        --fs ${fs} \
                        --shiftms ${shiftms} \
                        --mcep_dim ${mcep_dim} \
                        --mcep_alpha ${mcep_alpha} \
                        --fftl ${fftl} \
                        --highpass_cutoff ${highpass_cutoff} \
                        --init true \
                        --n_jobs ${n_jobs}
        
                # check the number of feature files
                n_feats=`find hdf5/${trn}_init/${spk} -name "*.h5" | wc -l`
                echo "${n_feats}/${n_wavs} files are successfully processed."

                # update job counts
                nj=$(( ${nj}+1  ))
                if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                    wait
                    nj=0
                fi
            fi
        done
    fi
    find hdf5/${trn}_init -name "*.h5" | sort > data/${trn}/feats_init.scp
    echo "###########################################################"
    echo "#              INIT SPEAKER STATISTICS STEP               #"
    echo "###########################################################"
    expdir=exp/init_spk_stat/${trn}
    mkdir -p $expdir
    if true; then
    #if false; then
        for spk in ${spks[@]};do
            echo $spk
            cat data/${trn}/feats_init.scp | grep \/${spk}\/ > data/${trn}/feats_init_spk-${spk}.scp
            ${train_cmd} ${expdir}/init_stat_spk-${spk}.log \
                spk_stat.py \
                    --expdir ${expdir} \
                    --feats data/${trn}/feats_init_spk-${spk}.scp \
                    --spkr ${spk}
        done
        echo "Init. spk statistics are successfully calculated. Please change the initial values accordingly"
    fi
fi
# }}}


# STAGE 1 {{{
if [ `echo ${stage} | grep 1` ];then
    echo "###########################################################"
    echo "#               FEATURE EXTRACTION STEP                   #"
    echo "###########################################################"
    nj=0
    if true; then
    #if false; then
        for set in ${trn} ${dev} ${tst};do
            echo $set
            for spk in ${spks[@]};do
                echo $spk
                [ ! -e exp/feature_extract/${set} ] && mkdir -p exp/feature_extract/${set}
                # make scp of each speaker
                scp=exp/feature_extract/${set}/wav.${spk}.scp
                n_wavs=`cat data/${set}/wav.scp | grep "\/${spk}\/" | wc -l`
                echo $n_wavs
                if [ $n_wavs -gt 0 ]; then
                    cat data/${set}/wav.scp | grep "\/${spk}\/" > ${scp}

                    # set f0 range 
                    minf0=`cat conf/${spk}.f0 | awk '{print $1}'`
                    maxf0=`cat conf/${spk}.f0 | awk '{print $2}'`

                    # feature extract
                    ${train_cmd} --num-threads ${n_jobs} \
                        exp/feature_extract/feature_extract_${set}.${spk}.log \
                        feature_extract.py \
                            --expdir exp/feature_extract \
                            --waveforms ${scp} \
                            --wavdir wav_filtered/${set}/${spk} \
                            --wavanasyndir wav_anasyn/${set}/${spk} \
                            --hdf5dir hdf5/${set}/${spk} \
                            --fs ${fs} \
                            --shiftms ${shiftms} \
                            --minf0 ${minf0} \
                            --maxf0 ${maxf0} \
                            --mcep_dim ${mcep_dim} \
                            --mcep_alpha ${mcep_alpha} \
                            --highpass_cutoff ${highpass_cutoff} \
                            --fftl ${fftl} \
                            --n_jobs ${n_jobs} & 

                    # update job counts
                    nj=$(( ${nj}+1  ))
                    if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                        wait
                        nj=0
                    fi
                fi
            #fi
            done
            wait

            # check the number of feature files
            n_wavs=`cat data/${set}/wav.scp | wc -l`
            n_feats=`find hdf5/${set} -name "*.h5" | wc -l`
            echo "${n_feats}/${n_wavs} files are successfully processed."

            # make scp files
            find wav_filtered/${set} -name "*.wav" | sort > data/${set}/wav_filtered.scp
            find hdf5/${set} -name "*.h5" | sort > data/${set}/feats.scp
        done
    fi
fi
# }}}



# STAGE 2 {{{
if [ `echo ${stage} | grep 2` ];then
    echo "###########################################################"
    echo "#              CALCULATE STATISTICS STEP                  #"
    echo "###########################################################"
    cat data/${trn}/feats.scp > data/${trn}/feats_all.scp
    cat data/${dev}/feats.scp >> data/${trn}/feats_all.scp
    ${train_cmd} exp/calculate_statistics/calc_stats_${trn}.log \
        calc_stats.py \
            --feats data/${trn}/feats_all.scp \
            --stats data/${trn}/stats.h5
    echo "statistics are successfully calculated."
fi
# }}}



# STAGE 3 {{{
if [ `echo ${stage} | grep 3` ];then
    echo "###########################################################"
    echo "#                   NOISE SHAPING STEP                    #"
    echo "###########################################################"
    if true; then
    #if false; then
        nj=0
        [ ! -e exp/noise_shaping ] && mkdir -p exp/noise_shaping
        for set in ${trn} ${dev} ${tst};do
        #for set in ${trn} ${dev};do
        #for set in ${trn} ${tst};do
        #for set in ${dev};do
        #for set in ${tst};do
            echo $set
            for spk in ${spks[@]};do
                echo $spk
                # make scp of each speaker
                scp=exp/noise_shaping/wav_filtered.${set}.${spk}.scp
                n_wavs=`cat data/${set}/wav_filtered.scp | grep "\/${spk}\/" | wc -l`
                echo $n_wavs
                if [ $n_wavs -gt 0 ]; then
                    cat data/${set}/wav_filtered.scp | grep "\/${spk}\/" > ${scp}
            
                    # apply noise shaping
                    ${train_cmd} --num-threads ${n_jobs} \
                        exp/noise_shaping/noise_shaping_apply.${set}.${spk}.log \
                        noise_shaping.py \
                            --waveforms ${scp} \
                            --stats data/${trn}/stats.h5 \
                            --writedir wav_ns/${set}/${spk} \
                            --fs ${fs} \
                            --shiftms ${shiftms} \
                            --fftl ${fftl} \
                            --mcep_dim_start ${stdim} \
                            --mcep_alpha ${mcep_alpha} \
                            --mag ${mag} \
                            --inv true \
                            --n_jobs ${n_jobs} & 

                    # update job counts
                    nj=$(( ${nj}+1  ))
                    if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
                        wait
                        nj=0
                    fi
                fi
            done
            wait
            # check the number of feature files
            n_wavs=`cat data/${set}/wav_filtered.scp | wc -l`
            n_ns=`find wav_ns/${set} -name "*.wav" | wc -l`
            echo "${n_ns}/${n_wavs} files are successfully processed."

            # make scp files
            find wav_ns/${set} -name "*.wav" | sort > data/${set}/wav_ns.scp
        done
    fi
fi
# }}}


# STAGE 4 {{{
# set variables
spk_list="$(IFS=_; echo "${spks[*]}")"

if [ $model == "laplace" ];then
    setting=cswnv-stftcmplx-${model}_${data_name}_lr${lr}_bs${batch_size}_aux${n_aux}_resch${n_hidch}_skipch${n_skipch}_dd${dilation_depth}_dr${dilation_repeat}_ks${kernel_size}_auxks${aux_kernel_size}_auxds${aux_dilation_size}_do${do_prob}_seg${seg}_ep${epoch_count}_fft${n_fft_facts}_lpc${lpc}_caf${caf}_cwf${cwf}
elif [ $model == "softmax" ]; then
    setting=dswnv-${model}_${data_name}_lr${lr}_bs${batch_size}_aux${n_aux}_resch${n_hidch}_skipch${n_skipch}_dd${dilation_depth}_dr${dilation_repeat}_ks${kernel_size}_auxks${aux_kernel_size}_auxds${aux_dilation_size}_do${do_prob}_ep${epoch_count}_cwf${cwf}
fi

expdir=exp/tr${powmcep_dim}_${setting}
if [ `echo ${stage} | grep 4` ];then
    echo "###########################################################"
    echo "#               WAVENET TRAINING STEP                     #"
    echo "###########################################################"
    echo $expdir
    waveforms=data/${trn}/wav_ns.scp
    waveforms_eval=data/${dev}/wav_ns.scp

    if [ $model == "laplace" ];then
        #${cuda_cmd} ${expdir}/log/train_resume-${idx_resume}.log \
        ${cuda_cmd} ${expdir}/log/train.log \
            train_cswnv_laplace-stftcmplx_shift1.py \
                --waveforms ${waveforms} \
                --waveforms_eval ${waveforms_eval} \
                --feats data/${trn}/feats.scp \
                --feats_eval data/${dev}/feats.scp \
                --stats data/${trn}/stats.h5 \
                --expdir ${expdir} \
                --lr ${lr} \
                --do_prob ${do_prob} \
                --epoch_count ${epoch_count} \
                --batch_size ${batch_size} \
                --upsampling_factor ${upsampling_factor} \
                --n_aux ${n_aux} \
                --kernel_size ${kernel_size} \
                --dilation_depth ${dilation_depth} \
                --dilation_repeat ${dilation_repeat} \
                --aux_kernel_size ${aux_kernel_size} \
                --aux_dilation_size ${aux_dilation_size} \
                --seg ${seg} \
                --skip_chn ${n_skipch} \
                --hid_chn ${n_hidch} \
                --n_fft_facts ${n_fft_facts} \
                --string_path ${string_path} \
                --lpc ${lpc} \
                --aux_conv2d_flag ${caf} \
                --GPU_device ${GPU_device} \
                --wav_conv_flag ${cwf}
                #--resume ${expdir}/checkpoint-${idx_resume}.pkl \
        echo "done"
    elif [ $model == "softmax" ]; then
        #${cuda_cmd} ${expdir}/log/train_resume-${idx_resume}.log \
        ${cuda_cmd} ${expdir}/log/train.log \
            train_dswnv_softmax.py \
                --waveforms ${waveforms} \
                --waveforms_eval ${waveforms_eval} \
                --feats data/${trn}/feats.scp \
                --feats_eval data/${dev}/feats.scp \
                --stats data/${trn}/stats.h5 \
                --expdir ${expdir} \
                --lr ${lr} \
                --do_prob ${do_prob} \
                --epoch_count ${epoch_count} \
                --batch_size ${batch_size} \
                --upsampling_factor ${upsampling_factor} \
                --n_aux ${n_aux} \
                --kernel_size ${kernel_size} \
                --dilation_depth ${dilation_depth} \
                --dilation_repeat ${dilation_repeat} \
                --aux_kernel_size ${aux_kernel_size} \
                --aux_dilation_size ${aux_dilation_size} \
                --skip_chn ${n_skipch} \
                --hid_chn ${n_hidch} \
                --n_quantize ${n_quantize} \
                --string_path ${string_path} \
                --GPU_device ${GPU_device} \
                --wav_conv_flag ${cwf}
                #--resume ${expdir}/checkpoint-${idx_resume}.pkl \
    fi
fi
# }}}


# STAGE 5 {{{
if [ `echo ${stage} | grep 5` ];then
    echo "###########################################################"
    echo "#               WAVENET DECODING STEP                     #"
    echo "###########################################################"
    echo ${setting}
    outdir=${expdir}/wav_${model}_${data_name}_${min_idx}
    checkpoint=${expdir}/checkpoint-${min_idx}.pkl
    config=${expdir}/model.conf

    wavs=data/${tst}/wav_ns.scp
    feats=data/${tst}/feats.scp

    [ ! -e exp/decoding/${setting} ] && mkdir -p exp/decoding/${setting}
    #nj=0
    #for spk in ${spks[@]};do
        # make scp of each speaker
        scp=exp/decoding/${setting}/feats_${min_idx}_${spk_trg}.scp
        wav_scp=exp/decoding/${setting}/wavs_${min_idx}_${spk_trg}.scp
        cat $feats | grep "\/${spk_trg}\/" > ${scp}
        cat $wavs | grep "\/${spk_trg}\/" > ${wav_scp}

        # decode
        if [ $model == "laplace" ];then
            ${cuda_cmd} exp/decoding/${setting}/decode_${min_idx}_${spk_trg}.log \
                decode_cswnv_laplace-shift1.py \
                    --feats ${scp} \
                    --outdir ${outdir}/${spk_trg} \
                    --checkpoint ${checkpoint} \
                    --config ${config} \
                    --fs ${fs} \
                    --batch_size ${decode_batch_size} \
                    --n_gpus ${n_gpus} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device}
        elif [ $model == "softmax" ]; then
            ${cuda_cmd} exp/decoding/${setting}/decode_${min_idx}_${spk_trg}.log \
                decode_dswnv_softmax.py \
                    --feats ${scp} \
                    --outdir ${outdir}/${spk_trg} \
                    --checkpoint ${checkpoint} \
                    --config ${config} \
                    --fs ${fs} \
                    --batch_size ${decode_batch_size} \
                    --n_gpus ${n_gpus} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device}
        fi

        ## update job counts
        #nj=$(( ${nj}+1  ))
        #if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
        #    wait
        #    nj=0
        #fi
    #done
    #wait
fi
# }}}


# STAGE 6 {{{
if [ `echo ${stage} | grep 6` ];then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    outdir=${expdir}/wav_${model}_${data_name}_${min_idx}
    [ ! -e exp/noise_shaping/${setting} ] && mkdir -p exp/noise_shaping/${setting}
    #nj=0
    #for spk in ${spks[@]};do
        # make scp of each speaker
        scp=exp/noise_shaping/${setting}/wav_generated_${min_idx}_${spk_trg}.scp
        find ${outdir}/${spk_trg} -name "*.wav" | grep "\/${spk_trg}\/" | sort > ${scp}

        # restore noise shaping
        ${train_cmd} --num-threads ${n_jobs} \
            exp/noise_shaping/${setting}/noise_shaping_restore_${min_idx}_${spk_trg}.log \
            noise_shaping.py \
                --waveforms ${scp} \
                --stats data/${trn}/stats.h5 \
                --writedir ${outdir}_restored/${spk_trg} \
                --fs ${fs} \
                --shiftms ${shiftms} \
                --fftl ${fftl} \
                --mcep_dim_start ${stdim} \
                --mcep_alpha ${mcep_alpha} \
                --mag ${mag} \
                --inv false \
                --n_jobs ${n_jobs}

        # update job counts
        #nj=$(( ${nj}+1  ))
        #if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
        #    wait
        #    nj=0
        #fi
    #done
    #wait
fi
# }}}

pretraindir=${expdir}

spk_trg_list="$(IFS=_; echo "${spks_trg_ft[*]}")"

if ${init_acc};then
    expdir=exp/ft_init_${min_idx}_${setting}-${spk_trg_list}
else
    expdir=exp/ft_${min_idx}_${setting}-${spk_trg_list}
fi

# STAGE 7 {{{
if [ `echo ${stage} | grep 7` ];then
    echo $min_idx ${spk_trg_ft[@]}
    pretrained=${pretraindir}/checkpoint-${min_idx}.pkl
    config=${pretraindir}/model.conf
    echo ${pretraindir}
    echo "############################################################"
    echo "#               WAVENET FINE-TUNE STEP                     #"
    echo "############################################################"
    echo ${expdir}
    mkdir -p ${expdir} 

    feats=data/${trn}/feats.scp
    feats_eval=data/${dev}/feats.scp
    waveforms=data/${trn}/wav_ns.scp
    waveforms_eval=data/${dev}/wav_ns.scp

    feats_scp=${expdir}/feats_tr_${spk_trg_list}.scp
    feats_eval_scp=${expdir}/feats_ev_${spk_trg_list}.scp
    waveforms_scp=${expdir}/waveforms_tr_${spk_trg_list}.scp
    waveforms_eval_scp=${expdir}/waveforms_ev_${spk_trg_list}.scp
    rm -fr ${feats_scp} ${feats_eval_scp} ${waveforms_scp} ${waveforms_eval_scp}
    for spk in ${spks_trg_ft[@]}; do
        cat ${feats} | grep "\/${spk}\/" >> ${feats_scp}
        cat ${waveforms} | grep "\/${spk}\/" >> ${waveforms_scp}
        cat ${feats_eval} | grep "\/${spk}\/" >> ${feats_eval_scp}
        cat ${waveforms_eval} | grep "\/${spk}\/" >> ${waveforms_eval_scp}
    done

    if [ $model == "laplace" ];then
        ${cuda_cmd} ${expdir}/log/fine_tune_${spk_trg_list}.log \
            fine-tune_cswnv_laplace-stftcmplx_shift1.py \
                --waveforms ${waveforms_scp} \
                --waveforms_eval ${waveforms_eval_scp} \
                --feats ${feats_scp} \
                --feats_eval ${feats_eval_scp} \
                --expdir ${expdir} \
                --pretrained ${pretrained} \
                --config ${config} \
                --string_path ${string_path} \
                --GPU_device ${GPU_device} \
                --init_acc ${init_acc}
                #--resume ${expdir}/checkpoint-${idx_resume}.pkl \
    elif [ $model == "softmax" ];then
        ${cuda_cmd} ${expdir}/log/fine_tune_${spk_trg_list}.log \
            fine-tune_dswnv_softmax.py \
                --waveforms ${waveforms_scp} \
                --waveforms_eval ${waveforms_eval_scp} \
                --feats ${feats_scp} \
                --feats_eval ${feats_eval_scp} \
                --expdir ${expdir} \
                --pretrained ${pretrained} \
                --config ${config} \
                --init_acc ${init_acc} \
                --string_path ${string_path} \
                --GPU_device ${GPU_device} \
                --init_acc ${init_acc}
                #--resume ${expdir}/checkpoint-${idx_resume}.pkl \
    fi
fi
# }}}

# STAGE 8 {{{
if [ `echo ${stage} | grep 8` ];then
    echo "######################################################################"
    echo "#               FINE-TUNED WAVENET DECODING STEP                     #"
    echo "######################################################################"
    echo $expdir
    outdir=${expdir}/wav_${min_idx}-${min_idx_ft}
    checkpoint=${expdir}/checkpoint-${min_idx_ft}.pkl
    config=${expdir}/model.conf

    feats=data/${tst}/feats.scp

    [ ! -e exp/decoding/${setting} ] && mkdir -p exp/decoding/${setting}
    #nj=0
    #for spk in ${spks[@]};do
        # make scp of each speaker
        scp=exp/decoding/${setting}/feats_${min_idx}-${min_idx_ft}_${spk_trg}.scp
        cat $feats | grep "\/${spk_trg}\/" > ${scp}

        # decode
        if [ $model == "laplace" ];then
            ${cuda_cmd} exp/decoding/${setting}/decode_${min_idx}-${min_idx_ft}_${spk_trg}.log \
                decode_cswnv_laplace-shift1.py \
                    --feats ${scp} \
                    --outdir ${outdir}/${spk_trg} \
                    --checkpoint ${checkpoint} \
                    --config ${config} \
                    --fs ${fs} \
                    --batch_size ${decode_batch_size} \
                    --n_gpus ${n_gpus} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device}
        elif [ $model == "softmax" ];then
            ${cuda_cmd} exp/decoding/${setting}/decode_${min_idx}-${min_idx_ft}_${spk_trg}.log \
                decode_dswnv_softmax.py \
                    --feats ${scp} \
                    --outdir ${outdir}/${spk_trg} \
                    --checkpoint ${checkpoint} \
                    --config ${config} \
                    --fs ${fs} \
                    --batch_size ${decode_batch_size} \
                    --n_gpus ${n_gpus} \
                    --GPU_device_str ${GPU_device_str}
                    #--GPU_device ${GPU_device}
        fi

        # update job counts
        #nj=$(( ${nj}+1  ))
        #if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
        #    wait
        #    nj=0
        #fi
    #done
    #wait
fi
# }}}


# STAGE 9 {{{
if [ `echo ${stage} | grep 9` ];then
    echo "###########################################################"
    echo "#             RESTORE NOISE SHAPING STEP                  #"
    echo "###########################################################"
    outdir=${expdir}/wav_${min_idx}-${min_idx_ft}
    [ ! -e exp/noise_shaping/${setting} ] && mkdir -p exp/noise_shaping/${setting}
    nj=0
    #for spk in ${spks[@]};do
        # make scp of each speaker
        scp=exp/noise_shaping/${setting}/wav_generated_${min_idx}-${min_idx_ft}_${spk_trg}.scp
        find ${outdir}/${spk_trg} -name "*.wav" | grep "\/${spk_trg}\/" | sort > ${scp}

        # restore noise shaping
        ${train_cmd} --num-threads ${n_jobs} \
            exp/noise_shaping/${setting}/noise_shaping_restore_${min_idx}-${min_idx_ft}_${spk_trg}.log \
            noise_shaping.py \
                --waveforms ${scp} \
                --stats data/${train}/stats.h5 \
                --writedir ${outdir}_restored/${spk_trg} \
                --fs ${fs} \
                --shiftms ${shiftms} \
                --fftl ${fftl} \
                --mcep_dim_start ${stdim} \
                --mcep_alpha ${mcep_alpha} \
                --mag ${mag} \
                --inv false \
                --n_jobs ${n_jobs}

        # update job counts
        #nj=$(( ${nj}+1  ))
        #if [ ! ${max_jobs} -eq -1 ] && [ ${max_jobs} -eq ${nj} ];then
        #    wait
        #    nj=0
        #fi
    #done
    #wait
fi
# }}}
