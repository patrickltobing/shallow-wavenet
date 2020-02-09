# Shallow WaveNet Vocoder with Laplacian Distribution using Multiple Samples Output based on Linear Prediction / with Softmax Output

## Install
Environment

    $ cd tools
    $ make

Python3.7 (for Python3.6, please edit Makefile accordingly)

    $ cd ..


Main directory is 'egs/shallow-wavenet'

    $ cd egs/shallow-wavenet

Waveforms are stored in
    
    egs/shallow-wavenet/wav_<sampling_rate>

Experiment output directory will be located in

    egs/shallow-wavenet/exp

h5 files directory will be located in 

    egs/shallow-wavenet/hdf5

stats and lists directory will be located in 

    egs/shallow-wavenet/data

Logging of training will be stored in

    egs/shallow-wavenet/exp/<expdir>/log/
    or
    egs/shallow-wavenet/exp/<expdir>/train.log

Logging of decoding will be stored in

    egs/shallow-wavenet/exp/decoding/<setting>/

## Usage
open run.sh

    set stage=0 for initialize data list and speaker configs.

    set stage=init for computing F0 statistics of speakers

    set stage=1 for feature extraction

    set stage=2 for statistics of features

    set stage=3 for noise shaping

    set stage=4 for training step

    set stage=5 for decoding

    set stage=6 for restoring noise-shaped waveform

    set stage=7 for fine-tuning step

    set stage=8 for decoding of fine-tuned model

    set stage=9 for restoring noise-shaped waveform from fine-tuned model

    $ bash run.sh

1. set *fs*, *spks*, *wav\_org\_dir*, *data_name*, and *GPU\_device* variables accordingly
2. also set your design of training/development/testing sets in STAGE 0 accordingly
3. run STAGE 0init for new dataset to initialize configs. of the speakers, as well as the histograms of F0 of each speaker
4. please change the initial values of "*.f0" in "conf/" according to the calculated histograms in "exp/init\_spk\_stat/<data_directory>"
5. then, it's ready to start STAGE 2-4
6. [please set "n_jobs" variable accordingly for STAGE init/1, i.e., for doing parallel jobs of feature extraction]
7. run STAGE 7 for fine-tuning of target speaker using pretrained model from STAGE 4

set variables under "#  DECODING/FINE-TUNING SETTING" accordingly for STAGE 5 and 6 / 7 / 8 and 9

## Links
Please refer to wavenet\_trained\_example/egs/shallow-wavenet for example of script settings, speaker configs, trained model, logs, and/or synthesized waveforms
within the following link:

* [package](https://drive.google.com/open?id=18AapBApXuiiJDFUocxn1hWs2Jit7v0or)

* [ASRU 2019] (https://drive.google.com/open?id=1pBHiIj8M3jAb0N1oU_0uR4KzvvgekoXJ)

* [ICASSP 2020] (https://drive.google.com/open?id=1B1FxkUdqOxAj7PtZIbvtyyIZOPN1RwYh)

* [GitHub] (https://github.com/patrickltobing/shallow-wavenet)

## Logging

To summarize the training log for determining the possible optimum model, please edit 
"loss\_summary.sh" in the main directory accordingly, and run it.

* The contents of the output file from the "loss_summary.sh" may look something like this for continuous output with Laplacian distribution:

    262 -5.719502 (+- 1.225024) 9.539128 dB (+- 0.800174 dB) 0.002558 (+- 0.002514) -5.836762 (+- 0.987462) 9.676376 dB (+- 0.662477 dB) 0.002122 (+- 0.001677)

> The followings are their correspondences:

    262 --> number of epoch

    -5.719502 (+- 1.225024) --> negative log-likelihood (NLL) in development set
    9.539128 dB (+- 0.800174 dB) --> log-spectral distortion (LSD) in development set
    0.002558 (+- 0.002514) --> absolute-error of waveform samples in development set

    -5.836762 (+- 0.987462) --> negative log-likelihood (NLL) in training set
    9.676376 dB (+- 0.662477 dB) --> log-spectral distortion (LSD) in training set
    0.002122 (+- 0.001677) --> absolute-error of waveform samples in training set

* The contents of the output file from the "loss_summary.sh" may look something like this for discrete softmax output:

    13 1.911825 (+- 0.605316) 1.836777 (+- 0.607977)

> The followings are their correspondences:

    13 --> number of epoch

    1.911825 (+- 0.605316) --> cross-entropy in development set

    1.836777 (+- 0.607977) --> cross-entropy in training set

*all values should be lower for better*

## References

This system is based on the following papers:

1. P. L. Tobing, T. Hayashi, and T. Toda, "Investigation of Shallow WaveNet Vocoder with Laplacian Distribution Output", in Proc. IEEE ASRU, Sentosa, Singapore, Dec. 2019, pp. 176--183.

2. P. L. Tobing, Y.-C. Wu, T. Hayashi, K. Kobayashi, and T. Toda, "Efficient Shallow WaveNet Vocoder using Multiple Samples Output based on Laplacian Distribution and Linear Prediction", Accepted for ICASSP 2020.

Structures of the scripts are heavily influenced by a WaveNet implementation package in [https://github.com/kan-bayashi/PytorchWaveNetVocoder]

##

For further information/troubleshooting, please inquire to:

Patrick Lumban Tobing (Patrick)

Graduate School of Informatics, Nagoya University

patrick.lumbantobing@g.sp.m.is.nagoya-u.ac.jp
