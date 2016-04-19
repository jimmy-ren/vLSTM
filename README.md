# vLSTM
Vectorized Long Short-term Memory (LSTM) using Matlab and GPU <br>

It supports both the regular LSTM described [here](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) and the multimodal LSTM described [here](http://www.jimmyren.com/papers/AAAI16_Ren.pdf). <br>

If you are interested, visit [here](https://github.com/jimmy-ren/lstm_speaker_naming_aaai16) for details of the experiments described in the multimodal LSTM [paper](http://www.jimmyren.com/papers/AAAI16_Ren.pdf).

## Hardware/software requirements
To run the code, you have to have a NVidia GPU with at least 4GB GPU memory. The code was tested in Ubuntu 14.04 and Windows 7 using MATLAB 2014b.

## Character level language generation
The task is the same as that in the [char-rnn](https://github.com/karpathy/char-rnn) project, which is a good indicator to show if the LSTM implementation is effective.

### Generation using a pre-trained model
Open the `applications/writer` folder but don't enter it. Run `lstm_writer_test.m` and it will start to generate. In the first a few lines of `lstm_writer_val.m` you can adjust the starting character. Currently, it starts with "I", so a typical generation is like <br>

`I can be the most programmers who would be try to them. But I was anyway that the most professors and press right. It's hard to make them things like the startups that was much their fundraising the founders who was by being worth in the side of a startup would be to be the smart with good as work with an angel round by companies and funding a lot of the partners is that they want to competitive for the top was a strange could be would be a company that was will be described startups in the paper we could probably be were the same thing that they can be some to investors...`

### Data generation and training
Paul Graham's [essay](http://www.paulgraham.com/articles.html) is used in this sample. All text is stored in `data/writer/all_text.mat` as a string. You may load it manually and see the content. The whole text contains about 2 million characters. To generate the training data, please run `data/writer/gen_char_data_from_text_2.m`. It will generate four .mat files under `data/writer/graham`, each file contains 10000 character sequences of length 50, so the four files adds upto 2 million characters.<br>

Once the data is ready, you may run `lstm_writer_train.m` under `applications/writer` to start the training. During training, intermediate models will be saved under `results/writer`. You may launch another Matlab and run `lstm_writer_test.m` with the newly saved model instead of `writer.mat` to test it.

## Multimodal LSTM for speaker naming
The training procedure of the Multimodal speaker naming LSTM as well as the pre-processed data (the one you can use off-the-shelf) has been releaseed. Please follow the instruction below to perform the training.
### Download data
Please go [here](https://drive.google.com/folderview?id=0B6nl_KFEGWG0QWVJakhRcEUyVDQ&usp=sharing) or [here](http://pan.baidu.com/s/1kV6KbOF) to download all the pre-processed training data and put all the files under `data/speaker-naming/processed_training_data/`, following the existing folder structure inside. <br>

In addition, please go [here](https://drive.google.com/folderview?id=0B6nl_KFEGWG0NkdYcEduc2twQW8&usp=sharing) or [here](http://pan.baidu.com/s/1bpymRHd) to download the pre-processed multimodal validation data and put all the files under `data/speaker-naming/raw_full/`, following the existing folder structure inside. <br>
### Start training
Once all the data is in place, you may start to train 3 types of models, namly the model only classifies the face features, the model only classifies the audio features and the model simultaneously classifies the face+audio multimodal features (multimodal LSTM). <br>

To train the face only model, you may run this [script](https://github.com/jimmy-ren/vLSTM/blob/master/applications/speaker-naming/face_only/sn_face_train.m). <br>
To train the audio only model, you may run this [script](https://github.com/jimmy-ren/vLSTM/blob/master/applications/speaker-naming/audio_only/sn_audio_train.m). <br>
To train the face+audio multimodal LSTM model, you may run this [script](https://github.com/jimmy-ren/vLSTM/blob/master/applications/speaker-naming/face_audio/sn_FA_5c_train_v52.m). <br>

Meanwhile, you can also run tests for the aforementioned three models by using the pre-train models. <br>
This [script](https://github.com/jimmy-ren/vLSTM/blob/master/applications/speaker-naming/face_only/test_face_all.m) for testing the pre-train face only model. <br>
This [script](https://github.com/jimmy-ren/vLSTM/blob/master/applications/speaker-naming/audio_only/test_audio_all.m) for testing the pre-train audio only model. <br>
This [script](https://github.com/jimmy-ren/vLSTM/blob/master/applications/speaker-naming/face_audio/test_FA_all_v52.m) for testing the pre-train face-audio multimodal LSTM model. <br>

## Citations
Jimmy SJ. Ren, Yongtao Hu, Yu-Wing Tai, Chuan Wang, Li Xu, Wenxiu Sun, Qiong Yan, 
"[Look, Listen and Learn - A Multimodal LSTM for Speaker Identification](http://www.jimmyren.com/papers/AAAI16_Ren.pdf)", The 30th AAAI Conference on Artificial Intelligence (AAAI-16). <br>

