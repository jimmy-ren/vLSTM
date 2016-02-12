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

Once the data is ready, you may run `lstm_writer_train.m` under `applications/writer` to start the training.

## More applications using multimodal LSTM
TBA


## Citations
Jimmy SJ. Ren, Yongtao Hu, Yu-Wing Tai, Chuan Wang, Li Xu, Wenxiu Sun, Qiong Yan, 
"[Look, Listen and Learn - A Multimodal LSTM for Speaker Identification](http://www.jimmyren.com/papers/AAAI16_Ren.pdf)", The 30th AAAI Conference on Artificial Intelligence (AAAI-16). <br>

