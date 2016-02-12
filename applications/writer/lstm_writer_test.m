addpath core/
addpath optimization/
addpath utils/
addpath results/writer/
clearvars -global config;
global config;
gpuDevice(1);

load('results/writer/writer.mat');
config = model;
lstm_init_v52();
config.weights = model.weights;
lstm_writer_val();

