addpath utils/
addpath applications/writer/

clearvars -global config;
global config;
gpuDevice(1);

lstm_writer_configure();
lstm_init_v52();

in = config.NEW_MEM(rand(config.input_size, config.max_time_steps, config.batch_size));
in2 = reshape(in, size(in,1), size(in,2)*size(in,3));
for m = 1:size(in2, 2)    
    in2(randi(config.input_size),m) = 1;
end
in = reshape(in2, size(in));
label = config.NEW_MEM(zeros(config.output_size, config.max_time_steps, config.batch_size));
label2 = reshape(label, size(label,1), size(label,2)*size(label,3));
for m = 1:size(label2, 2)    
    label2(randi(config.output_size),m) = 1;
end
label = reshape(label2, size(label));

lstm_core_v52(in, label);
computeNumericalGradient(in, label, 1, 1);


