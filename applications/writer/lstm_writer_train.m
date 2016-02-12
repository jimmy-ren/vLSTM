addpath core/
addpath utils/
addpath optimization/
addpath data/writer/graham/

clearvars -global config;
global config;
gpuDevice(1);

lstm_writer_configure();
lstm_init_v52();

count = 0;
cost_avg = 0;
epoc = 0;
points_seen = 0;
display_points = 1000;
save_points = 10000;

fprintf('%s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
for p = 1:1000
    for m = 1:4
        load(strcat('data/writer/graham/seq_', num2str(m)));
        sample_seq = config.NEW_MEM(sample_seq);
        label_seq = config.NEW_MEM(label_seq);
        perm = randperm(size(sample_seq, 3));
        sample_seq = sample_seq(:,:,perm);
        label_seq = label_seq(:,:,perm);
        
        for i = 1:size(sample_seq, 3)/config.batch_size            
            points_seen = points_seen + config.batch_size;
            
            start_idx = (i-1) * config.batch_size + 1;
            end_idx = i * config.batch_size;
            in = sample_seq(:,:,start_idx:end_idx);
            label = label_seq(:,:,start_idx:end_idx);
            
            lstm_core_v52(in, label);
            
            if(cost_avg == 0)
                cost_avg = config.cost{1};
            else
                cost_avg = (cost_avg + config.cost{1}) / 2;
            end
            
            eta = config.learning_rate / (1 + points_seen*config.decay);
            adagrad_update(eta);
            
            % display point
            if(mod(points_seen, display_points) == 0)
                count = count + 1;
                fprintf('%d ', count);
            end
            
            % save point
            if(mod(points_seen, save_points) == 0)
                fprintf('\n%s', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
                epoc = epoc + 1;
                fprintf('\nepoc %d, training avg cost: %f\n', epoc, cost_avg);
                
                lstm_writer_val();
                save_weights(strcat('results/writer/epoc', num2str(epoc), '.mat'));         
                cost_avg = 0;
            end
        end
    end
end
