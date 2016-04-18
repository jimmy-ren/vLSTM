addpath core/
addpath utils/
addpath optimization/
addpath data/speaker-naming/processed_training_data/train_audio/
addpath data/speaker-naming/processed_training_data/val_audio/

clearvars -global config;
global config mem;
gpuDevice(1);
sn_audio_configure();
sn_audio_init();
lstm_init_v52();

count = 0;
cost_avg = 0;
epoc = 0;
points_seen = 0;
display_points = 1000;
save_points = 10000;

load('data/speaker-naming/processed_training_data/train_audio/1');
samples = reshape(samples, size(samples,1), []);
config.data_mean = mean(samples,2);
config.one_over_data_std = 1 ./ std(samples')';

load('data/speaker-naming/processed_training_data/val_audio/1');
test_samples = test_samples(:,:,1:2000);
test_labels = test_labels(:,1:2000);
test_labels = reshape(test_labels, size(test_labels,1), 1, size(test_labels,2));
test_labels = repmat(test_labels, [1 size(test_samples,2) 1]);
test_samples = config.NEW_MEM(test_samples);
test_labels = config.NEW_MEM(test_labels);
test_samples = bsxfun(@times, bsxfun(@minus, test_samples, config.data_mean), config.one_over_data_std);


fprintf('%s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
for p = 1:100
    for m = 1:11
        load(strcat('data/speaker-naming/processed_training_data/train_audio/', num2str(m)));
        labels = reshape(labels, size(labels,1), 1, size(labels,2));
        labels = repmat(labels, [1 size(samples,2) 1]);
        perm = randperm(size(labels, 3));
        samples = samples(:,:,perm);
        labels = labels(:,:,perm);
        
        samples = config.NEW_MEM(samples);
        labels = config.NEW_MEM(labels);
        samples = bsxfun(@times, bsxfun(@minus, samples, config.data_mean), config.one_over_data_std);
        
        for i = 1:size(samples, 3)/config.batch_size
            points_seen = points_seen + config.batch_size;
            
            start_idx = config.batch_size * (i-1) + 1;
            end_idx = start_idx + config.batch_size - 1;

            in = samples(:,:,start_idx:end_idx);
            label = labels(:,:,start_idx:end_idx);
            
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
                
                correct_num = 0;                
                train_correct_num = 0;
                for ii = 1:size(test_samples, 3)/config.batch_size
                    start_idx = config.batch_size * (ii-1) + 1;
                    end_idx = start_idx + config.batch_size - 1;

                    val_sample = test_samples(:,:,start_idx:end_idx);
                    val_label = test_labels(:,:,start_idx:end_idx);                    
                    
                    lstm_core_v52(val_sample, 1);                    
                    
                    [value, estimated_labels] = max(mem.net_out(:,end,:));
                    [value, true_labels] = max(val_label(:,end,:));
                    correct_num = correct_num + length(find(estimated_labels == true_labels));
                    
                    
                    val_sample = samples(:,:,start_idx:end_idx);
                    val_label = labels(:,:,start_idx:end_idx);
                    
                    lstm_core_v52(val_sample, 1);                    
                    
                    [value, estimated_labels] = max(mem.net_out(:,end,:));
                    [value, true_labels] = max(val_label(:,end,:));
                    train_correct_num = train_correct_num + length(find(estimated_labels == true_labels));
                end                
                
                acc = correct_num / size(test_samples, 3);
                train_acc = train_correct_num / size(test_samples, 3);
                fprintf('\nepoc %d, training avg cost: %f, train_acc: %.2f%%, val_acc: %.2f%%\n', epoc, cost_avg, train_acc*100, acc*100);
                
                model = config;
                save(strcat('results/speaker-naming/audio_only/', num2str(epoc), '.mat'), '-v7.3', 'model');
                
                cost_avg = 0;
            end
        end
    end
end





