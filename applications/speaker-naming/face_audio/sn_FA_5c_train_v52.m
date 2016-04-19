addpath core/
addpath utils/
addpath optimization/
addpath data/speaker-naming/processed_training_data/train_face/
addpath data/speaker-naming/processed_training_data/train_audio/

clearvars -global config;
global config mem;
gpuDevice(1);
sn_FA_configure();
lstm_init_v52();

solid_idx = [1,4,8,12,16,19,23,27,31,34,38,42,46];
%null_idx = [2:3,5:7,9:11,13:15,17:18,20:22,24:26,28:30,32:33,35:37,39:41,43:45,47:49];

count = 0;
cost_avg = 0;
epoc = 0;
points_seen = 0;
display_points = 1000;
save_points = 10000;

% training data
load('data/speaker-naming/processed_training_data/train_face/1.mat');
face_labels = labels;
tmp = reshape(samples, size(samples,1), []);
config.data_mean = mean(tmp,2);
config.one_over_data_std = 1 ./ std(tmp')'; clear tmp;
face_samples = zeros(53, 49, size(samples,3));
face_samples(:,solid_idx,:) = samples;
for t = 1:size(face_samples,2)
    if(face_samples(1,t,1) == 0)
        face_samples(:,t,:) = face_samples(:,t-1,:);
    end
end


load('data/speaker-naming/processed_training_data/train_audio/1');
samples = reshape(samples, size(samples,1), []);
config.data_mean = cat(1, config.data_mean, mean(samples,2));
config.one_over_data_std = cat(1, config.one_over_data_std, 1 ./ std(samples')');

% test data
load('data/speaker-naming/raw_full/test/5classes/1');   % normal test data
test_samples = test_samples(:,:,1:2000);
test_labels = test_labels(:,1:2000);
test_labels = reshape(test_labels, size(test_labels,1), 1, size(test_labels,2));
test_labels = repmat(test_labels, [1 size(test_samples,2) 1]);
test_samples = config.NEW_MEM(test_samples);
test_labels = config.NEW_MEM(test_labels);
test_samples = bsxfun(@times, bsxfun(@minus, test_samples, config.data_mean), config.one_over_data_std);

load('data/speaker-naming/raw_full/test/5classes/6');   % ourliers (face and audio does not belong to the same person)
outlier_test_samples = outlier_test_samples(:,:,1:2000);
outlier_test_samples = config.NEW_MEM(outlier_test_samples);
outlier_test_samples = bsxfun(@times, bsxfun(@minus, outlier_test_samples, config.data_mean), config.one_over_data_std);


fprintf('%s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
for p = 1:100
    for m = 1:11
        load(strcat('data/speaker-naming/processed_training_data/train_audio/', num2str(m)));
        labels_ = labels;
        labels = reshape(labels, size(labels,1), 1, size(labels,2));
        labels = repmat(labels, [1 size(samples,2) 1]);
        perm = randperm(size(labels, 3));
        samples = samples(:,:,perm);
        labels = labels(:,:,perm);
        labels_ = labels_(:,perm);
        
        samples = config.NEW_MEM(samples);
        labels = config.NEW_MEM(labels);
        
        samples = padarray(samples, [53 0 0], 'pre');        
        
        % match the face and audio training data
        for c = 1:5
            audio_idx = find(labels_(c,:) == 1);            
            face_idx = find(face_labels(c,:) == 1);
            selected_face_samples = face_samples(:,:,face_idx);
            
            if(length(face_idx) < length(audio_idx))
                perm = randperm(length(audio_idx) - length(face_idx));
                selected_face_samples = cat(3, selected_face_samples, selected_face_samples(:,:,perm));
            end
                
            perm = randperm(size(selected_face_samples, 3));
            selected_face_samples = selected_face_samples(:,:,perm);
            
            
            selected_face_samples = selected_face_samples(:,:,1:length(audio_idx));
            samples(1:53,:,audio_idx) = selected_face_samples;
        end
        
        samples = bsxfun(@times, bsxfun(@minus, samples, config.data_mean), config.one_over_data_std);
        
        for i = 1:size(samples, 3)/config.batch_size
            points_seen = points_seen + config.batch_size;
            
            start_idx = config.batch_size * (i-1) + 1;
            end_idx = start_idx + config.batch_size - 1;

            in = samples(:,:,start_idx:end_idx);
            label = labels(:,:,start_idx:end_idx);            
            
            lstm_core_v52(in, label);
            
            if(cost_avg == 0)
                cost_avg = config.cost{1} + config.cost{2};
            else
                cost_avg = (cost_avg + config.cost{1} + config.cost{2}) / 2;
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
                outlier_currect_num = 0;
                outlier_thres = 10;
                for ii = 1:size(test_samples, 3)/config.batch_size
                    start_idx = config.batch_size * (ii-1) + 1;
                    end_idx = start_idx + config.batch_size - 1;
                    
                    % outlier rejection acc
                    val_sample = outlier_test_samples(:,:,start_idx:end_idx);
                    
                    lstm_core_v52(val_sample, 1);
                    
                    [vv1, pos1] = max(mem.net_out(:,25:end,:,1));
                    [vv2, pos2] = max(mem.net_out(:,25:end,:,2));
                    tt = sum((pos1 == pos2));
                    tt = reshape(tt, 1, config.batch_size);                    
                    estimated_labels = zeros(1, config.batch_size);
                    estimated_labels(tt < outlier_thres) = -1;  % if less than 'outlier_thres' outputs agrees with each other, an outlier
                    true_labels = zeros(1, config.batch_size);
                    true_labels = true_labels - 1;
                    outlier_currect_num = outlier_currect_num + length(find(estimated_labels == true_labels));
                    
                    
                    % test acc
                    val_sample = test_samples(:,:,start_idx:end_idx);
                    val_label = test_labels(:,:,start_idx:end_idx);
                    
                    lstm_core_v52(val_sample, 1);
                    
                    [value, estimated_labels] = max(mem.net_out(:,end,:,1)+mem.net_out(:,end,:,2));
                    [vv1, pos1] = max(mem.net_out(:,25:end,:,1));
                    [vv2, pos2] = max(mem.net_out(:,25:end,:,2));
                    tt = sum((pos1 == pos2));                    
                    estimated_labels(tt < outlier_thres) = -1;  % to compute the real accuracy, apply outliear rejection first
                    [value, true_labels] = max(val_label(:,end,:));
                    correct_num = correct_num + length(find(estimated_labels == true_labels));
                    
                    
                    % training acc
                    val_sample = samples(:,:,start_idx:end_idx);
                    val_label = labels(:,:,start_idx:end_idx);                    
                    
                    lstm_core_v52(val_sample, 1);
                    
                    [value, estimated_labels] = max(mem.net_out(:,end,:,1)+mem.net_out(:,end,:,2));
                    [vv1, pos1] = max(mem.net_out(:,25:end,:,1));
                    [vv2, pos2] = max(mem.net_out(:,25:end,:,2));
                    tt = sum((pos1 == pos2));                    
                    estimated_labels(tt < outlier_thres) = -1;  % to compute the real accuracy, apply outliear rejection first
                    [value, true_labels] = max(val_label(:,end,:));
                    train_correct_num = train_correct_num + length(find(estimated_labels == true_labels));
                end
                
                acc = correct_num / size(test_samples, 3);
                train_acc = train_correct_num / size(test_samples, 3);
                outlier_acc = outlier_currect_num / size(test_samples, 3);
                
                fprintf('\nepoc %d, training avg cost: %f, train_acc: %.2f%%, val_acc: %.2f%%, outlier_acc: %.2f%%\n', epoc, cost_avg, train_acc*100, acc*100, outlier_acc*100);
                
                model = config;
                save(strcat('results/speaker-naming/face_audio/', num2str(epoc), '.mat'), '-v7.3', 'model');
                
                cost_avg = 0;
            end
        end
    end
end





