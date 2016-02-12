function lstm_init_v52()
    global config mem;
    
    config.hidden_layer_num = length(config.hidden_layer_size);
    
    config.slide_pieces = length(config.slice_points) + 1;
    slice_points = [0, config.slice_points, config.input_size];
    slice_len = zeros(1, length(slice_points)-1);
    for m = 1:config.slide_pieces
        slice_len(m) = slice_points(m+1) - slice_points(m);
    end    
    config.slide_endpoints = slice_points;
    
    
    config.NEW_MEM = @to_gpu;
    % input weights
    config.weights.Whg = {};
    config.weights.Wxg = {};
    % input gate weights
    config.weights.Whi = {};
    config.weights.Wxi = {};
    % output gate weights
    config.weights.Who = {};
    config.weights.Wxo = {};
    % forget gate weights
    config.weights.Whf = {};
    config.weights.Wxf = {};
    
    % bias
    config.weights.Bi = {};
    config.weights.Bf = {};
    config.weights.Bo = {};
    config.weights.Bg = {};
    
    config.weights.Wy = {};
    
    % one layer
    r = config.weight_range;    
    

%     two layers
%     r = 0.08;
%     b_offset = 0.3;
    
    % two layers
%     r = 0.28;
%     b_offset = 0.1;
    

    config.weights.Whg{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.hidden_layer_size(1))*2*r-r);
    config.weights.Whi{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.hidden_layer_size(1))*2*r-r);
    config.weights.Who{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.hidden_layer_size(1))*2*r-r);
    config.weights.Whf{1} = config.NEW_MEM(rand(config.hidden_layer_size(1),config.hidden_layer_size(1))*2*r-r);
    for s = 1:config.slide_pieces
        config.weights.Wxg{1}{s} = config.NEW_MEM(rand(config.hidden_layer_size(1),slice_len(s))*2*r-r);
        config.weights.Wxi{1}{s} = config.NEW_MEM(rand(config.hidden_layer_size(1),slice_len(s))*2*r-r);
        config.weights.Wxo{1}{s} = config.NEW_MEM(rand(config.hidden_layer_size(1),slice_len(s))*2*r-r);
        config.weights.Wxf{1}{s} = config.NEW_MEM(rand(config.hidden_layer_size(1),slice_len(s))*2*r-r);
        
        config.weights.Bi{1}{s} = config.NEW_MEM(rand(config.hidden_layer_size(1), 1)*2*r-r) + config.b_offset_input_gate;
        config.weights.Bf{1}{s} = config.NEW_MEM(rand(config.hidden_layer_size(1), 1)*2*r-r) + config.b_offset_forget_gate;
        config.weights.Bo{1}{s} = config.NEW_MEM(rand(config.hidden_layer_size(1), 1)*2*r-r) + config.b_offset_output_gate;
        config.weights.Bg{1}{s} = config.NEW_MEM(rand(config.hidden_layer_size(1), 1));
    end
    
    for L = 2:config.hidden_layer_num
        config.weights.Whg{L} = config.NEW_MEM(rand(config.hidden_layer_size(L),config.hidden_layer_size(L))*2*r-r);
        config.weights.Whi{L} = config.NEW_MEM(rand(config.hidden_layer_size(L),config.hidden_layer_size(L))*2*r-r);
        config.weights.Who{L} = config.NEW_MEM(rand(config.hidden_layer_size(L),config.hidden_layer_size(L))*2*r-r);
        config.weights.Whf{L} = config.NEW_MEM(rand(config.hidden_layer_size(L),config.hidden_layer_size(L))*2*r-r);
        
        for s = 1:config.slide_pieces
            config.weights.Wxg{L}{s} = config.NEW_MEM(rand(config.hidden_layer_size(L),config.hidden_layer_size(L-1))*2*r-r);
            config.weights.Wxi{L}{s} = config.NEW_MEM(rand(config.hidden_layer_size(L),config.hidden_layer_size(L-1))*2*r-r);
            config.weights.Wxo{L}{s} = config.NEW_MEM(rand(config.hidden_layer_size(L),config.hidden_layer_size(L-1))*2*r-r);
            config.weights.Wxf{L}{s} = config.NEW_MEM(rand(config.hidden_layer_size(L),config.hidden_layer_size(L-1))*2*r-r);

            config.weights.Bi{L}{s} = config.NEW_MEM(rand(config.hidden_layer_size(L), 1)*2*r-r) + config.b_offset_input_gate;
            config.weights.Bf{L}{s} = config.NEW_MEM(rand(config.hidden_layer_size(L), 1)*2*r-r) + config.b_offset_forget_gate;
            config.weights.Bo{L}{s} = config.NEW_MEM(rand(config.hidden_layer_size(L), 1)*2*r-r) + config.b_offset_output_gate;
            config.weights.Bg{L}{s} = config.NEW_MEM(rand(config.hidden_layer_size(L), 1));
        end
    end
    % output weights    
    config.weights.Wy = config.NEW_MEM(rand(config.output_size,config.hidden_layer_size(1))*2*r-r);
    
    
    mem.cell_in = {};
    mem.in_gate = {};
    mem.out_gate = {};
    mem.forget_gate = {};
    mem.cell_state = {};
    mem.cell_acts = {}; % activation of the lstm unit
    mem.net_acts = 0;
    mem.net_out = 0;
    
    % for each layer
    for L = 1:config.hidden_layer_num
        for s = 1:config.slide_pieces
            mem.cell_in{L}{s} = config.NEW_MEM(zeros(size(config.weights.Whg{L},1), config.batch_size, config.max_time_steps));
            mem.in_gate{L}{s} = config.NEW_MEM(zeros(size(mem.cell_in{L}{s})));
            mem.out_gate{L}{s} = config.NEW_MEM(zeros(size(mem.cell_in{L}{s})));
            mem.forget_gate{L}{s} = config.NEW_MEM(zeros(size(mem.cell_in{L}{s})));
            mem.cell_state{L}{s} = config.NEW_MEM(zeros(size(mem.cell_in{L}{s})));
            mem.cell_acts{L}{s} = config.NEW_MEM(zeros(size(mem.cell_in{L}{s})));
        end
    end
    
    % for training only
    mem.delta_x = {};
    mem.delta_h_prev = {};
    mem.partial_h_ratio = {};
    for L = 2:config.hidden_layer_num+1
        %if(L > 1)
        for s = 1:config.slide_pieces
            mem.delta_x{L}{s} = config.NEW_MEM(zeros(config.hidden_layer_size(L-1), config.batch_size, config.max_time_steps));
        end
        %end
        %mem.delta_h_prev{L} = config.NEW_MEM(zeros(config.hidden_layer_size(L), config.batch_size, config.max_time_steps+1));
        %mem.partial_h_ratio{L} = config.NEW_MEM(ones(config.hidden_layer_size(L), config.batch_size, config.max_time_steps+1));
    end
    %mem.delta_x{L+1} = config.NEW_MEM(zeros(config.hidden_layer_size(L), config.batch_size, config.max_time_steps)); % before output
    
    
    % input gradients
    mem.grad.Whg = {};
    mem.grad.Wxg = {};
    % input gate gradients
    mem.grad.Whi = {};
    mem.grad.Wxi = {};
    % output gate gradients
    mem.grad.Who = {};
    mem.grad.Wxo = {};
    % forget gate gradients
    mem.grad.Whf = {};
    mem.grad.Wxf = {};
    
    % bias gradients
    mem.grad.Bi = {};
    mem.grad.Bf = {};
    mem.grad.Bo = {};
    mem.grad.Bg = {};
    
    %mem.grad.Wy = {};
    
    for L = 1:config.hidden_layer_num
        mem.grad.Whg{L} = config.NEW_MEM(zeros(size(config.weights.Whg{L})));
        mem.grad.Whi{L} = config.NEW_MEM(zeros(size(config.weights.Whi{L})));
        mem.grad.Who{L} = config.NEW_MEM(zeros(size(config.weights.Who{L})));
        mem.grad.Whf{L} = config.NEW_MEM(zeros(size(config.weights.Whf{L})));
        for s = 1:config.slide_pieces
            mem.grad.Wxg{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxg{L}{s})));
            mem.grad.Wxi{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxi{L}{s})));
            mem.grad.Wxo{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxo{L}{s})));
            mem.grad.Wxf{L}{s} = config.NEW_MEM(zeros(size(config.weights.Wxf{L}{s})));
            mem.grad.Bi{L}{s} = config.NEW_MEM(zeros(config.hidden_layer_size(L), 1));
            mem.grad.Bf{L}{s} = config.NEW_MEM(zeros(config.hidden_layer_size(L), 1));
            mem.grad.Bo{L}{s} = config.NEW_MEM(zeros(config.hidden_layer_size(L), 1));
            mem.grad.Bg{L}{s} = config.NEW_MEM(zeros(config.hidden_layer_size(L), 1));
        end
    end
    mem.grad.Wy = config.NEW_MEM(zeros(size(config.weights.Wy)));
    
    adagrad_init();
end





