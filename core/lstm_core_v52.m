function lstm_core_v52(in, label, previous_cell_acts, previous_cell_state)
    % in is a n by t by b matrix, where n is the input data dimension
    % t is the number of input samples (time steps)
    % b is batch size    
    
    % add support for slicing inputs (cross modality weight sharing mode, Whx, Whi, Who, Whg and Wy)
    
    global config mem;
    batch_size = size(in, 3); %config.batch_size;
    gen_net_out();
    % forward pass
    for L = 1:config.hidden_layer_num    % for each hidden layer
        for s = 1:config.slide_pieces    % for each modality
            x = get_x(L, s, in, batch_size);
            
            cell_in_in = pagefun(@mtimes, repmat(config.weights.Wxg{L}{s}, [1 1 size(in,2)]), x);
            in_gate_in = pagefun(@mtimes, repmat(config.weights.Wxi{L}{s}, [1 1 size(in,2)]), x);
            forget_gate_in = pagefun(@mtimes, repmat(config.weights.Wxf{L}{s}, [1 1 size(in,2)]), x);
            out_gate_in = pagefun(@mtimes, repmat(config.weights.Wxo{L}{s}, [1 1 size(in,2)]), x);
            
            for t = 1:size(in,2)    % for each time step
                if exist('previous_cell_acts', 'var')
                    [hprev, csprev] = get_hprev_csprev(t, L, s, batch_size, previous_cell_acts, previous_cell_state);
                else
                    [hprev, csprev] = get_hprev_csprev(t, L, s, batch_size);
                end
                mem.cell_in{L}{s}(:,1:batch_size,t) = tanh(bsxfun(@plus, config.weights.Whg{L} * hprev + cell_in_in(:,:,t), config.weights.Bg{L}{s}));
                mem.in_gate{L}{s}(:,1:batch_size,t) = sigmoid(bsxfun(@plus, config.weights.Whi{L} * hprev + in_gate_in(:,:,t), config.weights.Bi{L}{s}));
                mem.forget_gate{L}{s}(:,1:batch_size,t) = sigmoid(bsxfun(@plus, config.weights.Whf{L} * hprev + forget_gate_in(:,:,t), config.weights.Bf{L}{s}));
                mem.out_gate{L}{s}(:,1:batch_size,t) = sigmoid(bsxfun(@plus, config.weights.Who{L} * hprev + out_gate_in(:,:,t), config.weights.Bo{L}{s}));

                mem.cell_state{L}{s}(:,1:batch_size,t) = mem.cell_in{L}{s}(:,1:batch_size,t) .*  mem.in_gate{L}{s}(:,1:batch_size,t) + csprev .* mem.forget_gate{L}{s}(:,1:batch_size,t);
                mem.cell_acts{L}{s}(:,1:batch_size,t) = tanh(mem.cell_state{L}{s}(:,1:batch_size,t)) .* mem.out_gate{L}{s}(:,1:batch_size,t);
            end
        end
    end
    
    %dm = config.NEW_MEM(uint8(rand(size(mem.cell_acts{L}))));
    %mem.cell_acts{L} = mem.cell_acts{L} .* dm;
    
    % softmax
    for s = 1:config.slide_pieces
        mem.net_out(:,:,1:batch_size,s) = permute(softmax(pagefun(@mtimes, repmat(config.weights.Wy, [1 1 size(in,2)]), mem.cell_acts{config.hidden_layer_num}{s}(:,1:batch_size,:)), config.temperature), [1 3 2]);
    end   
        
    % numel of in is different from numel of label, indicates it's test time
    if(numel(in) ~= numel(label))
        return;
    end
    
    % cross entropy error
    label(:,1:size(in,2)-config.output_valid_len,:) = 0;
    cost_arr = cell(1, config.slide_pieces);
    config.cost = {};
    for s = 1:config.slide_pieces
        cost_arr{s} = sum(-sum(label(:,size(in,2)-config.output_valid_len+1:end,:) .* log(mem.net_out(:,size(in,2)-config.output_valid_len+1:end,:,s))), 3) / batch_size;
        config.cost{s} = sum(cost_arr{s}); % overall cost
    end
    
    % backprop
    set_grad_to_zeros_v52();
    for s = 1:config.slide_pieces
        delta_out_arr = permute((mem.net_out(:,:,:,s) - label) / batch_size, [1 3 2]);
        mem.delta_x{config.hidden_layer_num+1}{s} = pagefun(@mtimes, repmat(config.weights.Wy',1,1,config.max_time_steps), delta_out_arr);
        mem.grad.Wy = mem.grad.Wy + sum(pagefun(@mtimes, delta_out_arr, permute(mem.cell_acts{config.hidden_layer_num}{s}, [2 1 3])), 3);
    end
    
    gclip = 5;
    %tic
    for L = config.hidden_layer_num:-1:1        
        pad_states = config.NEW_MEM(zeros(config.hidden_layer_size(L), config.batch_size));
        for s = 1:config.slide_pieces
            gWo_mul = mem.delta_x{L+1}{s} .* tanh(mem.cell_state{L}{s}) .* deri_sigmoid(mem.out_gate{L}{s});

            dEdCt = mem.delta_x{L+1}{s} .* mem.out_gate{L}{s} .* deri_tanh(tanh(mem.cell_state{L}{s}));
            dCtdf = cat(3, pad_states, mem.cell_state{L}{s}(:,:,1:size(mem.cell_state{L}{s},3)-1)) .* deri_sigmoid(mem.forget_gate{L}{s});
            dCtdi = mem.cell_in{L}{s} .* deri_sigmoid(mem.in_gate{L}{s});
            dCtdg = mem.in_gate{L}{s} .* deri_tanh(mem.cell_in{L}{s});
            gWf_mul = dEdCt .* dCtdf;
            gWi_mul = dEdCt .* dCtdi;
            gWg_mul = dEdCt .* dCtdg;
            mem.grad.Who{L} = mem.grad.Who{L} + sum(pagefun(@mtimes, gWo_mul(:,:,2:end), permute(mem.cell_acts{L}{s}(:,:,1:end-1), [2 1 3])), 3);
            mem.grad.Whf{L} = mem.grad.Whf{L} + sum(pagefun(@mtimes, gWf_mul(:,:,2:end), permute(mem.cell_acts{L}{s}(:,:,1:end-1), [2 1 3])), 3);
            mem.grad.Whi{L} = mem.grad.Whi{L} + sum(pagefun(@mtimes, gWi_mul(:,:,2:end), permute(mem.cell_acts{L}{s}(:,:,1:end-1), [2 1 3])), 3);
            mem.grad.Whg{L} = mem.grad.Whg{L} + sum(pagefun(@mtimes, gWg_mul(:,:,2:end), permute(mem.cell_acts{L}{s}(:,:,1:end-1), [2 1 3])), 3);

            if(L == 1)
                mem.grad.Wxo{L}{s} = mem.grad.Wxo{L}{s} + sum(pagefun(@mtimes, gWo_mul, permute(in(config.slide_endpoints(s)+1:config.slide_endpoints(s+1),:,:), [3 1 2])), 3);
                mem.grad.Wxf{L}{s} = mem.grad.Wxf{L}{s} + sum(pagefun(@mtimes, gWf_mul, permute(in(config.slide_endpoints(s)+1:config.slide_endpoints(s+1),:,:), [3 1 2])), 3);
                mem.grad.Wxi{L}{s} = mem.grad.Wxi{L}{s} + sum(pagefun(@mtimes, gWi_mul, permute(in(config.slide_endpoints(s)+1:config.slide_endpoints(s+1),:,:), [3 1 2])), 3);
                mem.grad.Wxg{L}{s} = mem.grad.Wxg{L}{s} + sum(pagefun(@mtimes, gWg_mul, permute(in(config.slide_endpoints(s)+1:config.slide_endpoints(s+1),:,:), [3 1 2])), 3);
            else
                mem.grad.Wxo{L}{s} = mem.grad.Wxo{L}{s} + sum(pagefun(@mtimes, gWo_mul, permute(mem.cell_acts{L-1}{s}, [2 1 3])), 3);
                mem.grad.Wxf{L}{s} = mem.grad.Wxf{L}{s} + sum(pagefun(@mtimes, gWf_mul, permute(mem.cell_acts{L-1}{s}, [2 1 3])), 3);
                mem.grad.Wxi{L}{s} = mem.grad.Wxi{L}{s} + sum(pagefun(@mtimes, gWi_mul, permute(mem.cell_acts{L-1}{s}, [2 1 3])), 3);
                mem.grad.Wxg{L}{s} = mem.grad.Wxg{L}{s} + sum(pagefun(@mtimes, gWg_mul, permute(mem.cell_acts{L-1}{s}, [2 1 3])), 3);

                mem.delta_x{L}{s} = pagefun(@mtimes, repmat(config.weights.Wxo{L}{s}', [1 1 config.max_time_steps]), gWo_mul);
                mem.delta_x{L}{s} = mem.delta_x{L}{s} + pagefun(@mtimes, repmat(config.weights.Wxf{L}{s}', [1 1 config.max_time_steps]), gWf_mul) + ...
                                                  pagefun(@mtimes, repmat(config.weights.Wxi{L}{s}', [1 1 config.max_time_steps]), gWi_mul) + ...
                                                  pagefun(@mtimes, repmat(config.weights.Wxg{L}{s}', [1 1 config.max_time_steps]), gWg_mul);
            end

            mem.grad.Bi{L}{s} = mem.grad.Bi{L}{s} + sum(sum(gWi_mul, 3), 2);
            mem.grad.Bf{L}{s} = mem.grad.Bf{L}{s} + sum(sum(gWf_mul, 3), 2);
            mem.grad.Bo{L}{s} = mem.grad.Bo{L}{s} + sum(sum(gWo_mul, 3), 2);
            mem.grad.Bg{L}{s} = mem.grad.Bg{L}{s} + sum(sum(gWg_mul, 3), 2);

            if(L == config.hidden_layer_num)
                t_bprop = size(in,2)-config.output_valid_len+1;
            else
                t_bprop = 1;
            end


            for t = size(in,2):-1:t_bprop    % for each time step
                cum_prod = cumprod(mem.forget_gate{L}{s}(:,:,2:t), 3, 'reverse');
                dEdCt_rep = repmat(dEdCt(:,:,t),1,1,t-1);
                mem.grad.Whf{L} = mem.grad.Whf{L} + sum(pagefun(@mtimes, dEdCt_rep(:,:,1:end-1) .* cum_prod(:,:,2:end) .* dCtdf(:,:,2:t-1), permute(mem.cell_acts{L}{s}(:,:,1:t-2), [2 1 3])), 3);
                mem.grad.Whi{L} = mem.grad.Whi{L} + sum(pagefun(@mtimes, dEdCt_rep(:,:,1:end-1) .* cum_prod(:,:,2:end) .* dCtdi(:,:,2:t-1), permute(mem.cell_acts{L}{s}(:,:,1:t-2), [2 1 3])), 3);
                mem.grad.Whg{L} = mem.grad.Whg{L} + sum(pagefun(@mtimes, dEdCt_rep(:,:,1:end-1) .* cum_prod(:,:,2:end) .* dCtdg(:,:,2:t-1), permute(mem.cell_acts{L}{s}(:,:,1:t-2), [2 1 3])), 3);

                if(L == 1)
                    in_ = in(config.slide_endpoints(s)+1:config.slide_endpoints(s+1),:,:);
                    mem.grad.Wxf{L}{s} = mem.grad.Wxf{L}{s} + sum(pagefun(@mtimes, dEdCt_rep .* cum_prod .* dCtdf(:,:,1:t-1), permute(in_(:,1:t-1,:), [3 1 2])), 3);
                    mem.grad.Wxi{L}{s} = mem.grad.Wxi{L}{s} + sum(pagefun(@mtimes, dEdCt_rep .* cum_prod .* dCtdi(:,:,1:t-1), permute(in_(:,1:t-1,:), [3 1 2])), 3);
                    mem.grad.Wxg{L}{s} = mem.grad.Wxg{L}{s} + sum(pagefun(@mtimes, dEdCt_rep .* cum_prod .* dCtdg(:,:,1:t-1), permute(in_(:,1:t-1,:), [3 1 2])), 3);
                else
                    mem.grad.Wxf{L}{s} = mem.grad.Wxf{L}{s} + sum(pagefun(@mtimes, dEdCt_rep .* cum_prod .* dCtdf(:,:,1:t-1), permute(mem.cell_acts{L-1}{s}(:,:,1:t-1), [2 1 3])), 3);
                    mem.grad.Wxi{L}{s} = mem.grad.Wxi{L}{s} + sum(pagefun(@mtimes, dEdCt_rep .* cum_prod .* dCtdi(:,:,1:t-1), permute(mem.cell_acts{L-1}{s}(:,:,1:t-1), [2 1 3])), 3);
                    mem.grad.Wxg{L}{s} = mem.grad.Wxg{L}{s} + sum(pagefun(@mtimes, dEdCt_rep .* cum_prod .* dCtdg(:,:,1:t-1), permute(mem.cell_acts{L-1}{s}(:,:,1:t-1), [2 1 3])), 3);

                    mem.delta_x{L}{s}(:,:,1:t-1) = mem.delta_x{L}{s}(:,:,1:t-1) + ...
                                    pagefun(@mtimes, repmat(config.weights.Wxf{L}{s}', [1 1 size(cum_prod,3)]), dEdCt_rep .* cum_prod .* dCtdf(:,:,1:t-1)) + ...
                                    pagefun(@mtimes, repmat(config.weights.Wxi{L}{s}', [1 1 size(cum_prod,3)]), dEdCt_rep .* cum_prod .* dCtdi(:,:,1:t-1)) + ...
                                    pagefun(@mtimes, repmat(config.weights.Wxg{L}{s}', [1 1 size(cum_prod,3)]), dEdCt_rep .* cum_prod .* dCtdg(:,:,1:t-1));
                end

                mem.grad.Bf{L}{s} = mem.grad.Bf{L}{s} + sum(sum(dEdCt_rep .* cum_prod .* dCtdf(:,:,1:t-1), 3), 2);
                mem.grad.Bi{L}{s} = mem.grad.Bi{L}{s} + sum(sum(dEdCt_rep .* cum_prod .* dCtdi(:,:,1:t-1), 3), 2);
                mem.grad.Bg{L}{s} = mem.grad.Bg{L}{s} + sum(sum(dEdCt_rep .* cum_prod .* dCtdg(:,:,1:t-1), 3), 2);
            end
            
            % gradient clipping
            mem.grad.Wxo{L}{s}(mem.grad.Wxo{L}{s}>gclip) = gclip;
            mem.grad.Wxf{L}{s}(mem.grad.Wxf{L}{s}>gclip) = gclip;
            mem.grad.Wxi{L}{s}(mem.grad.Wxi{L}{s}>gclip) = gclip;
            mem.grad.Wxg{L}{s}(mem.grad.Wxg{L}{s}>gclip) = gclip;           
            mem.grad.Wxo{L}{s}(mem.grad.Wxo{L}{s}<-gclip) = -gclip;
            mem.grad.Wxf{L}{s}(mem.grad.Wxf{L}{s}<-gclip) = -gclip;
            mem.grad.Wxi{L}{s}(mem.grad.Wxi{L}{s}<-gclip) = -gclip;
            mem.grad.Wxg{L}{s}(mem.grad.Wxg{L}{s}<-gclip) = -gclip;

            mem.grad.Bi{L}{s}(mem.grad.Bi{L}{s}>gclip) = gclip;
            mem.grad.Bf{L}{s}(mem.grad.Bf{L}{s}>gclip) = gclip;
            mem.grad.Bo{L}{s}(mem.grad.Bo{L}{s}>gclip) = gclip;
            mem.grad.Bg{L}{s}(mem.grad.Bg{L}{s}>gclip) = gclip;
            mem.grad.Bi{L}{s}(mem.grad.Bi{L}{s}<-gclip) = -gclip;
            mem.grad.Bf{L}{s}(mem.grad.Bf{L}{s}<-gclip) = -gclip;
            mem.grad.Bo{L}{s}(mem.grad.Bo{L}{s}<-gclip) = -gclip;
            mem.grad.Bg{L}{s}(mem.grad.Bg{L}{s}<-gclip) = -gclip;
        end
        
        mem.grad.Who{L}(mem.grad.Who{L}>gclip) = gclip;
        mem.grad.Whf{L}(mem.grad.Whf{L}>gclip) = gclip;
        mem.grad.Whi{L}(mem.grad.Whi{L}>gclip) = gclip;
        mem.grad.Whg{L}(mem.grad.Whg{L}>gclip) = gclip;
        mem.grad.Who{L}(mem.grad.Who{L}<-gclip) = -gclip;
        mem.grad.Whf{L}(mem.grad.Whf{L}<-gclip) = -gclip;
        mem.grad.Whi{L}(mem.grad.Whi{L}<-gclip) = -gclip;
        mem.grad.Whg{L}(mem.grad.Whg{L}<-gclip) = -gclip;
        mem.grad.Wy(mem.grad.Wy>gclip) = gclip;
        mem.grad.Wy(mem.grad.Wy<-gclip) = -gclip;
    end
    %toc
end

function [hprev, csprev] = get_hprev_csprev(t, L, s, batch_size, previous_cell_acts, previous_cell_state)
    % csprev: previous cell state
    global config mem;

    if(t == 1)
        if exist('previous_cell_acts', 'var')
            hprev = previous_cell_acts{L}{s};
            csprev = previous_cell_state{L}{s};
        else
            hprev = config.NEW_MEM(zeros(config.hidden_layer_size(L), batch_size));
            csprev = 0;
        end
    else
        hprev = mem.cell_acts{L}{s}(:,:,t-1);
        csprev = mem.cell_state{L}{s}(:,:,t-1);
    end
end

function x = get_x(L, s, in, batch_size)
    global config mem;
    if(L == 1)
        x = permute(in(config.slide_endpoints(s)+1:config.slide_endpoints(s+1),:,:), [1 3 2]);
    else
        x = mem.cell_acts{L-1}{s}(:,1:batch_size,:);
    end
end

function gen_net_out()
    global config mem;
    if(size(mem.net_out,2) == 1)
        mem.net_out = config.NEW_MEM(zeros(config.output_size, config.max_time_steps, config.batch_size, config.slide_pieces));
    end
end



