function lstm_writer_val()
    global config mem;

    % start from 'I', ascii 73
    input = config.NEW_MEM(zeros(95, 1, 1));
    input(73-31) = 1;
    fprintf('\nI');
    
    % number of characters to write
    char2write = 800;
    lstm_core_v52(input, -1)   
    
    out = gather(mem.net_out);
    out = out(:,1,1);
    [b, pos] = max(out);
    cc = char(pos+31);
    fprintf('%s', cc);
    
    in_next = config.NEW_MEM(zeros(95, 1));
    in_next(pos) = 1;
    
    
    config.temperature = 0.5;
    for c = 2:char2write
        hprev = mem.cell_acts;
        csprev = mem.cell_state;

        for L = 1:length(config.hidden_layer_size)
            for s = 1:config.slide_pieces
                hprev{L}{s} = hprev{L}{s}(:,1,1);
                csprev{L}{s} = csprev{L}{s}(:,1,1);
            end
        end
        lstm_core_v52(in_next, -1, hprev, csprev);
    
        out = gather(mem.net_out);
        out = out(:,1,1);
        %[b, pos] = max(out);
        pos = randsample(1:95, 1, true, out);
        
        cc = char(pos+31);        
        fprintf('%s', cc);
        
        in_next = config.NEW_MEM(zeros(95, 1));
        in_next(pos) = 1;
    end
    
    fprintf('\n');
    
    config.temperature = 1;
end
