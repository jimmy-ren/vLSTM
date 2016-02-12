function adagrad_init()
    global config;
    
    config.his_grad.Wy = config.NEW_MEM(single(zeros(size(config.weights.Wy))));
    for L = 1:config.hidden_layer_num
        config.his_grad.Who{L} = config.NEW_MEM(single(zeros(size(config.weights.Who{L}))));
        config.his_grad.Whf{L} = config.NEW_MEM(single(zeros(size(config.weights.Whf{L}))));
        config.his_grad.Whi{L} = config.NEW_MEM(single(zeros(size(config.weights.Whi{L}))));
        config.his_grad.Whg{L} = config.NEW_MEM(single(zeros(size(config.weights.Whg{L}))));
        for s = 1:config.slide_pieces
            config.his_grad.Wxo{L}{s} = config.NEW_MEM(single(zeros(size(config.weights.Wxo{L}{s}))));
            config.his_grad.Wxf{L}{s} = config.NEW_MEM(single(zeros(size(config.weights.Wxf{L}{s}))));
            config.his_grad.Wxi{L}{s} = config.NEW_MEM(single(zeros(size(config.weights.Wxi{L}{s}))));
            config.his_grad.Wxg{L}{s} = config.NEW_MEM(single(zeros(size(config.weights.Wxg{L}{s}))));
            config.his_grad.Bo{L}{s} = config.NEW_MEM(single(zeros(size(config.weights.Bo{L}{s}))));
            config.his_grad.Bf{L}{s} = config.NEW_MEM(single(zeros(size(config.weights.Bf{L}{s}))));
            config.his_grad.Bi{L}{s} = config.NEW_MEM(single(zeros(size(config.weights.Bi{L}{s}))));
            config.his_grad.Bg{L}{s} = config.NEW_MEM(single(zeros(size(config.weights.Bg{L}{s}))));
        end
    end
end
