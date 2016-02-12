function adagrad_update(eta)
    global config mem;
    
    fudge_factor = config.fudge_factor;
    learning_rate = config.learning_rate;    
    weight_decay = config.weight_decay;
    
    config.his_grad.Wy = config.his_grad.Wy + mem.grad.Wy .* mem.grad.Wy;
    for L = 1:config.hidden_layer_num
        config.his_grad.Who{L} = config.his_grad.Who{L} + mem.grad.Who{L} .* mem.grad.Who{L};
        config.his_grad.Whf{L} = config.his_grad.Whf{L} + mem.grad.Whf{L} .* mem.grad.Whf{L};
        config.his_grad.Whi{L} = config.his_grad.Whi{L} + mem.grad.Whi{L} .* mem.grad.Whi{L};
        config.his_grad.Whg{L} = config.his_grad.Whg{L} + mem.grad.Whg{L} .* mem.grad.Whg{L};
        for s = 1:config.slide_pieces                    
            config.his_grad.Wxo{L}{s} = config.his_grad.Wxo{L}{s} + mem.grad.Wxo{L}{s} .* mem.grad.Wxo{L}{s};
            config.his_grad.Wxf{L}{s} = config.his_grad.Wxf{L}{s} + mem.grad.Wxf{L}{s} .* mem.grad.Wxf{L}{s};
            config.his_grad.Wxi{L}{s} = config.his_grad.Wxi{L}{s} + mem.grad.Wxi{L}{s} .* mem.grad.Wxi{L}{s};
            config.his_grad.Wxg{L}{s} = config.his_grad.Wxg{L}{s} + mem.grad.Wxg{L}{s} .* mem.grad.Wxg{L}{s};
            config.his_grad.Bo{L}{s} = config.his_grad.Bo{L}{s} + mem.grad.Bo{L}{s} .* mem.grad.Bo{L}{s};
            config.his_grad.Bf{L}{s} = config.his_grad.Bf{L}{s} + mem.grad.Bf{L}{s} .* mem.grad.Bf{L}{s};
            config.his_grad.Bi{L}{s} = config.his_grad.Bi{L}{s} + mem.grad.Bi{L}{s} .* mem.grad.Bi{L}{s};
            config.his_grad.Bg{L}{s} = config.his_grad.Bg{L}{s} + mem.grad.Bg{L}{s} .* mem.grad.Bg{L}{s};
        end
    end

    config.weights.Wy = config.weights.Wy - learning_rate * (mem.grad.Wy ./ (fudge_factor + sqrt(config.his_grad.Wy))) - eta * weight_decay(1) * config.weights.Wy;
    for L = 1:config.hidden_layer_num
        config.weights.Who{L} = config.weights.Who{L} - learning_rate * (mem.grad.Who{L} ./ (fudge_factor + sqrt(config.his_grad.Who{L}))) - eta * weight_decay(1) * config.weights.Who{L};
        config.weights.Whf{L} = config.weights.Whf{L} - learning_rate * (mem.grad.Whf{L} ./ (fudge_factor + sqrt(config.his_grad.Whf{L}))) - eta * weight_decay(1) * config.weights.Whf{L};
        config.weights.Whi{L} = config.weights.Whi{L} - learning_rate * (mem.grad.Whi{L} ./ (fudge_factor + sqrt(config.his_grad.Whi{L}))) - eta * weight_decay(1) * config.weights.Whi{L};
        config.weights.Whg{L} = config.weights.Whg{L} - learning_rate * (mem.grad.Whg{L} ./ (fudge_factor + sqrt(config.his_grad.Whg{L}))) - eta * weight_decay(1) * config.weights.Whg{L};
        for s = 1:config.slide_pieces                    
            config.weights.Wxo{L}{s} = config.weights.Wxo{L}{s} - learning_rate * (mem.grad.Wxo{L}{s} ./ (fudge_factor + sqrt(config.his_grad.Wxo{L}{s}))) - eta * weight_decay(s) * config.weights.Wxo{L}{s};
            config.weights.Wxf{L}{s} = config.weights.Wxf{L}{s} - learning_rate * (mem.grad.Wxf{L}{s} ./ (fudge_factor + sqrt(config.his_grad.Wxf{L}{s}))) - eta * weight_decay(s) * config.weights.Wxf{L}{s};
            config.weights.Wxi{L}{s} = config.weights.Wxi{L}{s} - learning_rate * (mem.grad.Wxi{L}{s} ./ (fudge_factor + sqrt(config.his_grad.Wxi{L}{s}))) - eta * weight_decay(s) * config.weights.Wxi{L}{s};
            config.weights.Wxg{L}{s} = config.weights.Wxg{L}{s} - learning_rate * (mem.grad.Wxg{L}{s} ./ (fudge_factor + sqrt(config.his_grad.Wxg{L}{s}))) - eta * weight_decay(s) * config.weights.Wxg{L}{s};
            config.weights.Bo{L}{s} = config.weights.Bo{L}{s} - learning_rate * (mem.grad.Bo{L}{s} ./ (fudge_factor + sqrt(config.his_grad.Bo{L}{s}))) - eta * weight_decay(s) * config.weights.Bo{L}{s};
            config.weights.Bf{L}{s} = config.weights.Bf{L}{s} - learning_rate * (mem.grad.Bf{L}{s} ./ (fudge_factor + sqrt(config.his_grad.Bf{L}{s}))) - eta * weight_decay(s) * config.weights.Bf{L}{s};
            config.weights.Bi{L}{s} = config.weights.Bi{L}{s} - learning_rate * (mem.grad.Bi{L}{s} ./ (fudge_factor + sqrt(config.his_grad.Bi{L}{s}))) - eta * weight_decay(s) * config.weights.Bi{L}{s};
            config.weights.Bg{L}{s} = config.weights.Bg{L}{s} - learning_rate * (mem.grad.Bg{L}{s} ./ (fudge_factor + sqrt(config.his_grad.Bg{L}{s}))) - eta * weight_decay(s) * config.weights.Bg{L}{s};
        end
    end
end
