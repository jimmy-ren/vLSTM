function y = softmax(x, T)
    if nargin < 2
        T = 1;
    end
    M = bsxfun(@minus, x/T, max(x/T, [], 1));
    numerator = exp(M);
    denominator = sum(numerator) + 0.00001;    
    y = bsxfun(@rdivide, numerator, denominator) + 0.00001;    
end
