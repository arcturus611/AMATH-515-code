function output = update_grad_huber(M, L_prev, lambdaS)
    X = M - L_prev;
    sz = size(X);
    len = sz(1)*sz(2);
    output = zeros(len, 1);
    x = reshape(X, len, 1);
    for i = 1:len
        if (x(i)>lambdaS)
            output(i)= -lambdaS;
        elseif (x(i)<-lambdaS)
            output(i) = lambdaS;
        else
            output(i) = -x(i);
        end 
    end
end
