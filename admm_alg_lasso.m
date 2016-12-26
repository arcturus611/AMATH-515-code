% admm
function output = admm_alg_lasso(A, b, lambda, rho)
    sz = size(A);
    n = sz(2);
    x_curr = zeros(n, 1);
    z = x_curr;
    u = x_curr;
    
    AtA_ = A'*A + rho*eye(n);
    delta = 10;
    nIter = 0;
    epsilon = 10^(-8);
    
    while (delta > epsilon)
        x_prev = x_curr;
        nIter = nIter + 1; 
        
        % variable updates
        Atb_ = A'*b + rho*z - u;

        x_curr = AtA_\Atb_;
        
        z = prox_op_l1(x_curr + u/rho, lambda/rho);
        
        u = u + rho*(x_curr - z);
        
        % check convergence
        delta_vec(nIter) = norm(x_prev - x_curr, 2)^2; 
        delta = delta_vec(nIter);
        val(nIter) = eval_lasso(A, b, lambda, x_curr); 
    end
    
    output.x_opt = x_curr;
    output.delta_vec = delta_vec;
    output.nIter = nIter; 
    output.val = val; 
end