function output = parmin_proximal_gradient_RPCA(M, lambdaL, lambdaS)
%% Prox-Gradient algorithm
% pre-calculate constants
delta = 10;
epsilon = 10^(-8);
iter = 1;
prox_nuc_norm_thres = lambdaL;

% allocate memory to matrices
L_curr = zeros(size(M)); 
sz = size(L_curr);

%% Iterate until convergence
while (delta > epsilon)
    L_prev = L_curr;
    
    % update grad_huber
    grad_huber_curr_vec = update_grad_huber(M, L_prev, lambdaS);
    grad_huber_curr = reshape(grad_huber_curr_vec, sz(1), sz(2));
    
    % prox of nuclear norm
    L_curr = prox_op_nuc_norm(L_curr - grad_huber_curr, prox_nuc_norm_thres);
    
    % check convergence
    delta_vec(iter) = norm(L_curr - L_prev, 'fro')^2;
    delta = delta_vec(iter);
    
    iter = iter+1;
end

% pack all output in struct
output.L_opt = L_curr;
output.nIter = iter;
output.delta_vec = delta_vec;
end
