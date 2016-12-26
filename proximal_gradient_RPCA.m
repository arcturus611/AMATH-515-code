function output = proximal_gradient_RPCA(M, lambdaL, lambdaS) 
%% Prox-Gradient algorithm
% pre-calculate constants
delta = 10;
epsilon = 10^(-8);
iter = 1;
prox_l1_thres =  lambdaS;
prox_nuc_norm_thres = lambdaL;

% allocate memory to matrices
L_curr = zeros(size(M)); 
S_curr = L_curr;
sz = size(S_curr);
len = sz(1)*sz(2);

%% Iterate until convergence
while (delta > epsilon)
    L_prev = L_curr;
    S_prev = S_curr;
    
    % prox of l1 norm 
    S_curr_vec = reshape(M-L_prev, len, 1);
    S_curr_vec = prox_op_l1(S_curr_vec, prox_l1_thres);
    S_curr = reshape(S_curr_vec, sz(1), sz(2));
    
    % prox of nuclear norm
    L_curr = prox_op_nuc_norm(M-S_prev, prox_nuc_norm_thres);
    
    % check convergence
    delta_vec(iter) = norm(L_curr - L_prev, 'fro')^2 + norm(S_curr- S_prev, 'fro')^2;
    delta = delta_vec(iter);
    
    iter = iter+1;
end

% pack all output in struct
output.L_opt = L_curr;
output.S_opt = S_curr;
output.nIter = iter;
output.delta_vec = delta_vec;
output.opt_val = 0.5*norm(M - output.L_opt - output.S_opt, 'fro')^2 + lambdaL*nuclear_norm(output.L_opt) + lambdaS*l1_norm(reshape(output.S_opt, len, 1));
end
