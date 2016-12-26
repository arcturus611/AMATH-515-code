% Dec 25, 2016
% FISTA for lasso 
function output = fista_lasso(A, b, lam)
    %% initialize vars
    AtA = A'*A; Atb = A'*b; 
    L = norm(AtA, 2)/lam; 
    
    x = zeros(size(A, 2), 1); 
    y = zeros(size(A, 2), 1); 
    
    delta = 10;
    epsilon = 10^(-5);
    num_itrs = 1; 
    
    t(num_itrs) = 1; 
    
    %% FISTA
    while(delta > epsilon)
        
        val(num_itrs) = eval_lasso(A, b, lam, x(:, num_itrs)); 
                
        u = y - (AtA*x(:, num_itrs) - Atb)/(lam*L); 
        
        x(:, num_itrs + 1) = prox_l1(u, 1/L); 
        
        t(num_itrs+1) = (1 + sqrt(1 + 4*t(num_itrs)^2))/2; 
        
        y = x(:,num_itrs +1) + (1/t(num_itrs+1))*(t(num_itrs) - 1)*(x(:,num_itrs+1) - x(:,num_itrs)); 
        
        delta_vec(num_itrs+1) = norm( x(:, num_itrs+1) - x(:, num_itrs), 2); 
        
        delta = delta_vec(num_itrs+1)
          
        num_itrs = num_itrs + 1; 
       
    end
output.x_opt = x;     
output.nIter = num_itrs; 
output.delta_vec = delta_vec; 
output.val = val; 
end
