% computes prox of l1 at u: argmin_x { norm(x, 1) + (1/alpha)*norm(x - u, 2)_2^2 }  
% essentially soft thresholding
% to see why this is the case, see neal parikh/stephen boyd's paper
% (chapter 6)
function output = prox_l1(u, alpha)
    if (alpha<=0) 
        error('alpha must be positive\n'); 
    end
    upos = (u>=alpha); 
    uneg = (u<=-alpha); 
    output = upos.*(u - alpha) + uneg.*(u + alpha);
end