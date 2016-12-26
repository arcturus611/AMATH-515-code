% AMATH 515 Homework 2 Problem 5
%% Swati  Padmanabhan
clear all; close all; clc

%% set up data
[A, b, lambda, L, x0] = lasso_init();

%% Coordinate descent
cd_lasso = coordinate_descent_lasso(A, b, lambda);

%% Prox Gradient 
prox_grad_lasso = proximal_gradient_lasso(A, b, lambda, L) ;

%% ADMM 
admm_lasso = admm_alg_lasso(A, b, lambda, 1);

%% FISTA 
%fista_lasso = fista_alg_lasso(A, b, lambda);
fista_lasso = fista_lasso(A, b, lambda); 

n = size(A, 2);
% Run CVX here 
cvx_begin quiet
    cvx_precision low
    variable xCVX(n)
    minimize 0.5*sum_square(A*xCVX - b) + lambda*norm(xCVX, 1)
cvx_end

%% Uncomment lines to see how close your answer is
%% output results

fprintf('\n Distance between CVX solution and coordinate descent: %5.4f \n', norm(cd_lasso.x_opt - xCVX));
fprintf('\n Distance between CVX solution and proximal splitting: %5.4f \n', norm(prox_grad_lasso.x_opt - xCVX));
fprintf('\n Distance between CVX solution and fast proximal splitting: %5.4f \n', norm(fista_lasso.x_opt - xCVX));
fprintf('\n Distance between CVX solution and admm: %5.4f \n', norm(admm_lasso.x_opt - xCVX));

fprintf('\n Number of iterations for coordinate descent: %d, proximal gradient: %d, FISTA: %d and ADMM: %d, \n', cd_lasso.nIter, prox_grad_lasso.nIter, fista_lasso.nIter, admm_lasso.nIter);

all_num_iters = sort([cd_lasso.nIter; prox_grad_lasso.nIter; fista_lasso.nIter; admm_lasso.nIter]); 
plot_num_iters = all_num_iters(1); 
% PLOT_ITER = 8;
% figure, plot(cd_lasso.delta_vec(1:PLOT_ITER), 'r'); hold on; 
% plot(prox_grad_lasso.delta_vec(1:PLOT_ITER), 'g');hold on; 
% plot(fista_lasso.delta_vec(1:PLOT_ITER), 'b'); hold on; 
% plot(admm_lasso.delta_vec(1:PLOT_ITER), 'k'); 
% legend('Coordinate Descent', 'Proximal Gradient', 'FISTA', 'ADMM'); 
% xlabel('# iterations'); ylabel('Delta');
% axis([1 20 0 3]);
% title('Big data set LASSO');

PLOT_ITER = 60%plot_num_iters;
%figure, plot(cd_lasso.val(1:PLOT_ITER), 'r'); hold on; 
plot(prox_grad_lasso.val(1:PLOT_ITER), 'g');hold on; 
plot(fista_lasso.val(1:PLOT_ITER), 'b'); hold on; 
%plot(admm_lasso.val(1:PLOT_ITER), 'k'); 
%legend('Coordinate Descent', 'Proximal Gradient', 'FISTA', 'ADMM'); 
legend('Proximal Gradient', 'FISTA'); 
xlabel('# iterations'); ylabel('Delta');
%axis([1 20 0 3]);
title('Big data set LASSO');