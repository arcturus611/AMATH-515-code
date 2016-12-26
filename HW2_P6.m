clear all; close all; clc;

rng(101);
m   = 400; n = 500; lambdaL = .25; lambdaS = 1e-2;
rk  = round(.5*min(m,n) );
Q1  = haar_rankR(m,rk,false);
Q2  = haar_rankR(n,rk,false);
Y   = Q1*diag(.1+rand(rk,1))*Q2';
mdn = median(abs(Y(:)));
Y   = Y + exprnd( .1*mdn, m, n );
M = Y; % to make notation match with theory

% proximal gradient robust PCA
prox_grad_RPCA = proximal_gradient_RPCA(M, lambdaL, lambdaS); 

% robust PCA by partial minimization proximal gradient 
parmin_prox_grad_RPCA = parmin_proximal_gradient_RPCA(M, lambdaL, lambdaS);

