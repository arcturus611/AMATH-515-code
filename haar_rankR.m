function Q = haar_rankR(n, r, cmplx)
% Q = HAAR_RANKR( N , R)
%   returns a N x N unitary matrix Q
%   from the Haar measure on the
%   Circular Unitary Ensemble (CUE)
%
% Q = HAAR_RANKR( N, R, COMPLEX )
%   if COMPLEX is false, then returns a real matrix from the
%   Circular Orthogonal Ensemble (COE). By default, COMPLEX = true
% 
% For more info, see http://www.ams.org/notices/200705/fea-mezzadri-web.pdf
% "How to Generate Random Matrices fromthe Classical Compact Groups"
% by Francesco Mezzadri (also at http://arxiv.org/abs/math-ph/0609050 )
%
% Other references:
%   "How to generate a random unitary matrix" by Maris Ozols
%
% Stephen Becker, 2/24/11. updated 11/1/12 for rank r version.
%
% To test that the eigenvalues are evenly distributed on the unit
% circle, do this:   a = angle(eig(haar(1000))); hist(a,100);
%
% This calls "randn", so it will change the stream of the prng

if nargin < 3, cmplx = true; end
if nargin < 2 || isempty(r) , r=n; end

if cmplx
    z = (randn(n,r) + 1i*randn(n,r))/sqrt(2.0);
else
    z = randn(n,r);
end
[Q,R] = qr(z,0); % time consuming
d = diag(R);
ph = d./abs(d);
% Q = multiply(q,ph,q) % in python. this is q <-- multiply(q,ph)
%   where we multiply each column of q by an element in ph

%Q = Q.*repmat( ph', n, 1 );
%  use bsxfun for this to make it fast...
Q = bsxfun( @times, Q, ph' );
