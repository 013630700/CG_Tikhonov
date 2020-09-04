function [x,iter,err,funct] = prox_grad(A,sino,obj,W,opts);

% Input: A = system matrix
%        sino = sinogram
%        W = matrix of weights
%        x0 = initial guess
%        sol = true solution
%
% Output: x = solution
%         iter = number of iteration to convergence
%         err = vector containing the error at each iteration
%         funct = vector containing the objective function value at each iteration 

% Input parameters
if isfield(opts,'x0'), x = opts.x0; x = x(:); else error('Provide initial guess'); end
if isfield(opts,'gamma'), gamma = opts.gamma; else gamma = 1; end % regularization parameter
if isfield(opts,'M'), M = opts.M; else M = 10; end % maxit for power iteration(Lipschitz constant estimation - inner loop)
if isfield(opts,'maxit'), maxit = opts.maxit; else maxit = 100; end % maxit for gradient descent (outer loop)
if isfield(opts,'stopcrit'), stopcrit = opts.stopcrit; else stopcrit = 1; end 
if isfield(opts,'tol'), tol = opts.tol; else tol = 1e-2; end
% Allocation
iter = 1;
loop = true;
err = [];
funct = [];

% Vectorization
sino = sino(:);
obj = obj(:);

normest_A = normest(A);
N = sqrt(size(A,2));
AT = @(x) A'*x; 
A = @(x) A*x;
W = diag(W);

% Computations needed only once
norm_crit = norm(x-obj); 
norm_obj = norm(obj);

L = normest_A^2 + gamma;
% Estimate Lipschitz constant
y = randn(size(x));
for j = 1:M
    z = A(y)/norm(y);
    y = AT(z);
end
L_est = norm(y) + gamma;
fprintf('\n L = %f \t L_est=%f (iter=%d)', L, L_est,j);

D = spdiags([-ones(N-1,1), ones(N-1,1); 0, 1], 0:1, N, N);
D1 = kron(speye(N),D);
D2 = kron(D, speye(N));
D1T = @(x) D1'*x;
D2T = @(x) D2'*x;
D1 = @(x) D1*x;
D2 = @(x) D2*x;
regu = norm(D1(x))^2 + norm(D2(x))^2;

t = 1/(L_est + gamma*8);

err(1) = norm_crit/norm_obj;
LS = A(x) - sino;
fv = 0.5*(norm(sqrt(W).*LS)^2 + gamma*regu);
funct(1) = fv;

while loop
    % step
    Ax = A(x);
    LSw = W.*(Ax-sino);
    g = AT(LSw) + gamma*( D1T(D1(x)) + D2T(D2(x)) );
    xtry = x - t * g;
    
    % stopping criterion    
    LSwtry = W.*(A(xtry)-sino);
    ftry = 0.5*(norm(sqrt(LSwtry))^2 + gamma*(norm(D1(xtry))^2 + norm(D2(xtry))^2));
    gtry = AT(LSwtry) + gamma*( D1T(D1(xtry)) + D2T(D2(xtry)) );
    
    if iter >= 2
        switch stopcrit
            case 1
                normstep = norm(xtry-x)/norm(x);
