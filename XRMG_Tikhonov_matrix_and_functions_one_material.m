% Example computations related to X-ray tomography. Here we apply Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% Jennifer Mueller and Samuli Siltanen, October 2012

% Regularization parameter
alpha = 10;

% Maximum number of iterations
MAXITER = 100;               

% Measure computation time later; start clocking here
tic

% Construct phantom. You can modify the resolution parameter N.
N      = 40;
I1 = imread('HY_Al.bmp');
target = double(I1);
target = imresize(target, [N N]);

% Choose measurement angles (given in degrees, not radians). 
Nang    = N; 
angle0  = -90;
measang = angle0 + [0:(Nang-1)]/Nang*180;

% % Initialize measurement matrix of size (M*P) x N^2, where M is the number of
% % X-ray directions and P is the number of pixels that Matlab's Radon
% % function gives.
% P  = length(radon(target,0));
% M  = length(measang);
% A = sparse(M*P,N^2);
% 
% % Construct measurement matrix column by column. The trick is to construct
% % targets with elements all 0 except for one element that equals 1.
% for mmm = 1:M
%     for iii = 1:N^2
%         tmpvec                  = zeros(N^2,1);
%         tmpvec(iii)             = 1;
%         A((mmm-1)*P+(1:P),iii) = radon(reshape(tmpvec,N,N),measang(mmm));
%         if mod(iii,100)==0
%             disp([mmm, M, iii, N^2])
%         end
%     end
% end
% 
% % Test the result
% Rtemp = radon(target,measang);
% Rtemp = Rtemp(:);
% Mtemp = A*target(:);
% disp(['If this number is small, then the matrix A is OK: ', num2str(max(max(abs(Mtemp-Rtemp))))]);
% 
% % Save the result to file (with filename containing the resolution N)
% eval(['save RadonMatrix', num2str(N), ' A measang target N P Nang']);

% Load radonMatrix
eval(['load RadonMatrix', num2str(N), ' A measang target N P Nang']);

%% Load noisy measurements EIPÄS, tehdään RIKOS
%eval(['load XRMC_NoCrime', num2str(N), ' N mnc mncn']);
[m,s] = radon(target,measang);

% Construct system matrix and first-order term for the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter.o
b     = ATmult(A,m);  %old version: b     = A.'*m(:);

% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
K   = 80;         % maximum number of iterations
x   = b;          % initial iterate is the backprojected data
rho = zeros(K,1); % initialize parameters
% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!

% very old version: Hx     = (A.')*(A*x) + alpha*x; 
% second try:       Hx     = (A.')*Amult(A,x) + alpha*x;
Hx     = ATmult(A,Amult(A,x)) + alpha*x; % Third version, which seems to work as well
r      = b-Hx;
rho(1) = r.'*r;
% Start iteration
for kkk = 1:K
    if kkk==1
        p = r;
    else
        beta = rho(kkk)/rho(kkk-1);
        p    = r + beta*p;
    end
    % very old version: w          = (A.')*(A*p) + alpha*p; 
    % second try:       w          = (A.')*Amult(A,p) +alpha*p;
    w          = ATmult(A,Amult(A,p))+ alpha*p; % with new functions
    a          = rho(kkk)/(p.'*w);
    x          = x + a*p;
    r          = r - a*w;
    rho(kkk+1) = r.'*r;
    disp([kkk K])
end
recn = reshape(x,N,N);

% Determine computation time
comptime = toc;

% Compute relative errors
err_sup = max(max(abs(target-recn)))/max(max(abs(target)));
err_squ = norm(target(:)-recn(:))/norm(target(:));

% Save result to file
% eval(['save XRMG_Tikhonov', num2str(N), ' recn alpha target comptime err_sup err_squ']);

% View the results
%XRMG_Tikhonov_plot(N)
figure(100);
imshow(recn,[]);