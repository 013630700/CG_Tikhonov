% Example computations related to X-ray tomography. Here we apply Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% Salla Latva-Äijö and Samuli Siltanen, November 2019

% Regularization parameter
alpha = 1;              

% Measure computation time later; start clocking here
tic

% Define coefficients: Iodine and Al
c11 = 42.2057; %Iodine 30kV
c21 = 60.7376; %Iodine 50kV
c12 = 3.044;   %Al 30kV
c22 = 0.994;   %Al 50kV

% Construct phantom. You can modify the resolution parameter N.
N      = 40;
target1 = imresize(double(imread('HY_Al.bmp')),[N N]);
target2 = imresize(double(imread('HY_square_inv.jpg')),[N N]);
% Vektorize
g1 = target1(:);
g2 = target2(:);
% Combine
g=[g1;g2];

% Choose measurement angles (given in degrees, not radians). 
Nang    = N; 
angle0  = -90;
measang = angle0 + [0:(Nang-1)]/Nang*180;

% % Initialize measurement matrix of size (M*P) x N^2, where M is the number of
% % X-ray directions and P is the number of pixels that Matlab's Radon
% % function gives.
% target = target1;
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
a = A;

% Simulate noisy measurements; here including inverse crime
m = A2x2mult(a,c11,c12,c21,c22,g);
m = m+0.01*max(abs(m))*randn(size(m));

% Solve the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter

b = A2x2Tmult(a,c11,c12,c21,c22,m);

% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
K   = 80;         % maximum number of iterations
g1   = b(1:end/2);          % initial iterate is the backprojected data
g2   = b((end/2+1):end);    % initial iterate is the backprojected data
rho = zeros(K,1); % initialize parameters

% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!

% very old version: Hx     = (A.')*(A*x) + alpha*x; 
% second try:       Hx     = (A.')*Amult(A,x) + alpha*x;
Hg     = A2x2Tmult(a,c11,c12,c21,c22,A2x2mult(a,c11,c12,c21,c22,g))
Hg     = Hg + alpha*g;
r      = b-Hg;
rho(1) = r(:).'*r(:);

% Start iteration
for kkk = 1:K
    if kkk==1
        p = r;
    else
        beta = rho(kkk)/rho(kkk-1);
        p    = r + beta*p;
    end
    Ap         = A2x2mult(a,c11,c12,c21,c22,p1,p2);
    w          = AT*Ap;
    %Lis�t��n alpha*x regularisointi termi
    w          = w + alpha*x;
    aS          = rho(kkk)/(p.'*w);
    x          = x + aS*p;
    r          = r - aS*w;
    rho(kkk+1) = r(:).'*r(:);
    disp([kkk K])
end

recn = x;
recn1 = x(1:1600);
recn1 = reshape(recn1,N,N);
recn2 = x(1601:3200);
recn2 = reshape(recn2,N,N);

% Determine computation time
comptime = toc;

% Compute relative errors
%err_sup = max(max(abs(g1-recn)))/max(max(abs(g1)));
%err_squ = norm(g1(:)-recn(:))/norm(g1(:));

% Save result to file
% eval(['save XRMG_Tikhonov', num2str(N), ' recn alpha target comptime err_sup err_squ']);

% View the results
%XRMG_Tikhonov_plot(N)
%figure(100);
%imshow(recn,[]);
%% Take a look at the results. we plot the original phantoms and their
% reconstructions into the same figure
figure(3);
% Original target1
subplot(2,2,1);
imagesc(reshape(g1,N,N));
colormap gray;
axis square;
axis off;
title({'M1, original'});
% Reconstruction of target1
subplot(2,2,2)
imagesc(recn1);
colormap gray;
axis square;
axis off;
title('M1 CG recn1 ');
% Original target2
subplot(2,2,3)
imagesc(reshape(g2,N,N));
colormap gray;
axis square;
axis off;
title({'M2, original'});
% Reconstruction of target2
subplot(2,2,4)
imagesc(recn2);
colormap gray;
axis square;
axis off;
title(['M2 CG recn2, iter=' num2str(K)]);