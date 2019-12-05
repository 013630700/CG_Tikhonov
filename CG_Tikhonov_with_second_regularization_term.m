% Example computations related to X-ray tomography. Here we apply Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% Salla Latva-Äijö and Samuli Siltanen, November 2019
clear all;

% Regularization parameter
alpha1 = 10000; %Kun alpha1 on suuri, tulee toiseenkin kuvaan ympyr�
alpha2 = 2;
N      = 40;

% Measure computation time later; start clocking here
tic

% Define coefficients: Iodine and Al
c11 = 42.2057; %Iodine 30kV
c21 = 60.7376; %Iodine 50kV
c12 = 3.044;   %Al 30kV
c22 = 0.994;   %Al 50kV

% Construct phantom. You can modify the resolution parameter N.
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
%m = m+0.001*max(abs(m))*randn(size(m));

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
K   = 100;         % maximum number of iterations
g   = b;  % initial iterate is the backprojected data
rho = zeros(K,1); % initialize parameters

% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!
% First iteration (special one)

N2 = size(g,1);
%Initialize second regularization term R
R = zeros(N2,1);

%Count the gradient
i1 = 1:N2; i2 = [N2/2+1:N2,1:N2/2];
for j = 1:N2
    R(j) = g(i1(j))*g(i2(j))^2;
end
figure(10);
imshow(R,[]);

% very old version: Hx     = (A.')*(A*x) + alpha*x; 
% second try:       Hx     = (A.')*Amult(A,x) + alpha*x;
Hg     = A2x2Tmult(a,c11,c12,c21,c22,A2x2mult(a,c11,c12,c21,c22,g)) + alpha1*g;
% % And now we ADD second regularization term
Hg     = Hg +alpha2*R;
r      = b-Hg;
rho(1) = r(:).'*r(:);
figure(1);
% Start iteration
for kkk = 1:K
%     Count the gradient
    i1 = 1:N2; i2 = [N2/2+1:N2,1:N2/2];
       for j = 1:N2
       R(j) = g(i1(j))*g(i2(j))^2;
       end
    R = R(:);
    imshow(R,[]);
    if kkk==1
        p = r;
    else
        beta = rho(kkk)/rho(kkk-1);
        p    = r + beta*p;
    end
    w          = A2x2Tmult(a,c11,c12,c21,c22,A2x2mult(a,c11,c12,c21,c22,p));
    w          = w + alpha1*p;
    w          = w +alpha2*R;
    aS         = rho(kkk)/(p.'*w);
    g          = g + aS*p;
    r          = r - aS*w;
    rho(kkk+1) = r(:).'*r(:);
    disp([kkk K])
%   figure(1)
%     recn1 = g(1:1600);
%     recn1 = reshape(recn1,N,N);
%     imshow(recn1,[]);
%     pause(0.2);
% %   recn2 = g(1601:3200);
% %   recn2 = reshape(recn2,N,N);
% %   imshow(recn2,[]);
% %   pause(0.2);
end
recn = g;
recn1 = g(1:1600);
recn1 = reshape(recn1,N,N);
recn2 = g(1601:3200);
recn2 = reshape(recn2,N,N);

% Determine computation time
comptime = toc;

% Compute relative errors
err_sup1 = max(max(abs(g1-recn1(:))))/max(max(abs(g1)));
err_squ1 = norm(g1(:)-recn1(:))/norm(g1(:));

err_sup2 = max(max(abs(g2-recn2(:))))/max(max(abs(g2)));
err_squ2 = norm(g2(:)-recn2(:))/norm(g2(:));

% Save result to file
% eval(['save XRMG_Tikhonov', num2str(N), ' recn alpha target comptime err_sup err_squ']);

% View the results
%XRMG_Tikhonov_plot(N)
%figure(100);
%imshow(recn,[]);
%% Take a look at the results. we plot the original phantoms and their
% reconstructions into the same figure
figure(25);
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