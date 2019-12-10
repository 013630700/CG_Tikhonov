%% Level set method, example 2, case n=2
% This modified level set method allows target to change in time.
% Tomographic reconstruction is found as a minimizer of a nonlinear
% functional, which contains a regularization term penalizing the L^2 norms
% up to n derivatives of the reconstruction. For the case n=2 we aim to 
% minimize the functional F2 but drop for simplicity the mixed derivatives from the functional. 

% Salla Latva-ƒijˆ revisited 8.12.2019

clear all;

% Regularization parameter
alpha1  = 10;%10000
alpha2  = 1000;%1
N       = 40;

% Choose relative noise level in simulated noisy data
noiselevel = 0.00001;

% Measure computation time later; start clocking here
tic

% Define coefficients: Iodine and Al
c11     = 42.2057; %Iodine 30kV
c21     = 60.7376; %Iodine 50kV
c12     = 3.044;   %Al 30kV
c22     = 0.994;   %Al 50kV

% Construct phantom. You can modify the resolution parameter N.
target1 = imresize(double(imread('HY_Al.bmp')),[N N]);
target2 = imresize(double(imread('HY_square_inv.jpg')),[N N]);
% Vektorize
g1      = target1(:);
g2      = target2(:);
% Normalize original data
g1=normalize(g1);
g2=normalize(g2);

% Combine
g       = [g1;g2];

% Choose measurement angles (given in degrees, not radians). 
Nang    = 40; 
angle0  = -90;
ang = angle0 + [0:(Nang-1)]/Nang*180;

% % Initialize measurement matrix of size (M*P) x N^2, where M is the number of
% % X-ray directions and P is the number of pixels that Matlab's Radon
% % function gives.
% target = target1;
% P  = length(radon(target,0));
% M  = length(ang);
% A = sparse(M*P,N^2);
% 
% % Construct measurement matrix column by column. The trick is to construct
% % targets with elements all 0 except for one element that equals 1.
% for mmm = 1:M
%     for iii = 1:N^2
%         tmpvec                  = zeros(N^2,1);
%         tmpvec(iii)             = 1;
%         A((mmm-1)*P+(1:P),iii) = radon(reshape(tmpvec,N,N),ang(mmm));
%         if mod(iii,100)==0
%             disp([mmm, M, iii, N^2])
%         end
%     end
% end
% 
% % Test the result
% Rtemp = radon(target,ang);
% Rtemp = Rtemp(:);
% Mtemp = A*target(:);
% disp(['If this number is small, then the matrix A is OK: ', num2str(max(max(abs(Mtemp-Rtemp))))]);
% 
% % Save the result to file (with filename containing the resolution N)
% eval(['save RadonMatrix', num2str(N), ' A ang target N P Nang']);

% Load radonMatrix
eval(['load RadonMatrix', num2str(N), ' A ang target N P Nang']);
a = A;

% Simulate noisy measurements; here including inverse crime
m = A2x2mult(a,c11,c12,c21,c22,g);

% figure(5)
% imshow(reshape(m,122,40),[]);
% Add noise
%m       = m + noiselevel*max(abs(m(:)))*randn(size(m));
% Load the data
 m = m(:);
 tt = 1; % number of time steps
 b = A2x2Tmult(a,c11,c12,c21,c22,m);
 %systemMatrix = load('ASTRA_A_fanflat_numDetPix_280_objSize_64_angles_0_6_354_reorg');
 %A0 = a;

% % Collect different time steps into matrix A
%  A = [];
%  disp('Creating matrix A...')
%  for ii = 1:tt;
%     A = blkdiag(A,A0);
%     if mod(ii,10) == 0
%     disp([num2str(ii),'/',num2str(tt)]);
%     end
%  end
%  disp('Matrix A ready!');
%%
% Initial guess for u
 %g = zeros(size(A,2),1);
 ss = sqrt(size(g,1)/tt); 
% u = reshape(u,[ss,ss,tt]);

%alpha = 0.5; % regularization parameter

ht = 1; % adjusts the amount of regularization in temporal direction

% First iteration
graddu2 = [];
%u = reshape(u,[ss,ss,tt]);
for ii = 1:ss
    for jj = 1:ss
        for kk = 1:tt
            if ii == ss
                term1 = -1;
                uusterm1 = -1;%?
            elseif ii == ss-1
                term1 = g(ii+1,jj,kk);
                uusterm1 = -1;
            else
                term1 = g(ii+1,jj,kk);
                uusterm1 = g(ii+2,jj,kk);
            end
            if jj == ss
                term2 = -1;
                uusterm2 = -1;%?
            elseif jj == ss-1
                term2 = g(ii,jj+1,kk);
                uusterm2 = -1;
            else
                term2 = g(ii,jj+1,kk);
                uusterm2 = g(ii,jj+2,kk);
            end
            if ii == 1
                term3 = -1;
                uusterm3 = -1;
            elseif ii == 2
                term3 = g(ii-1,jj,kk);
                uusterm3 = -1;
            else
                term3 = g(ii-1,jj,kk);
                uusterm3 = g(ii-2,jj,kk);
            end
            if jj == 1
                term4 = -1;
                uusterm4 = -1;
            elseif jj == 2
                term4 = g(ii,jj-1,kk);
                uusterm4 = -1;
            else
                term4 = g(ii,jj-1,kk);
                uusterm4 = g(ii,jj-2,kk);
            end
            if kk == tt
                term5 = -1;
                uusterm5 = -1;
            elseif kk == tt-1
                term5 = g(ii,jj,kk+1);
                uusterm5 = -1;
            else
                term5 = g(ii,jj,kk+1);
                uusterm5 = g(ii,jj,kk+2);
            end
            if kk == 1
                term6 = -1;
                uusterm6 = -1;
            elseif kk == 2
                term6 = g(ii,jj,kk-1);
                uusterm6 = -1;
            else
                term6 = g(ii,jj,kk-1);
                uusterm6 = g(ii,jj,kk-2);
            end
            graddu2(ii,jj,kk) = (32+4/ht+12/ht^2)*g(ii,jj,kk)-...
                10*(term1+term2+...
                term3+term4)+...
                2*(uusterm1+uusterm2+uusterm3+uusterm4)-...
                ((2/ht)+8/ht^2)*(term5+term6)+...
                (2/ht^2)*(uusterm5+uusterm6);
        end
    end
end
disp('Creating gradient function...');
graddu2 = graddu2(:);
g = g(:);

% New part, building delta function for replacing F1 with F2
delta = 10^-1;
f_delta = @(tau) (tau>0).*sqrt(tau.^2+delta);
df_delta = @(tau) (tau>0).*tau./sqrt(tau.^2+delta);
g = g(:);
% Make a diagonal matrix
Df_delta = spdiags(df_delta(g),0,size(g,1),size(g,1));
%
%gradF2 = 2*Df_delta*(A'*(A*f_delta(u)))-2*Df_delta*(A'*m)+alpha1*graddu2;
%alpha matriisin tilalle: Reg_mat
Reg_mat = [alpha1*eye(N^2),zeros(N^2);zeros(N^2),alpha2*eye(N^2)];
gradF2     = A2x2Tmult(a,c11,c12,c21,c22,A2x2mult(a,c11,c12,c21,c22,g)) + Reg_mat*g;

disp('Gradient function ready!');

lam = .0001;

oldu = g;
g = g-lam*gradF2;

%% Iterate
disp('Iterating... ' );
for l = 1:7
graddu2 = [];
g = reshape(g,[ss,ss,tt]);
for ii = 1:ss
    for jj = 1:ss
        for kk = 1:tt
            if ii == ss
                term1 = -1;
                uusterm1 = -1;%?
            elseif ii == ss-1
                term1 = g(ii+1,jj,kk);
                uusterm1 = -1;
            else
                term1 = g(ii+1,jj,kk);
                uusterm1 = g(ii+2,jj,kk); % koska t‰ss‰ lis‰t‰‰n riviin2, menee yli jos on viimeist‰ edellinen rivi
            end
            if jj == ss
                term2 = -1;
                uusterm2 = -1;%?
            elseif jj == ss-1
                term2 = g(ii,jj+1,kk);
                uusterm2 = -1;
            else
                term2 = g(ii,jj+1,kk);
                uusterm2 = g(ii,jj+2,kk);
            end
            if ii == 1
                term3 = -1;
                uusterm3 = -1;
            elseif ii == 2
                term3 = g(ii-1,jj,kk);
                uusterm3 = -1;
            else
                term3 = g(ii-1,jj,kk);
                uusterm3 = g(ii-2,jj,kk);
            end
            if jj == 1
                term4 = -1;
                uusterm4 = -1;
            elseif jj == 2
                term4 = g(ii,jj-1,kk);
                uusterm4 = -1;
            else
                term4 = g(ii,jj-1,kk);
                uusterm4 = g(ii,jj-2,kk);
            end
            if kk == tt
                term5 = -1;
                uusterm5 = -1;
            elseif kk == tt-1
                term5 = g(ii,jj,kk+1);
                uusterm5 = -1;
            else
                term5 = g(ii,jj,kk+1);
                uusterm5 = g(ii,jj,kk+2);
            end
            if kk == 1
                term6 = -1;
                uusterm6 = -1;
            elseif kk == 2
                term6 = g(ii,jj,kk-1);
                uusterm6 = -1;
            else
                term6 = g(ii,jj,kk-1);
                uusterm6 = g(ii,jj,kk-2);
            end
            graddu2(ii,jj,kk) = (32+4/ht+12/ht^2)*g(ii,jj,kk)-...
                10*(term1+term2+...
                term3+term4)+...
                2*(uusterm1+uusterm2+uusterm3+uusterm4)-...
                ((2/ht)+8/ht^2)*(term5+term6)+...
                (2/ht^2)*(uusterm5+uusterm6);
        end
    end
end
graddu2 = graddu2(:);
g = g(:);
 
Df_delta = spdiags(df_delta(g),0,size(g,1),size(g,1));
oldgradF2 = gradF2;
%gradF2 = 2*A'*(A*u)-2*A'*m+alpha*graddu2;
%gradF2 = 2*(A'*(A*f_delta(u)))-2*(A'*m)+alpha1*graddu2;
%Nyt johonkin kertolaskuista tulisi lis‰t‰ tuo kertominen f_delta(u):lla
gradF2 = 2*A2x2Tmult_matrixfree(c11,c12,c21,c22,A2x2mult_matrixfree(c11,c12,c21,c22,g,ang,N),ang)-b+alpha1*g;
lam = (g-oldu)'*(g-oldu)/((g-oldu)'*(gradF2-oldgradF2));
oldu = g;
g = g-lam*gradF2;
disp([num2str(l),'/7']);
end
%% Show reconstruction
g = reshape(g,[ss,ss,tt]);
u2 = f_delta(g);
for ii = 1:tt
figure(1);
imagesc(imrotate(u2(:,:,ii),-98,'bilinear','crop'));
title(['Frame ',num2str(ii)]);
colormap jet;
axis square;
colorbar
pause(0.2);
%pause;
end
%% Normalize and watch as a black and white video
% u = u - min(u(:));
% u = u ./ max(u(:));
% colormap jet;
% implay(u);