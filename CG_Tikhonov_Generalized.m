% Example computations related to X-ray tomography. Here we apply Tikhonov 
% regularization and solve the normal equations using the conjugate 
% gradient method. The approach uses sparse matrix A and is much more
% efficient computationally than the singular value decomposition approach.
%
% Salla Latva-Ã„ijÃ¶ and Samuli Siltanen, November 2019
clear all;
% Measure computation time later; start clocking here
tic

% Regularization parameter
alpha1  = 10;%10;
alpha2  = 6;%6;
N       = 40;
iter    = 400;% maximum number of iterations

% Choose relative noise level in simulated noisy data
noiselevel = 0.001;
% Choose measurement angles (given in degrees, not radians). 
Nang    = 65; 
angle0  = -90;
ang = angle0 + [0:(Nang-1)]/Nang*180;

% Define coefficients: Iodine and Al
% c11     = 42.2057; %Iodine 30kV
% c21     = 60.7376; %Iodine 50kV
% c12     = 3.044;   %Al 30kV
% c22     = 0.994;   %Al 50kV

% Define attenuation coefficients: Iodine and PVC
material1='PVC';
material2='Iodine';
c11    = 42.2057; %Iodine 30kV
c21    = 60.7376; %Iodine 50kV
c12    = 2.096346;%PVC 30kV
c22    = 0.640995;%PVC 50kV

%Huonommin toimivat materiaalit?
% material1='Iodine';
% material2='bone';
% c12    = 37.57646; %Iodine 30kV
% c22    = 32.404; %Iodine 50kV
% c11    = 2.0544;%Bone 30kV
% c21    = 0.448512;%Bone 50kV

% Construct phantom. You can modify the resolution parameter N.
%M1 = imresize(double(imread('HY_Al.bmp')),[N N]);
%M2=scale01(M2);
%M2 = imresize(double(imread('HY_square_inv.jpg')),[N N]);
M1 = imresize(double(imread('new_HY_material_one_bmp.bmp')), [N N]);
M2 = imresize(double(imread('new_HY_material_two_bmp.bmp')), [N N]);
%M1 = imresize(double(imread('selkaranka_phantom.jpg')), [N N]);
%M2 = imresize(double(imread('selkaranka_phantom_nurin.jpg')), [N N]);
M1=M1(:,:,1);
M2=M2(:,:,1);

% % Try to normalize the image between 0 and 255
% min1=min(min(M1));
% max1=max(max(M1));
% M1 = double(255 .* ((double(M1)-double(min1))) ./ double(max1-min1));
% 
% min1=min(min(M2));
% max1=max(max(M2));
% M2 = double(255 .* ((double(M2)-double(min1))) ./ double(max1-min1));

%M1=scale01(M1);
%Take away negative pixels
%M1 = max(M1,0);
%M2 = max(M2,0);

% Vektorize
g1      = M1(:);
g2      = M2(:);
% Combine
g       = [g1;g2];
%g=g/normest(g);%Taas harmaa kuva
% % Initialize measurement matrix of size (M*P) x N^2, where M is the number of
% % X-ray directions and P is the number of pixels that Matlab's Radon
% % function gives.
% target = M1;
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
%a=A;
% Simulate noisy measurements; here including inverse crime
m = A2x2mult(a,c11,c12,c21,c22,g);
%m=m/normest(m);%Ei toiminut, tuotti harmaan kuvan ja ison virheen
% Add noise
m = m + noiselevel*max(abs(m(:)))*randn(size(m));

% Solve the minimization problem
%         min (x^T H x - 2 b^T x), 
% where 
%         H = A^T A + alpha*I
% and 
%         b = A^T mn.
% The positive constant alpha is the regularization parameter
b = A2x2Tmult(a,c11,c12,c21,c22,m);
%%
% Solve the minimization problem using conjugate gradient method.
% See Kelley: "Iterative Methods for Optimization", SIAM 1999, page 7.
g   = zeros(2*N*N,1); % initial iterate is zero matrix (before it was b, backprojected data);  % initial iterate is the backprojected data
rho = zeros(iter,1); % initialize parameters

% Compute residual using sparse matrices. NOTE CAREFULLY: it is important
% to write (A.')*(A*x) on the next line instead of ((A.')*A)*x, 
% because (A.')*A may be a full matrix and in that case we lose 
% the advantage of the iterative solution method!

% very old version: Hx     = (A.')*(A*x) + alpha*x; 
% second try:       Hx     = (A.')*Amult(A,x) + alpha*x;

%alpha matriisin tilalle: Reg_mat
Reg_mat = [alpha1*eye(N^2),zeros(N^2);zeros(N^2),alpha2*eye(N^2)];
%Tämän tilalle laitetaaankin L matriisi!!!
%L = [alpha1*-eye(N^2)+diag(ones(1,N^2-1),1),zeros(N^2);zeros(N^2),alpha2*-eye(N^2)+diag(ones(1,N^2-1),1)];
%L(end,1)=1; % Tässä oli jossain kohdassa nolla väärin

L1 = [alpha1*-eye(N^2)+diag(ones(1,N^2-1),1),zeros(N^2)];
L1(end,1)=1;
L2 = [zeros(N^2),alpha2*-eye(N^2)+diag(ones(1,N^2-1),1)];
L2(end,1)=1;

L=[L1;L2];

% ***New***
%Q2 = [alpha*eye(N^2),beta*eye(N^2);beta*eye(N^2),alpha*eye(N^2)]
Hg     = A2x2Tmult(a,c11,c12,c21,c22,A2x2mult(a,c11,c12,c21,c22,g)) + L*g;

r      = b-Hg;
rho(1) = r(:).'*r(:);

% Start iteration
%figure(1);
for kkk = 1:iter
    if kkk==1
        p = r;
    else
        beta = rho(kkk)/rho(kkk-1);
        p    = r + beta*p;
    end
    w         = A2x2Tmult(a,c11,c12,c21,c22,A2x2mult(a,c11,c12,c21,c22,p));
    w          = w + Reg_mat*p;
    aS         = rho(kkk)/(p.'*w);
    g          = g + aS*p;
    r          = r - aS*w;
    rho(kkk+1) = r(:).'*r(:);
    %disp([kkk K])
%   figure(1)
%     recn1 = g(1:1600);
%     recn1 = reshape(recn1,N,N);
%     imshow(recn1,[]);
%     pause(0.2);
%   recn2 = g(1601:3200);
%   recn2 = reshape(recn2,N,N);
%   imshow(recn2,[]);
%  pause;
end
CGM1 = reshape(g(1:(end/2),1:end),N,N);
CGM2 = reshape(g((end/2)+1:end,1:end),N,N);

% Determine computation time
comptime = toc;

% Compute relative errors
%err_sup1 = max(max(abs(g1-recn1(:))))/max(max(abs(g1)));
err_CGM1 = norm(M1(:)-CGM1(:))/norm(M1(:));
%err_sup2 = max(max(abs(g2-recn2(:))))/max(max(abs(g2)));
err_CGM2 = norm(M2(:)-CGM2(:))/norm(M2(:));

% Save result to file
% eval(['save XRMG_Tikhonov', num2str(N), ' recn alpha target comptime err_sup err_squ']);

%% Take a look at the results
figure(3);
% Original target1
subplot(2,2,1);
imagesc(reshape(M1,N,N));
colormap gray;
axis square;
axis off;
title({'CGTik +matrix, PVC'});
% Reconstruction of target1
subplot(2,2,2)
imagesc(CGM1);
colormap gray;
axis square;
axis off;
title(['Relative error=', num2str(err_CGM1), ', \alpha_1=', num2str(alpha1),',\alpha_2=' num2str(alpha2)]);
% Original target2
subplot(2,2,3)
imagesc(reshape(M2,N,N));
colormap gray;
axis square;
axis off;
title({'CGTik +matrix, Iodine'});
% Reconstruction of target2
subplot(2,2,4)
imagesc(CGM2);
colormap gray;
axis square;
axis off;
title(['Relative error=' num2str(err_CGM2), ', iter=' num2str(iter)]);

% Save the result to disc 
save('from_CG_Tik_with_matrix', 'CGM1', 'CGM2', 'err_CGM1', 'err_CGM2');

% %% Level Set reconstruction of the image
% %
% %Load the data
% %loadcommand = ['load cheese_NoCrime_', num2str(M), '_', num2str(Nang), ' A M n Nang angles'];
% %eval(loadcommand)
% 
% tt = 1;
% 
% % initial guess for u
% u = zeros(size(A,2)*tt,1);
% ss = sqrt(size(u,1)/tt);
% u=reshape(u, [ss,ss,tt]);
% 
% alpha = 5;
% 
% ht = 1;
% 
% % First iteration
% graddu2 = [];
% graddu2=(8+4/ht)*u-2*([u(2:end,:,:);zeros(1,ss,tt)]+[u(:,2:end,:),...
% zeros(ss,1,tt)]+[zeros(1,ss,tt);u(1:end-1,:,:)]+[zeros(ss,1,tt),...
% u(:,1:end-1,:)])-(2/ht)*(cat(3,u(:,:,2:end),zeros(ss,ss,1))+...
% cat(3,zeros(ss,ss,1),u(:,:,1:end-1)));
% disp('Creating gradient function...');
% graddu2 = graddu2(:);
% u = u(:);
% 
% % Build delta function
% delta = 10^-1;
% f_delta = @(tau) (tau>0).*sqrt(tau.^2+delta);
% df_delta = @(tau) (tau>0).*tau./sqrt(tau.^2+delta);
% u = u(:);
% % Calculate the gradient fuction
% gradF2 = 2*blkmulti(A',blkmulti(A,f_delta(u)))-2*blkmulti(A',m)+alpha*graddu2;
% disp('Gradient function ready!');
% 
% lam = .0001;
% 
% oldu = u;
% u = u-lam*gradF2;
% 
% %% Iterate
% % Number of iterations 
% iter = 50;
% disp('Iterating... ' );
% for l = 1:iter
% graddu2 = [];
% u = reshape(u,[ss,ss,tt]);
% for ii = 1:ss
%     for jj = 1:ss
%         for kk = 1:tt
%             
%             if ii == ss
%                 term1 = -1;
%                 uusterm1 = -1;%?
%             elseif ii == ss-1
%                 term1 = u(ii+1,jj,kk);
%                 uusterm1 = -1;
%             else
%                 term1 = u(ii+1,jj,kk);
%                 uusterm1 = u(ii+2,jj,kk); % koska tässä lisätään riviin2, menee yli jos on viimeistä edellinen rivi
%             end
%             
%             if jj == ss
%                 term2 = -1;
%                 uusterm2 = -1;%?
%             elseif jj == ss-1
%                 term2 = u(ii,jj+1,kk);
%                 uusterm2 = -1;
%             else
%                 term2 = u(ii,jj+1,kk);
%                 uusterm2 = u(ii,jj+2,kk);
%             end
%             
%             if ii == 1
%                 term3 = -1;
%                 uusterm3 = -1;
%             
%             elseif ii == 2
%                 term3 = u(ii-1,jj,kk);
%                 uusterm3 = -1;
%             else
%                 term3 = u(ii-1,jj,kk);
%                 uusterm3 = u(ii-2,jj,kk);
%             end
%             if jj == 1
%                 term4 = -1;
%                 uusterm4 = -1;
%             elseif jj == 2
%                 term4 = u(ii,jj-1,kk);
%                 uusterm4 = -1;
%             else
%                 term4 = u(ii,jj-1,kk);
%                 uusterm4 = u(ii,jj-2,kk);
%             end
%             
%             
%             if kk == tt
%                 term5 = -1;
%                 uusterm5 = -1;
%             elseif kk == tt-1
%                 term5 = u(ii,jj,kk+1);
%                 uusterm5 = -1;
%             else
%                 term5 = u(ii,jj,kk+1);
%                 uusterm5 = u(ii,jj,kk+2);
%             end
%             if kk == 1
%                 term6 = -1;
%                 uusterm6 = -1;
%             elseif kk == 2
%                 term6 = u(ii,jj,kk-1);
%                 uusterm6 = -1;
%             else
%                 term6 = u(ii,jj,kk-1);
%                 uusterm6 = u(ii,jj,kk-2);
%             end
%             
%             
%             if ii == ss
%                 mixterm1 = -1;
%             elseif jj == ss
%                 mixterm1 = -1;
%             else
%                 mixterm1 = u(ii+1,jj+1,kk);
%             end
%             
%             
%             if ii == ss
%                 mixterm2 = -1;
%             elseif jj == 1
%                 mixterm2 = -1;
%             else 
%                 mixterm2 = u(ii+1,jj-1,kk);
%             end
%             
%             if ii ==1
%                 mixterm3 = -1;
%             elseif jj == ss
%                 mixterm3 = -1;
%             else
%                 mixterm3 = u(ii-1,jj+1,kk);
%             end
%             
%             if ii == 1
%                 mixterm4 = -1;
%             elseif jj == 1
%                 mixterm4 = -1;
%             else
%                 mixterm4 = u(ii-1,jj-1,kk);
%             end
%                 
%             graddu2(ii,jj,kk) = (40+16/ht^2)*u(ii,jj,kk)-...
%                 14*(term1+term2+...
%                 term3+term4)+...
%                 2*(uusterm1+uusterm2+uusterm3+uusterm4)-...
%                 (10/ht^2)*(term5+term6)+...
%                 (2/ht^2)*(uusterm5+uusterm6)+2*(mixterm1+mixterm2+mixterm3+mixterm4);
%         end
%     end
% end
% graddu2 = graddu2(:);
% u = u(:);
%  
% Df_delta = spdiags(df_delta(u),0,size(u,1),size(u,1));
% oldgradF2 = gradF2;
% %gradF2 = 2*A'*(A*u)-2*A'*m+alpha*graddu2;
% %gradF2 = 2*(A'*(A*f_delta(u)))-2*(A'*m)+alpha*graddu2; % Tästäkö puuttuu
% %Df_delta? näin: gradF2 =
% %2*Df_delta*(A'*(A*f_delta(u)))-2*Df_delta*(A'*m)+alpha*graddu2; Kommentoin
% %tämän pois
% % Vaihdoin sen näin niin toimi näinkin:
% gradF2 = 2*Df_delta*(A'*(A*f_delta(u)))-2*Df_delta*(A'*m)+alpha*graddu2;
% lam = (u-oldu)'*(u-oldu)/((u-oldu)'*(gradF2-oldgradF2));
% oldu = u;
% %u = max(0,u-lam*gradF2);
% u = u-lam*gradF2;
% %fprintf('Iteraatiosarja 2...  \n');
% disp([num2str(l),'/',num2str(iter)]);
% % %% Show reconstruction
% % u = reshape(u,[ss,ss,tt]);
% % u_uusi = reshape(u,[ss,ss,tt]);
% % u2 = f_delta(u);
% % u_uusi2 = f_delta(u_uusi);
% % for ii = 1:tt
% % figure(1);
% % imagesc(u2(:,:,ii));
% % %imagesc(u_uusi2(:,:,ii));
% % title(['Frame ',num2str(l)]);
% % axis square;
% % colormap jet; 
% % %colormap gray
% % colorbar
% % %pause(0.5);
% % pause;
% % end
% end
% %% Show reconstruction
% u = reshape(u,[ss,ss,tt]);
% u2 = f_delta(u);
% for ii = 1:tt
% figure(1);
% imagesc(u2(:,:,ii));
% title(['Frame ',num2str(ii)]);
% axis square;
% colormap jet; 
% %colormap gray
% colorbar
% pause(0.1);
% %pause;
% end