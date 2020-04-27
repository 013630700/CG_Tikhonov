%%% PLOT MULTI-ENERGY SUPER RESULTS
%
%
% 
% Load precomputed results of the routine (commented above)
clear all;
loadcommand = ['load from_BB_Tik_with_matrix'];
eval(loadcommand)

loadcommand = ['load from_BB_Tik_matrixfree'];
eval(loadcommand)

loadcommand = ['load from_CG_Tik_with_matrix'];
eval(loadcommand)

loadcommand = ['load from_CG_Tik_matrixfree'];
eval(loadcommand)

figure(1)

%%
%%%%%%%%%%%%%%%%%%% MATERIAL M1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original M1, original Iodine phantom
subplot(2,5,1);
imagesc(M1);
colormap gray;
axis square;
axis off;
title({'Iodine'});

% Reconstruction of M1 (Barzilein Borwein Tikhonov with matrix) BBM
subplot(2,5,2)
imagesc(BBM1); 
colormap gray;
axis square;
axis off;
title(['BBM, err = ', num2str(err_BBM1)]);

% Reconstruction of M1 (Barzilain Borwein Tikhonov, matrix free) BB
subplot(2,5,3)
imagesc(BB1);
colormap gray;
axis square;
axis off;
title(['BB, err = ', num2str(err_BB1)]);

% Reconstruction of M1 (CG Tikhonov with Matrix) CGM
subplot(2,5,4)
imagesc(CGM1);
colormap gray;
axis square;
axis off;
title(['CGM, err = ', num2str(err_CGM1)]);

% Reconstruction of M1 (CG Tikhonov, matrix free) CG
subplot(2,5,5)
imagesc(CG1);
colormap gray;
axis square;
axis off;
title(['CG, err = ', num2str(err_CG1)]);

%%
%%%%%%%%%%%%%% MATERIAL M2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Original M2, (original PVC Phantom)
subplot(2,5,6)
imagesc(M2);
colormap gray;
axis square;
axis off;
title({'PVC'});

% Reconstruction of M2 (Barzilain Borwein Tikhonov with matrix) BBM
subplot(2,5,7)
imagesc(BBM2);
colormap gray;
axis square;
axis off;
title(['BBM, err = ', num2str(err_BBM2)]);

% Reconstruction of M2 (Barzilain Borwein Tikhonov, matrix free) BB
subplot(2,5,8)
imagesc(BB2);
colormap gray;
axis square;
axis off;
title(['BB, err = ', num2str(err_BB2)]);

% Reconstruction of M2 (Conjugate Gradient Tikhonov with matrix) CGM
subplot(2,5,9)
imagesc(CGM2);
colormap gray;
axis square;
axis off;
title(['CGM, err = ', num2str(err_CGM2)]);

% Reconstruction of M2 (Conjugate Gradient Tikhonov, matrix free) CG
subplot(2,5,10)
imagesc(CG2);
colormap gray;
axis square;
axis off;
title(['CG, err = ', num2str(err_CG2)]);