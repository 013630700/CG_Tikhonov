function [ res ] = A2x2mult( A,c11,c12,c21,c22,g1,g2)
% Make ATA * g
res = [(c11.^2 + c21.^2)*(A.')*(A*g1(:))+(c11*c12+c21*c22)*(A.')*(A*g2(:)); (c12*c11+c22*c21)*(A.')*(A*g1(:))+(c12.^2+c22.^2)*(A.')*(A*g2(:))];
end


% Tässä käytin vahingossa isoa Ata A_beautifulin sijaanThis function makes the combination matrix and multiplies it with vector
% image vector g
%A_Big = [c11*A c12*A; c21*A c22*A];
%AT = [c11*A.' c12*A.'; c21*A.' c22*A.'];
%ATA = AT*A_Big;
%res = [(c11^2 + c21^2)*ATA*g1(:)+(c11*c12+c21*c22)*ATA*g2(:); (c12*c11+c22*c21)*ATA*g1(:)+(c12^2+c22^2)*ATA*g2(:)];

% Tässä yritin tehdä yksinkertaistetun version, mutta g:n dimensiota tuli liikaa A = [c11*A c12*A; c21*A c22*A];
% A = [c11*A c12*A; c21*A c22*A]; 
% g = [g1(:); g2(:)];
% res = A*g;