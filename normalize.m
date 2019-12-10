function res = normalize(im)
% This function normalizes the given image (or array) im to have values on
% the interval [0,1]
%
% Esa Niemi May 2014
%Revisited by Salla 2019


res = im - min(im(:));
res = res / max(res(:));