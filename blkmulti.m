% Auxiliary function for matrix multiplication
function res = blkmulti(A0,b)
res = [];
n = size(A0,2);% Eli matriisin sarakkeet
for ii = 1:(size(b,1)/n) % b:n rivieen m‰‰r‰ /A0:n sarakkeiden m‰‰r‰ll‰
    res = [res;A0*b(((ii-1)*n+1):ii*n)];
end
