function resImg = LoG(srcImg)

% face illumination preprocessing using LoG method
% LoG: Laplacian of Gaussian
% Basic principle: Gradient or edge extraction
% References: Y.Adini, Y.Moses, S.Ullman, Face recognition: the problem of compensating
%             for changes in illumination direction, IEEE Transactions on Pattern Analysis
%             and Machine Intelligence 19(1997)721¨C732.
%
% INPUT:
% srcImg: the input image
%
% OUTPUT:
% resImg: the LoG image

hsize = 9;
sigma = (hsize/2 - 1)*0.3 + 0.8;

%% 64x80
% sigma = 3.4;
% hsize = ((sigma - 0.8)/0.3 + 1) * 2;
% hsize = double( round(hsize) );

h = fspecial('log', hsize, sigma);
resImg = filter2(h, double(srcImg));
resImg = uint8( mat2gray(resImg) * 255 );
end
% eof