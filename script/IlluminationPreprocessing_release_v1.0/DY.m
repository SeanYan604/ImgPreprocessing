function resImg = DY(srcImg)

% face illumination preprocessing using DY method
% DY: directional gray-scale derivative in Y direction
% Basic principle: Gradient or edge extraction
% References: Y.Adini, Y.Moses, S.Ullman, Face recognition: the problem of compensating
%              for changes in illumination direction, IEEE Transactions on Pattern Analysis
%              and Machine Intelligence 19(1997)721¨C732.
%
% INPUT:
% srcImg: the input image
%
% OUTPUT:
% resImg: the DY image

[FX, FY] = gradient(double(srcImg));
resImg = uint8( mat2gray(FY) * 255 );
end
% eof