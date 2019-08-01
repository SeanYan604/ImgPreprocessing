function img_out = LT(img_in)

% face illumination preprocessing using LT method
% LT: logarithmic transform
% Basic principle: Gray-level transformation
% References: Y.Adini, Y.Moses, S.Ullman, Face recognition: the problem of compensating
%             for changes in illumination direction, IEEE Transactions on Pattern Analysis
%             and Machine Intelligence 19(1997)721¨C732.
%
% INPUT:
% img_in: the input image
%
% OUTPUT:
% img_out: the LT image
%

img_data = log  ( double(img_in) + double(1) );
img_out  = uint8( 255 * mat2gray(img_data) );
end % end of function
% eof