function img_out = HE(img_in, localMode, numY_pat, numX_pat, cordY_pat, cordX_pat, sizeY_pat, sizeX_pat)

% face illumination preprocessing using HE method
% HE: histogram equalization
% Basic principle: Gray-level transformation
% References: S.M.Pizer, E.P.Amburn, J.D.Austin, R.Cromartie, A.Geselowitz, T.Greer,  B.H. Romeny, 
%             J.B.Zimmerman, K.Zuiderveld, Adaptive histogram equalization and its variations, 
%             Computer Vision, Graphic, and Image Processing 39 (1987) 355¨C368.
%
% INPUT:
% img_in: the input image
% localMode:
%   0: holistic approach
%   1: localization for holistic approach
% numY_pat: number of patches in Y direction
% numX_pat: number of patches in X direction
% cordY_pat: left most coordinate of every patch
% cordX_pat: top most coordinate of every patch
% sizeY_pat: number of columns in a patch, width
% sizeX_pat: number of rows in a patch, height
%
% OUTPUT:
% img_out: the HE image


img_out = zeros(size(img_in), 'uint8');
nTotalChannel = size(img_in, 3);

for inx_Channel = 1 : nTotalChannel
    in_cha = img_in(:, :, inx_Channel);
    [hei, wid] = size(in_cha);

    if ~localMode        % holistic approach
        img_out(:, :, inx_Channel) = histeq(in_cha);
        clear in_cha;
    else                % localization for holistic approach
        res_cha = zeros(hei, wid);
        counter = zeros(hei, wid);
        inx = 0;
        for i = 1 : numY_pat
            for j = 1 : numX_pat
                inx = inx + 1;
                y1 = cordY_pat(inx);
                x1 = cordX_pat(inx);
                y2 = cordY_pat(inx) + sizeY_pat(inx) - 1;
                x2 = cordX_pat(inx) + sizeX_pat(inx) - 1;
                
                % a patch
                in_pat = in_cha(y1: y2, x1 : x2);
                % HE patch
                res_pat = double( histeq(in_pat) );
                clear in_pat;
                
                res_cha(y1: y2, x1 : x2) = res_cha(y1: y2, x1 : x2) + res_pat;
                clear res_pat;
                counter(y1: y2, x1 : x2) = counter(y1: y2, x1 : x2) + 1;
            end % end of j
        end % end of i
        res_cha = res_cha ./ counter;
        img_out(:, :, inx_Channel) = uint8( mat2gray(res_cha) * 255.0 );
        clear res_cha;
        clear counter;        
    end % end of else
end % end of channel
end % end of function

% eof