function img_out = LN(img_in, localMode, numY_pat, numX_pat, cordY_pat, cordX_pat, sizeY_pat, sizeX_pat)

% face illumination preprocessing using LN method
% LN: Local normalization
% Basic principle: Reflectance field estimation
% References: X.Xie, K.Lam, An efficient illumination normalization method for face
%             recognition, Pattern Recognition Letters 27(2006)609¨C617.
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
% img_out: the LN image


img_out = zeros(size(img_in), 'uint8');
nTotalChannel = size(img_in, 3);
for inx_Channel = 1 : nTotalChannel
    in_cha = double( img_in(:, :, inx_Channel) );
    [hei, wid] = size(in_cha);
    
    if ~localMode   % holistic approach
        inImgVec = double( reshape(img_in, hei * wid, 1) );
        inImgMean = mean(inImgVec);
        inImgStd  = std (inImgVec);
        if inImgStd < 1e-6
            inImgStd = 1e-6;
        end
        inImgVec = (inImgVec - inImgMean) / inImgStd;
        ouImg = reshape(inImgVec, hei, wid);
        img_out = uint8( mat2gray(ouImg) * 255);
    else        %localization for holistic approach
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
                
                % input pat
                in_pat = in_cha(y1: y2, x1 : x2);
                in_pat_vec = reshape (in_pat, sizeY_pat(inx) * sizeX_pat(inx), 1);
                clear in_pat;
                pat_mean = mean(in_pat_vec);
                pat_std  = std (in_pat_vec);
                
                % handle std = 0
                if pat_std == 0
                    pat_std = 1;
                end
                
                res_pat_vec = (in_pat_vec - pat_mean) / pat_std;
                clear pat_mean;
                clear pat_std;
                clear in_pat_vec;
                
                res_pat = reshape(res_pat_vec, sizeY_pat(inx), sizeX_pat(inx));
                clear res_pat_vec;
                
                res_cha(y1: y2, x1 : x2) = res_cha(y1: y2, x1 : x2) + res_pat;
                counter(y1: y2, x1 : x2) = counter(y1: y2, x1 : x2) + 1;
                clear res_pat;
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