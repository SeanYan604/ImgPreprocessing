function img_out = DCTN(img_in, Cov2LOG, blkPattern, numY_pat, numX_pat, cordY_pat, cordX_pat, sizeY_pat, sizeX_pat)

% face illumination preprocessing using LDCT method
% DCTN: Discrete cosine transforms normalization
% Basic principle: Reflectance field estimation
% References: W.Chen, M.J.Er, S.Wu, Illumination compensation and normalization for
%              robust face recognition using discrete cosine transform in logarithm domain,
%              IEEE Transactions on Systems, Man, and Cybernetics PartB: Cybernetics 36 (2006) 458¨C466.
%
% INPUT:
% img_in: the input image
% Cov2LOG: whether use logarithmic operation (true or false)
% blkPattern: 
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
% img_out: the LDCT image


[row, col] = size(img_in);

if blkPattern == 0
    if Cov2LOG
        log_img = double(img_in);
        log_img = log(log_img + 1);
        C = dct2(log_img);
        
        mu   = mean(mean(log_img));
              
        Ddis = 15;
        %Ddis = 22 * sqrt( double(row * row + col * col) ) / sqrt(120.0 * 120.0 + 105.0 * 105.0);
    else
        C = dct2(double(img_in));
        
        mu   = mean(mean(img_in));
        Ddis = 22 * sqrt( double(row * row + col * col) ) / sqrt(120.0 * 120.0 + 105.0 * 105.0);
    end    
     for i = 1 : row
         for j = 1 : col
             if (i + j) <= (Ddis + 1)
                 C(i, j) = 0;
             end
         end
     end
    
    C(1, 1) = log(mu) * sqrt(row * col);
    img_rst = idct2(C);
    img_rst = mat2gray(img_rst) * 255.0;
    img_out = uint8(img_rst);
else
    res_img = zeros(row, col);
    counter = zeros(row, col);
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
            [pat_row, pat_col] = size(in_pat);
            
            %pat_mu = mean( reshpae(in_pat, pat_row * pat_col, 1) );
            pat_mu = mean(mean(in_pat));
            Ddis = 22 * sqrt(pat_row * pat_row + pat_col * pat_col) / sqrt(120.0 * 120.0 + 105.0 * 105.0);
            
            C = dct2(in_pat);
            for k = 1 : row
                for l = 1 : col
                    if (k + l) <= (Ddis + 1)
                        C(k, l) = 0;
                    end
                end
            end
            C(1, 1) = log(pat_mu) * sqrt(pat_row * pat_col);

            res_pat = idct2(C);

            res_img(y1: y2, x1 : x2) = res_img(y1: y2, x1 : x2) + res_pat;
            counter(y1: y2, x1 : x2) = counter(y1: y2, x1 : x2) + 1;
        end
    end
    res_img = res_img ./ counter;
    img_out = uint8( mat2gray(res_img) * 255.0 );
    
    clear res_img;
    clear counter;
end
% eof
