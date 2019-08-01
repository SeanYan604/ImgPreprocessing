function img_out = GIC(img_in, localMode, numY_pat, numX_pat, cordY_pat, cordX_pat, sizeY_pat, sizeX_pat)

% face illumination preprocessing using GIC method
% GIC: gamma intensity correction
% Basic principle: Gray-level transformation
% References: S.Shan, W.Gao, B.Cao, D.Zhao, Illumination normalization for robust face
%             recognition against varying lighting conditions, in: Proceedings of the ICCV
%             Workshop on Analysis and Modeling of Faces and Gestures, 2003, pp. 157¨C164.
%
% INPUT:
% img_in: the input image
% localMode:
%   0: perform GIC in holistic manner, compare the entire image with the
%   reference image
%   1: perform GIC in local manner, compare each patch with the
%   reference patch
% numY_pat: number of patches in Y direction
% numX_pat: number of patches in X direction
% cordY_pat: left most coordinate of every patch
% cordX_pat: top most coordinate of every patch
% sizeY_pat: number of columns in a patch, width
% sizeX_pat: number of rows in a patch, height
%
% OUTPUT:
% img_out: the GIC image


img_out = zeros(size(img_in), 'uint8');
nTotalChannel = size(img_in, 3);

load('GIC_Canonical_PIEIllum_64x80.mat', 'GIC_Canonical_64x80');
can_img = GIC_Canonical_64x80;
can_img = mat2gray(can_img);

gamma_val_vec = [10/1, 10/2, 10/4, 10/6, 10/8, 10/10, 8/10,  6/10,  4/10,  2/10,  1/10];
num_gamma = length(gamma_val_vec);

for inx_Channel = 1 : nTotalChannel
    in_cha = img_in(:, :, inx_Channel);
    [hei, wid] = size(in_cha);
    
    if ~localMode
        % normlized current channel to [0, 1]
        in_cha = double( mat2gray(in_cha) );
        
        in_cha_vec = reshape(in_cha, hei * wid, 1);
        res_cha_mat = zeros(hei * wid, num_gamma);
        for index = 1 : num_gamma
            gamma_val = gamma_val_vec(index);
            res_cha_mat(:, index) = in_cha_vec .^ gamma_val;
        end
        clear in_cha_vec;
        
        can_img_vec = reshape(can_img, hei * wid, 1);
        can_img_mat = repmat (can_img_vec, 1, num_gamma);
        clear can_img;
        clear can_img_vec;
        
        diff_mat = (res_cha_mat - can_img_mat) .^ 2;
        clear res_cha_mat;
        clear can_img_mat;
        
        diff = sum(diff_mat);
        clear diff_mat;
        [min_val, min_inx] = min(diff);
        clear diff;
        img_res = in_cha .^ gamma_val_vec(min_inx);
        %fprintf([num2str(gamma_val_vec(min_inx)) '\n\n']);
        img_res = reshape(img_res, hei, wid);
        img_out(:, :, inx_Channel) = uint8( mat2gray(img_res) * 255.0 );
        clear img_res;
    else
        % normlized current channel to [0, 1]
        in_cha = double( mat2gray(in_cha) );
        
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
                
                % canonical mat
                can_pat = can_img(y1: y2, x1 : x2);
                can_pat_vec = reshape(can_pat, sizeY_pat(inx) * sizeX_pat(inx), 1);
                clear can_pat;
                can_pat_mat = repmat(can_pat_vec, 1, num_gamma);                
                clear can_pat_vec;
                                                
                rst_pat_mat = zeros(sizeY_pat(inx) * sizeX_pat(inx), num_gamma);
                for index = 1 : num_gamma
                    gamma_val = gamma_val_vec(index);
                    rst_pat_mat(:, index) = in_pat_vec .^ gamma_val;
                end
                clear in_pat_vec;
                
                diff_mat = (rst_pat_mat - can_pat_mat) .^ 2;
                clear rst_pat_mat;
                clear can_pat_mat;
                diff = sum(diff_mat);
                clear diff_mat;
                [min_val, min_inx] = min(diff);
                clear min_val;
                clear diff;
                
                res_pat = in_pat .^ gamma_val_vec(min_inx);
                clear in_pat;
                res_cha(y1: y2, x1 : x2) = res_cha(y1: y2, x1 : x2) + res_pat;
                clear res_pat;
                counter(y1: y2, x1 : x2) = counter(y1: y2, x1 : x2) + 1;
            end
        end
        res_cha = res_cha ./ counter;
        img_out(:, :, inx_Channel) = uint8( mat2gray(res_cha) * 255.0 );
        clear res_cha;
        clear counter;
    end % end of else    
end % end of inx_channel

end % end of function
% eof