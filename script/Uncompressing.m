function [output_img, output_mask] = Uncompressing(input_img, src_img)
% Extracting the square picture back to source image.

    [m,n] = size(src_img);
%     img_compressed = 255*ones(m,n);
    pix_vect = input_img(:);
    length = 0;
    [len,x] = size(pix_vect);
    output_img(:,:,1) = src_img;
    output_img(:,:,2) = src_img;
    output_img(:,:,3) = src_img;
    output_mask = zeros(m,n);
    for i=1:n
        for j=1:m
            if(src_img(j,i) > 0 && length < len)
                length = length + 1;
                if(pix_vect(length) > 0)
                    output_img(j,i,:) = [255,0,0];
                    output_mask(j,i) = 255;
                end
            end
        end        
    end
    

    
end

