function [img_compressed] = CompressImg(target_img)
% Compressing source image into a square picture.
    
    [m,n] = size(target_img);
%     img_compressed = 255*ones(m,n);
    pix_vect = zeros(m*n,1);
    length = 0;
    
    for i=1:n
        for j=1:m
            if(target_img(j,i) > 0)
                length = length + 1;
                pix_vect(length) = target_img(j,i);
            end
        end        
    end
    
%     if(rem(length,2) ~= 0 )
%         length = length -1;
%     end
    a = floor(sqrt(length));
    for k = a:-1:1
        if(rem(length,k) <= 4)
            h = k;
            w = floor(length/k);
            break;
        end
    end
    img_compressed = zeros(h, w);
    for i=1:w
        img_compressed(:,i) = pix_vect(h*(i-1)+1:h*i);
    end

end

