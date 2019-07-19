function [pix_vect] = pixExtraction(src)
% Extract each roi pixes value
[m,n] = size(src);
pix_vect = zeros(m*n,1);
len = 0;
for i=1:n
    for j=1:m
        if(src(j,i) > 0)
            len = len +1;
            pix_vect(len) = src(j,i);
        end
    end
end
pix_vect = pix_vect(1:len);
end

