function [result] = imgDownSample(input,kernal_size,m,n,type)
% type has tree options:
%       "average": average pooling;
%       "max"    : max pooling;
%       "center" : just get center pix;
    if(kernal_size / 2 == 0)
        kernal_size = kernal_size + 1;
    end
    bias = floor(kernal_size/2);
    
    if(type == "average")
        h_stride = floor(m / kernal_size);
        w_stride = floor(n / kernal_size);
        result = zeros(h_stride, w_stride);
        x = 1+bias; y = 1+bias;
        for i = 1:h_stride
            for j = 1:w_stride
                temp = input(y-bias:y+bias,x-bias:x+bias);
                result(i,j) = sum(sum(temp))/(kernal_size*kernal_size);
                x = x + kernal_size;
            end
            x = 1 + bias;
            y = y + kernal_size;
        end
    elseif(type == "max")
        h_stride = floor(m / kernal_size);
        w_stride = floor(n / kernal_size);
        result = zeros(h_stride, w_stride);
        x = 1+bias; y = 1+bias;
        for i = 1:h_stride
            for j = 1:w_stride
                temp = input(y-bias:y+bias,x-bias:x+bias);
                result(i,j) = max(max(temp));
                x = x + kernal_size;
            end
            x = 1 + bias;
            y = y + kernal_size;
        end
    elseif(type == "center")
        h_stride = floor(m / kernal_size);
        w_stride = floor(n / kernal_size);
        result = zeros(h_stride, w_stride);
        x = 1+bias; y = 1+bias;
        for i = 1:h_stride
            for j = 1:w_stride
                result(i,j) = input(y,x);
                x = x + kernal_size;
            end
            x = 1 + bias;
            y = y + kernal_size;
        end
    else
        h_stride = floor(m / kernal_size);
        w_stride = floor(n / kernal_size);
        result = zeros(h_stride, w_stride);
    end
    
end

