function [Xi] = calculateXi_(input_values,Q)
% When the distribution is not Guassian Distribution, in the same way, we
% divide it into Q sections, each section's border is returned as Xi.

Xi = zeros(1,Q);
i = 1;
total_pix = sum(input_values);
sum_pix = 0;
[x,len] = size(input_values);
for t = 1:len
    sum_rate = sum_pix/total_pix;
    if(sum_rate >= (i/(Q+1)))
        Xi(i) = t;
        i = i+1;
        while(sum_rate >= (i/(Q+1)))
            Xi(i) = t;
            i = i+1;
        end
        if(i == Q+1)
            break;
        end
        sum_pix = sum_pix + input_values(t);
    else
        sum_pix = sum_pix + input_values(t);
    end
end

end

