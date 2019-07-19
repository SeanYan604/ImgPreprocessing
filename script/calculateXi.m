function [Xi] = calculateXi(mu,sigma,Q)
% Dividing the normal distributuion into Q sections, calculate each
% saction's border as Xi.
Xi = zeros(1,Q);
sum = 0; i = 1;
for t = 1:255
    if(sum >= (i/(Q+1)))
        Xi(i) = t;
        i = i+1;
        while(sum >= (i/(Q+1)))
            Xi(i) = t;
            i = i+1;
        end
        if(i == Q+1)
            break;
        end
        sum = sum + normpdf(t, mu, sigma);
    else
        sum = sum + normpdf(t, mu, sigma);
    end
end

