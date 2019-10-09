function [Xi] = calculateXi_tri(mu,sigma,alpha,Q)
% Dividing the normal distributuion into Q sections, calculate each
% saction's border as Xi.
Xi = zeros(1,Q);
sum_ = 0; i = 1;
for t = 1:255
    if(sum_ >= (i/(Q+1)))
        Xi(i) = t;
        i = i+1;
        while(sum_ >= (i/(Q+1)))
            Xi(i) = t;
            i = i+1;
        end
        if(i == Q+1)
            break;
        end
        sum_ = sum_ + (alpha(1)*normpdf(t, mu(1), sigma(1))+alpha(2)*normpdf(t, mu(2), sigma(2))+alpha(3)*normpdf(t, mu(3), sigma(3)))/(sum(alpha));
    else
        sum_ = sum_ + (alpha(1)*normpdf(t, mu(1), sigma(1))+alpha(2)*normpdf(t, mu(2), sigma(2))+alpha(3)*normpdf(t, mu(3), sigma(3)))/(sum(alpha));
    end
end

