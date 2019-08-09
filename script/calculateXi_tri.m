function [Xi] = calculateXi_tri(mu,sigma,Q)
% Dividing the normal distributuion into Q sections, calculate each
% saction's border as Xi.
Xi = zeros(1,Q);
sum = 0; i = 1;
for t = 1:255
    if(sum >= 3*(i/(Q+1)))
        Xi(i) = t;
        i = i+1;
        while(sum >= 3*(i/(Q+1)))
            Xi(i) = t;
            i = i+1;
        end
        if(i == Q+1)
            break;
        end
        sum = sum + normpdf(t, mu(1), sigma(1))+normpdf(t, mu(2), sigma(2))+normpdf(t, mu(3), sigma(3));
    else
        sum = sum + normpdf(t, mu(1), sigma(1))+normpdf(t, mu(2), sigma(2))+normpdf(t, mu(3), sigma(3));
    end
end

