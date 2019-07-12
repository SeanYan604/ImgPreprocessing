function [quantized_angle] = quantizeAngle(angle)
% 将角度制的角度进行量化，0-360度分为8个bin；为00000001-10000000
    if(angle >=0)
        if(angle >= 90)
            if(angle >= 45)
                quantized_angle = 2;
            else
                quantized_angle = 1;
            end
        else
            if(angle >= 135)
                quantized_angle = 4;
            else
                quantized_angle = 8;
            end
        end
    else
        if(angle <= -90)
            if(angle <= -135)
                quantized_angle = 16;
            else
                quantized_angle = 32;
            end
        else
            if(angle <= -45)
                quantized_angle = 64;
            else
                quantized_angle = 128;
            end
        end
    end

end

