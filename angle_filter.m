function [contour,strong_angle] = angle_filter(mask,quantized_angle)
    temp_angle = mask.*quantized_angle;
    
    kernal_1 = 9;
    kernal_2 = 9;
    [m,n] = size(mask);
    strong_angle = zeros(m,n);
    contour = zeros(m,n);
    score_map = [5,3,1,0,0,0,1,3;
                 3,5,3,1,0,0,0,1;
                 1,3,5,3,1,0,0,0;
                 0,1,3,5,3,1,0,0;
                 0,0,1,3,5,3,1,0;
                 0,0,0,1,3,5,3,1;
                 1,0,0,0,1,3,5,3;
                 3,1,0,0,0,1,3,5];
    bias_1 = floor(kernal_1/2);
    bias_2 = floor(kernal_2/2);
    qt_angle = [1,2,4,8,16,32,64,128];
    
    
    for i = 1:m
        for j = 1:n
            if(mask(i,j))
                if(i-bias_1<1) h_t=1; else h_t=i-bias_1;end
                if(i+bias_1>m) h_b=m; else h_b=i+bias_1;end
                if(j-bias_1<1) w_l=1; else w_l=j-bias_1;end
                if(j+bias_1>m) w_r=m; else w_r=j+bias_1;end
                temp = temp_angle(h_t:h_b,w_l:w_r);
                temp = temp(:);
                [m_,n_] = size(temp);
                strong_temp = zeros(m_,n_);
                score_temp = zeros(m_,n_);
                [max_count,index] = max(histc(temp,qt_angle));
                strong_angle(i,j) = qt_angle(index);
                
                count = 0;
                for c = 1:m_*n_
                    if(temp(c))
                        score_temp(c)=score_map(log2(temp_angle(i,j))+1,log2(temp(c))+1);
                        strong_temp(c)=score_map(log2(strong_angle(i,j))+1,log2(temp(c))+1);
                        count = count + 1;
                    end
                end
                pix_score = sum(score_temp)/count;
                strong_score = sum(strong_temp)/count;
                if(max_count > 5 && (pix_score > 2 || strong_score > 2))
                    contour(i,j) = 1;
                end
            end
        end
    end
    contour = medfilt2(contour);
%     figure(1);
%     imshow(100+uint8(temp_angle));
%     figure(2);
%     subplot(1,2,1);
%     imshow(mask);
%     subplot(1,2,2);
%     imshow(255*uint8(contour));
end

