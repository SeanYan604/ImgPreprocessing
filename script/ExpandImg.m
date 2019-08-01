function [img_expanded, map] = ExpandImg(input_src)
%Expanding ROI to a 256*256 square image
% map is the index of vect of each column.
    [m,n] = size(input_src);
    img_expanded = zeros(m,n);
    map = zeros(m,n);
    for i = 1:n
        vect = zeros(256,1);
        vect_pos = zeros(256,1);
        count = 0;
        for j = 1:m
            if(input_src(j,i) > 0)
                count = count + 1;
                vect(count) = input_src(j,i);
                vect_pos(count) = j;
            end
        end
        if(count > 0)
            [vect, sort_index] = sort(vect(1:count));
            pos = vect_pos(sort_index);
            pad = floor(256/count);
            index = 0;
            for k = 1:count-1
                index=index+1;
                img_expanded(index,i) = vect(k);
                map(index,i) = pos(k);
                if(vect(k+1) > vect(k))
                    step = floor((vect(k+1)-vect(k))/pad);
                else
                    step = ceil((vect(k+1)-vect(k))/pad);
                end
                temp = vect(k);
                for l = 1:pad-1
                    index = index+1;
                    temp = temp + step;
                    img_expanded(index,i) = temp;
                end
            end
            index = index + 1;
            img_expanded(index:256,i) = vect(count);
            map(index,i) = pos(count);
%             if(add > 0)
%                 img_expanded(index+1:256,i) = vect(count);
%             end
        end
    end
    
%     figure;
%     imshow(uint8(img_expanded));
end

