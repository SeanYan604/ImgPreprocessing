function [numY_pat, numX_pat, tabY_pat, tabX_pat, tabRow_pat, tabCol_pat] = OverlappedBlock(row_img, col_img, size_pat, step_pat)

% calculate information of every overlapped block
% INPUT:
% row_img: the height of image
% col_img: the width of image 
% size_pat: the height(width) of the patch
% step_pat: patch step
%
% OUTPUT:
% numY_pat: number of patches in Y direction;
% numX_pat: number of patches in X direction;
% tabY_pat: left most coordinate of a patch;
% tabX_pat: top most coordinate of a patch;
% tabRow_pat: number of rows in a patch, height;
% tabCol_pat: number of columns in a patch, width;

row_pat = size_pat;
col_pat = size_pat;

numY_pat = floor( (row_img - row_pat) / step_pat );

% the remaining length in Y direction
modY_pat = row_img - row_pat - numY_pat * step_pat;
if modY_pat < 0 || modY_pat >= step_pat
    fprintf('Error in caculating modY_pat!\n');
end

numX_pat = floor( (col_img - col_pat) / step_pat );

% the remaining length in X direction
modX_pat = col_img - col_pat - numX_pat * step_pat;
if modX_pat < 0 || modX_pat >= step_pat
    fprintf('Error in caculating modX_pat!\n');
end

% case 1: be divided with no remainder in Y direction, be divided with no
% remainder in X direction
if (modY_pat == 0 ) && (modX_pat == 0)
    numY_pat = numY_pat + 1;
    numX_pat = numX_pat + 1;
    numTotal_pat = numY_pat * numX_pat;
    
    % the pathces is numbered from 1 to numTotal_pat
    tabY_pat   = zeros(numTotal_pat, 1);
    tabX_pat   = zeros(numTotal_pat, 1);
    tabRow_pat = zeros(numTotal_pat, 1);
    tabCol_pat = zeros(numTotal_pat, 1);
    
    inx = 0;
    for i = 1 : numY_pat
        for j = 1 : numX_pat
            inx = inx + 1;
            tabY_pat(inx) = 1 + (i - 1) * step_pat;
            tabX_pat(inx) = 1 + (j - 1) * step_pat;
            tabRow_pat(inx) = row_pat;
            tabCol_pat(inx) = col_pat;
        end % end j
    end % end i
end
% end case 1

% case 2: be divided with remainder in Y direction, be divided with 
% remainder in X direction
if (modY_pat ~= 0 ) && (modX_pat ~= 0)
    numY_pat = numY_pat + 1 + 1;
    numX_pat = numX_pat + 1 + 1;
    numTotal_pat = numY_pat * numX_pat;

    % the pathces is numbered from 1 to numTotal_pat
    tabY_pat   = zeros(numTotal_pat, 1);
    tabX_pat   = zeros(numTotal_pat, 1);
    tabRow_pat = zeros(numTotal_pat, 1);
    tabCol_pat = zeros(numTotal_pat, 1);
    
    inx = 0;
    for i = 1 : numY_pat
        for j = 1 : numX_pat
            inx = inx + 1;
            tabY_pat(inx) = 1 + (i - 1) * step_pat;
            tabX_pat(inx) = 1 + (j - 1) * step_pat;
            if i <= (numY_pat - 1)
                tabRow_pat(inx) = row_pat;
            else
                % the right most block in a row
                tabRow_pat(inx) = row_pat - (step_pat - modY_pat);
            end

            if j <= (numX_pat - 1)
                tabCol_pat(inx) = col_pat;
            else
                % the bottom most block in a col
                tabCol_pat(inx) = col_pat - (step_pat - modX_pat);
            end

        end % end j
    end % end i
end
% end case 2

% case 3: be divided with no remainder in Y direction, be divided with
% remainder in X direction
if (modY_pat == 0 ) && (modX_pat ~= 0)
    numY_pat = numY_pat + 1;
    numX_pat = numX_pat + 1 + 1;
    numTotal_pat = numY_pat * numX_pat;

    % the pathces is numbered from 1 to numTotal_pat
    tabY_pat   = zeros(numTotal_pat, 1);
    tabX_pat   = zeros(numTotal_pat, 1);
    tabRow_pat = zeros(numTotal_pat, 1);
    tabCol_pat = zeros(numTotal_pat, 1);
    
    inx = 0;
    for i = 1 : numY_pat
        for j = 1 : numX_pat
            inx = inx + 1;
            tabY_pat(inx) = 1 + (i - 1) * step_pat;
            tabX_pat(inx) = 1 + (j - 1) * step_pat;
            tabRow_pat(inx) = row_pat;
            if j <= (numX_pat - 1)
                tabCol_pat(inx) = col_pat;
            else
                % the bottom most block in a col
                tabCol_pat(inx) = col_pat - (step_pat - modX_pat);
            end
        end % end j
    end % end i
end
% end case 3

% case 4: be divided with remainder in Y direction, be divided with no
% remainder in X direction
if (modY_pat ~= 0 ) && (modX_pat == 0)
    numY_pat = numY_pat + 1 + 1;
    numX_pat = numX_pat + 1;
    numTotal_pat = numY_pat * numX_pat;

    % the pathces is numbered from 1 to numTotal_pat
    tabY_pat   = zeros(numTotal_pat, 1);
    tabX_pat   = zeros(numTotal_pat, 1);
    tabRow_pat = zeros(numTotal_pat, 1);
    tabCol_pat = zeros(numTotal_pat, 1);
    
    inx = 0;
    for i = 1 : numY_pat
        for j = 1 : numX_pat
            inx = inx + 1;
            tabY_pat(inx) = 1 + (i - 1) * step_pat;
            tabX_pat(inx) = 1 + (j - 1) * step_pat;
            tabCol_pat(inx) = col_pat;
            if i <= (numY_pat - 1)
                tabRow_pat(inx) = row_pat;
            else
                % the right most block in a row
                tabRow_pat(inx) = row_pat - (step_pat - modY_pat);
            end
        end % end j
    end % end i
end
% end case 4
end