function img_out = RETINEX(img_in, type_img, type_retinex, dGamma, nIterations)

% face illumination preprocessing using SSR method
% SSR: Single scale retinex
% Basic principle: Gradient or edge extraction
% References: D.J.Jobson, Z.Rahman, G.A.Woodell, Properties and performance of acenter/surround 
%             retinex, IEEE Transactionson Image Processing 6(1997)451¨C462.
%
% INPUT:
% img_in: the input image
% type_img: the type name of the input image. 'GRAY8' or 'RGB24'
% type_retinex: which type of retinex, 2 is recommended
% dGamma: gamma value for 'postlut' postprocessing
% nIterations: number of iterations
%
% OUTPUT:
% img_out: the SSR image
%

img_out = zeros(size(img_in), 'uint8');
switch type_img
    case 'GRAY8'
        nTotalChannel = 1;
    case 'RGB24'
        nTotalChannel = 3;
end

for inx_Channel = 1 : nTotalChannel
    chanlCur = img_in(:, :, inx_Channel);
    
    % logarithm transform log10()
    % log_chanlCur = log10( double(chanlCur) + 1 );
    log_chanlCur = log( double(chanlCur) + 1 );
    
    % normlized channel to [0, 1]
    %     minVal = min( min(log_chanlCur) );
    %     maxVal = max( max(log_chanlCur) );
    %     norm_log_chanlCur = (log_chanlCur - minVal) / (maxVal - minVal);
    norm_log_chanlCur = double(mat2gray(log_chanlCur));
    
    switch type_retinex
        % mccann99 retinex, slow
        case 1
            out = retinex_mccann99(norm_log_chanlCur, nIterations);
            % frankle_mccann retinex, fast
        case 2
            out = retinex_frankle_mccann(norm_log_chanlCur, nIterations);
            % error type
        otherwise
            fprintf('Wrong method type!\n');
    end
    
    % inverse log
    out = exp(out);
    % normalized to [0, 255]
    % out = out * 255;    
    %     minVal = min(min(out));
    %     maxVal = max(max(out));
    %     out = (out - minVal) / (maxVal - minVal) * 255;
    out = double(mat2gray(out)) * 255.0;
    out = uint8(out);
    
    % postlut of the output image
    % out = imadjust(out, [], [], dGamma);
    
    % return the result image
    img_out(:, :, inx_Channel) = out;
end