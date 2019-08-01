function img_out = LTV(img_in, lambda, nNeighbors)
    
% face illumination preprocessing using LTV method
% LTV: Logarithm total variation
% Basic principle: Reflectance field estimation
% References: T.Chen, W.Yin, X.S.Zhou, D.Comaniciu, T.S.Huang, Total variation models
%             for variable lighting face recognition, IEEE Transactions on Pattern Analysis
%             and Machine Intelligence 28(2006)1519�C1524.
%
% INPUT:
% img_in: the input image
% lambda: a positive scalar OR a matrix with positive scalars of double type
% nNeighbors:  
%               4 :  anisotropic  4 neighbors for 2D images
%            or 8 : anisotropic 16 neighbors for 2D images
%            or 16: anisotropic 16 neighbors for 2D images
%            or 5 :  isotropic    5 neighbors for 2D binary images, no longer provided
%            or 6 :  anisotropic  6 neighbors for 3D images
%
% OUTPUT:
% img_out: the LTV image

% record current directory
root=pwd;
% subdirectory
ver='TVL1_v2.32';

f_8bit = log( double(img_in + 1) );
f_8bit = uint8( 255 * mat2gray(f_8bit) );

if lambda == 0
    img_out = f_8bit;
    u = double(f_8bit) - double(img_out);
    u = uint8( mat2gray(u)*255 );
else
    cd(ver);
    u = Graph_anisoTV_L1_v2_consistent_weights(f_8bit, lambda, nNeighbors, 2);
%     u = Graph_anisoTV_L2_v2_consistent_weights(f_8bit, lambda, nNeighbors);
    cd(root);
    
    v = mat2gray(double(f_8bit) - double(u)) * 255.0;
    img_out = uint8( v );
    
end
end