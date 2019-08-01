function u=Graph_anisoTV_L2_v2_consistent_weights(F,lambda,nNeighbors)
% Input arguments:
% F=input 8/16-bit image: 2D/3D matrix of type uint8 or int16
% lambda=positive scalar or matrix
% nNeighbors: choice of neighborhood (2D: 4 or 16, 3D: 6)

%% test input
if ~(exist('F','var') || exist('lambda','var') || exist('nNeighbors','var'))
    warndlg('No input is specified. Running a demo.');
    F=logical([0 1 1; 1 0 0]);
    lambda = 1.5;
    nNeighbors = 4;
end

%% Graph neighborhood topology
    %% type information
    % each row correspondes to an arc
    % 1st element: 0 = incoming; 1 = outgoing;
    %              2 = related arcs;
    %              3 = incoming s arc;
    %              4 = outgoing t arc;
    %              5 = related incoming s arc;
    %              6 = ralated outgoing t arc;
    %              Now: only 1, 3, 4 are allowed
    % 2nd element: outgoing arc capacity (except for s-arc, that's the incoming capacity)
    % (3rd,4th) elements: (row col) offset to the partner node if not terminal arc
ndim = ndims(F);
if (any(size(F)==1)); error('There is a trivial dimension (size=1) in F. Please correct.'); end

if (nNeighbors==4 && ndim==2)
    w4=pi/4;
    % 2D: 4-point neighbors
    type = [
        1   w4         0   1 ;
        1   w4         1   0 ;
        1   w4         0  -1 ;
        1   w4        -1   0 ]';

elseif (nNeighbors==8 && ndim==2)
    w4=pi/8; w8=sqrt(2)*pi/16;
    % 2D: 8-point neighbors
    type = [
        1   w4         0   1 ;      
        1   w8         1   1 ;
        1   w4         1   0 ;
        1   w8         1  -1 ;
        1   w4         0  -1 ;
        1   w8        -1  -1 ;
        1   w4        -1   0 ;
        1   w8        -1   1 ]';

elseif (nNeighbors==16 && ndim==2)
    w4=atan(0.5)/2; w8=(pi/4-atan(0.5))/(2*sqrt(2)); w16=(atan(0.5)+(pi/4-atan(0.5)))/2/(2*sqrt(5));
    % 2D: 16-point neighbor
    type = [
        1   w4          0   1;
        1   w4          1   0;
        1   w4          0  -1;
        1   w4         -1   0;
        1   w8          1   1;
        1   w8          1  -1;
        1   w8         -1  -1;
        1   w8         -1   1;
        1   w16         1   2;
        1   w16         2   1;
        1   w16         2  -1;
        1   w16         1  -2;
        1   w16        -1  -2;
        1   w16        -2  -1;
        1   w16        -2   1;
        1   w16        -1   2]';
    
elseif (nNeighbors==6 && ndim==3)
    w6 = (pi/2)^2/pi;
    % 3D: 6-point neighbor
    type = [
        1   w6           0   0   1;
        1   w6           0   1   0;
        1   w6           0   0  -1;
        1   w6           0  -1   0;
        1   w6           1   0   0;
        1   w6          -1   0   0]';
else
    error('Unsupported dimension or neighbor type.');
end
        
    
%% call parametric max-flow
if (isa(F, 'logical'))
    if (isscalar(lambda))
        u = dnc_TV_L2_1bit(F,lambda,type);
    else
        u = dnc_TV_L2_1bit_A(F,lambda,type);
    end
elseif (isa(F, 'uint8'))
    if (isscalar(lambda))
        u = dnc_TV_L2_8bit(F,lambda,type);
    else
        u = dnc_TV_L2_8bit_A(F,lambda,type);
    end
elseif (isa(F, 'uint16'))
    if (isscalar(lambda))
        u = dnc_TV_L2_16bit(F,lambda,type);
    else
        u = dnc_TV_L2_16bit_A(F,lambda,type);
    end
else
    error('This program only supports inputs of type uint8 or uint16.');
end

%% Show final energy for verification
%energy_l2TVL1(F,u,lambda);
