% Run LBP on an image set and output the rsulting histograms. This
% routine histograms the uniform patterns and includes one
% additional bin for all non-uniform ones.

function [hist,ims] = lbp_image_set2(ims,imgsize,blocksize)

unipats = uniform_pattern(8)';
np = size(unipats,1)+1;
map = np*ones(2^8,1);
map(1+unipats) = [1:np-1]';
n = size(ims,2);
%hist = zeros(np*prod(imgsize)/prod(blocksize),n);
hist = zeros(np*ceil(imgsize(1)/blocksize(1))*ceil(imgsize(2)/blocksize(2)),n);

for i=1:n
    if 0==mod(i,1000) fprintf('\n     current processed %d  images',i);end
    im = reshape(ims(:,i),imgsize(1),imgsize(2));  
    im = lbp_image(im,'P8R2');
    im = map(1+im);
    h = lbp_histo([1:np]',im2col(im,blocksize,'distinct'));
    hist(:,i) = h(:);
    if nargout>1             
        ims(:,i) = im(:);
    end
end

    