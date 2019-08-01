function ims = lbp_image_set0(ims,imgsize)

% Run LBP on an image set and output the resulting mapped images as
% column vectors. The output is used for a distance transform based
% distance metric.

for i=1:size(ims,2)
    im = reshape(ims(:,i),imgsize(1),imgsize(2));
    im = lbp_image(im,'P8R2');
    ims(:,i) = im(:);
end
