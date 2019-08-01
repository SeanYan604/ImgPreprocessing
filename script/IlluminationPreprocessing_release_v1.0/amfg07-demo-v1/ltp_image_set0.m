function [hi,lo] = ltp_image_set0(ims,imgsize,thresh)
   hi = zeros(size(ims));
   lo = hi;
   for i=1:size(ims,2)
      im = reshape(ims(:,i),imgsize(1),imgsize(2));
      im=double(im);
      [h,l] = ltp_image(im,'P8R2',thresh);
      hi(:,i) = h(:);
      lo(:,i) = l(:);
   end
%end
