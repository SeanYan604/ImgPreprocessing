function [hist,ims] = ltp_image_set2(ims,imgsize,blocksize,thresh)
   unipats = uniform_pattern(8)';
   np = size(unipats,1)+1;
   map = np*ones(2^8,1);
   map(1+unipats) = [1:np-1]';
   n = size(ims,2);
   %hist = zeros(2*np*prod(imgsize)/prod(blocksize),n);
   hist = zeros(2*np*ceil(imgsize(1)/blocksize(1))*ceil(imgsize(2)/blocksize(2)),n);
   for i=1:n
      im = reshape(ims(:,i),imgsize(1),imgsize(2));
      if nargout>2
	 [h,l,im] = ltp_image(im,'P8R2',thresh);
      else
	 [h,l] = ltp_image(im,'P8R2',thresh);
      end
      h = map(1+h);
      l = map(1+l);
      hh = lbp_histo([1:np]',im2col(h,blocksize,'distinct'));
      ll = lbp_histo([1:np]',im2col(l,blocksize,'distinct'));
      hist(:,i) = [hh(:); ll(:)];
      if nargout>2
	 ims(:,i) = im(:);
      end
   end
%end
