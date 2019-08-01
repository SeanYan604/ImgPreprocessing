% (Non-symmetric) 'distance' between LBP images based on distance
% transform. The columns of A,B are LBP images and the output is the
% matrix of distances between them. For each image pair, for each LBP
% code, the distance transform of the image from A is evaluated at the
% positions containing the corresponding code in the image from B, and
% the result is summed over the image and the codes.

function D = dist_lbp_disttrans(A,B,imgsize,map,alpha,thresh)
   [d,m]=size(A);
   [d1,n]=size(B);
   if (d ~= d1)
      error('column length of A (%d) != column length of B (%d)\n',d,d1);
   end
   % Set up map for image values. The output values are [1:dmap].
   if isempty(map)
      map = 256;
   end
   if size(map,1)>1
      dmap = size(map);
   elseif map>0
      dmap = map;
      map = [1:dmap]';
   else % use uniform patterns if map==0, add "other" bin if map<0
      k = (map<0);
      unipats = uniform_pattern(8)';
      dmap = size(unipats,1);
      map = (dmap+1)*ones(2^8,1);
      map(1+unipats) = [1:dmap]';
      dmap = dmap+k;
   end
   D = zeros(m,n);
   
   start=clock;
   for i=1:m
      % show some process information

      elap_t = etime(clock,start);
      tot_t  = (elap_t/i)*m;
      fprintf(1,'\n processing image: %3d--time used: %3.0f s /total: %3.0f s ',i,elap_t,tot_t);
     
      a = reshape(A(:,i),imgsize(1),imgsize(2));
      a = map(1+a);
      % build set of distance transform maps, one for each
      % mapped LBP code.
      %fprintf('\n  *** starting to build distance transform maps....');
      dta = zeros(d,dmap);
      if alpha>0  % classical distance, trim large distances
          for k=1:dmap
              t = bwdist((a==k),'euclidean');
              dta(:,k) = min(t(:),thresh).^alpha;
          end
      elseif alpha<0 % decaying similarity measure, limit small distances
          for k=1:dmap
              t = bwdist((a==k),'euclidean');
              dta(:,k) = (t(:)+thresh).^alpha;
          end
      else  % special case - gaussian of width thresh
          for k=1:dmap
              t = bwdist((a==k),'euclidean');
              dta(:,k) = exp(-0.5*(t(:)/thresh).^2);
          end
      end
      
      %fprintf('\n  *** starting to compute pairwise distance ....');
      % this simple loop is the fastest way
      for j=1:n
          b = map(1+B(:,j));
          dist = 0;
          for k=1:dmap
              dist = dist + sum(dta(find(b==k),k));
              % dist = dist + sum(dta(:,k).*(b==k)); % this is slower
          end
          D(i,j) = dist;
      end
   end
%end
