function f=f_fcn(I,m,n)
%function to calculate fitness value
      Is=edge(I,'sobel');
      m=nnz(Is);      %sum of pixel intensities
      n_edgels=m;     %number of edge pixels
      H=entropy(I);   %entropy of enhanced image
      f=log(log(m)).*(n_edgels./(m.*n)).*H;
end
