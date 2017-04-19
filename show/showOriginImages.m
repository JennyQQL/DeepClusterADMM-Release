function showOriginImages(image)
   isize = sqrt(size(image,2));
    XXout = reshape(image,size(image,1),isize,isize);
    out = zeros(isize,size(XXout,1));
    for i=1:size(XXout,1)
        out(:,(i-1)*isize+1:i*isize) = squeeze(XXout(i,:,:));
    end
   figure; imshow(out);
