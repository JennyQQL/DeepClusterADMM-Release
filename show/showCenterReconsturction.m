function showCenterReconsturction(centro,w5,w6,w7,w8)
    XXout = reconstructionFromFeature(centro,w5,w6,w7,w8);
    hs = sqrt(size(XXout,2));
    XXout = reshape(XXout,size(centro,1),hs,hs);
    out = zeros(hs,size(XXout,1));
    for i=1:size(XXout,1)
        out(:,(i-1)*hs+1:i*hs) = squeeze(XXout(i,:,:));
    end
   figure; imshow(out);
