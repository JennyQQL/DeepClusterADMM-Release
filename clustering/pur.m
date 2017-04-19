function purity = pur( A, B ) 
%get purity of clustering
%A for cluster label,B for real label
if length( A ) ~= length( B)
    error('length( A ) must == length( B)');
end

%get cluser num
BL = unique(B);
clusternum = length(BL);
totnum = length(A);
pure_train=zeros(clusternum,clusternum);
pcounter_train=zeros(1,clusternum);
accnum = 0;

for i=1:totnum
    pure_train(A(i),B(i))=pure_train((A(i)),B(i))+1;
    pcounter_train(A(i))=pcounter_train(A(i))+1;
end

for i=1:clusternum
    findmax=0;
    for j=1:clusternum
        if findmax<=pure_train(i,j)
           findmax=pure_train(i,j);
        end        
    end
    accnum = accnum +findmax;    
end
purity = accnum/totnum;

