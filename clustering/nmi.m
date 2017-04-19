function MIhat = nmi( A, B )
% NMI Normalized mutual information
% http://en.wikipedia.org/wiki/Mutual_information
% http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
% Author: http://www.cnblogs.com/ziqiao/   [2011/12/13] 
if length( A ) ~= length( B)
    error('length( A ) must == length( B)');
end
total = length(A);
A_ids = unique(A);
B_ids = unique(B);

% Mutual information
MI = 0;
for idA = A_ids
    for idB = B_ids
         idAOccur = find( A == idA );
         idBOccur = find( B == idB );
         idABOccur = intersect(idAOccur,idBOccur); 
         
         px = length(idAOccur)/total;
         py = length(idBOccur)/total;
         pxy = length(idABOccur)/total;
         
         MI = MI + pxy*log2(pxy/(px*py)+eps); % eps : the smallest positive number

    end
end

% Normalized Mutual information
Hx = 0; % Entropies
for idA = A_ids
    idAOccurCount = length( find( A == idA ) );
    Hx = Hx - (idAOccurCount/total) * log2(idAOccurCount/total + eps);
end
Hy = 0; % Entropies
for idB = B_ids
    idBOccurCount = length( find( B == idB ) );
    Hy = Hy - (idBOccurCount/total) * log2(idBOccurCount/total + eps);
end

MIhat = 2 * MI / (Hx+Hy);
end
