function match = LabelMatch(true_labels, cluster_labels)
    % this function provide label match utils
    % it returns a matrix that each row corresponds to the true labels
    % each column that equal to 1 means the correct match
    n = length(true_labels);
    cat = spconvert([(1:n)' true_labels ones(n,1)]);
    cls = spconvert([(1:n)' cluster_labels ones(n,1)]);
    cls = cls';
    cmat = full(cls * cat);
    match = hungarian(-cmat);
end