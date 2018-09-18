function acc = absSpecSim(A, threshold)

if length(threshold)>1
    
    Do = diag(sum(A));
    D = diag(sum(threshold));
    B = (Do-A)-(D-threshold);
    [sV,sD]=eigs(B,1);
    [sVn,sDn] = eigs(Do-A,1);
    acc = real(1-sD(1)/sDn(1));
else
    acc = 0;
end
end
