load('LFR.mat');
T = 0:0.05:1;
output = zeros(length(T),1);

NoneZ = A(:);
zeroC = find(NoneZ==0);
NoneZ(zeroC) = [];

for j=1:length(T)
	threshold = A; 
	percentile = prctile(abs(NoneZ(:)),T(j)*100);
	threshold(A < percentile) = 0;
	output(j) = absSpecSim(A, threshold);
end
    
figure;
plot(T,output(:),'Marker','^');
    
