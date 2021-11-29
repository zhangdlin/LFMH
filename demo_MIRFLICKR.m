
% Please note that the demo applies only to MIRFlickr dataset!!!
% If you use other datasets, please select the corresponding parameters and settings!!!
% If you use other datasets, please select the corresponding parameters and settings!!!
% If you use other datasets, please select the corresponding parameters and settings!!!
% If you use other datasets, please select the corresponding parameters and settings!!!

%  If you have any questions, please contact me at anytime(dlinzzhang@gmail.com).
% If you use our code, please cite our article.


clear all;
clc;
load mirflickr25k;

fprintf('MIR Flickr dataset loaded...\n');
%% parameter settings
run = 1;
map = zeros(run,2);
alphas = [0.5 0.5];
beide= 0.1;
gamma = 1e-5;
lamda=  0.01;
iterum = 20;
bitsset = [32];


%% centralization
fprintf('centralizing data...\n');
I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));


for g1=1:numel(bitsset)
    bits=bitsset(g1);

for i = 1 : run
tic
fprintf('\n\n');
fprintf('run %d starts...\n', i);
I_temp = I_tr';
T_temp = T_tr';
[row, col]= size(I_temp);
[rowt, colt] = size(T_temp);


I_temp = bsxfun(@minus,I_temp , mean(I_temp,2));
T_temp = bsxfun(@minus,T_temp, mean(T_temp,2));
Im_te = (bsxfun(@minus, I_te', mean(I_tr', 2)))';
Te_te = (bsxfun(@minus, T_te', mean(T_tr', 2)))';

L = normalizeFea(L_tr);


%% solve objective function
fprintf('start solving CMFH...\n');
[U1, U2, W1, W2, Z1, Z2 ] = solveLFMH(I_temp, T_temp, L', alphas, beide, lamda, gamma, bits );

    V1 = Z1 * L';
    V2 = Z2 * L';
    
    HX1_tr = sign((bsxfun(@minus, V1, mean(V1,2)))');  
    HX1_te = sign((bsxfun(@minus, W1 * I_te', mean(V1,2)))');
    HX2_tr = sign((bsxfun(@minus, V2, mean(V2,2)))');   
    HX2_te = sign((bsxfun(@minus, W2 * T_te', mean(V2,2)))'); 
  

%% evaluate
fprintf('start evaluating...\n');
      
        B_HX1_tr = bitCompact(HX1_tr >= 0);
        B_HX1_te = bitCompact(HX1_te >= 0);
        B_HX2_tr = bitCompact(HX2_tr >= 0);
        B_HX2_te = bitCompact(HX2_te >= 0);

        sim00 = double(hammingDist(B_HX1_te, B_HX2_tr))';
        simXX = double(hammingDist(B_HX2_te, B_HX1_tr))';
        
        map(i, 1) = perf_metric4Label(L_tr, L_te, sim00); 
        map(i, 2)= perf_metric4Label(L_tr, L_te, simXX);  


fprintf('mAP at run %d runs for ImageQueryOnTextDB: %.4f\n', i,  map(i, 1));
fprintf('mAP at run %d runs for TextQueryOnImageDB: %.4f\n', i,  map(i, 2));
toc
end
mean(map);

    

fprintf('\nbits = %d, beide = %d, gamma = %.4f\n ,lamda = %.4f\n', bits, beide, gamma, lamda);
fprintf('average map over %d runs for ImageQueryOnTextDB: %.4f\n', run,  mean(map( : , 1)));
fprintf('average map over %d runs for TextQueryOnImageDB: %.4f\n', run,  mean(map( : , 2)));

end