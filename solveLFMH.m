function [ U1, U2, P1, P2, Z1, Z2 ] = solveLFMH( X1, X2, L, alphas, beide, gamma, lamda,  bits )
%

%% random initialization
[d1, ~] = size(X1);
[d2, ~] = size(X2);
[lab, ~] = size(L);
U1 = rand(d1,bits);
U2 = rand(d2,bits);
Z1 = rand(bits,lab);
Z2 = rand(bits,lab);
P1 = eye(bits, d1);
P2 = eye(bits, d2);

threshold = 0.01;
lastF = 99999999;
iter = 1;

%% compute iteratively
% while (true)
for iterum = 1:50
	% update U1 and U2
    U1=(X1*L'*Z1')/(Z1*L*L'*Z1'+lamda/alphas(1)*eye(bits));
    U2=(X2*L'*Z2')/(Z2*L*L'*Z2'+lamda/alphas(2)*eye(bits));
    
   Z1 = (alphas(1)*U1'*U1 + (beide+gamma+lamda)*eye(bits))\(alphas(1)*U1'*X1*L'+beide*Z2*L*L'+gamma*P1*X1*L')/(L*L');
    Z2 = (alphas(2)*U2'*U2 + (beide+gamma+lamda)*eye(bits))\(alphas(2)*U2'*X2*L'+beide*Z1*L*L'+gamma*P2*X2*L')/(L*L');
    
    P1= (Z1*L * X1')/(X1*X1'+lamda/gamma*eye(d1));
    P2= (Z2*L * X2')/(X2*X2'+lamda/gamma*eye(d2));
  
    
    % compute objective function
    o1 = alphas(1) * norm(X1 - U1 * Z1*L, 'fro') ^ 2;
    o2 = alphas(2) * norm(X2 - U2 * Z2*L, 'fro') ^ 2;
    o3 = beide * norm(Z1*L - Z2*L, 'fro')^2;
    o4 = gamma * (norm(Z1*L - P1 * X1, 'fro') ^ 2 + norm(Z2*L - P2 * X2, 'fro') ^ 2 );
    o5 = lamda * ( (norm(U1, 'fro') ^ 2 + norm(P1, 'fro') ^ 2 +  norm(U2, 'fro') ^ 2 + norm(P2, 'fro') ^ 2 + norm(Z1*L, 'fro') ^ 2+norm(Z2*L, 'fro') ^ 2));
    over = o1 + o2 + o3 + o4 + o5 ;
    % 打印第iter次迭代的各项损失函数值
    fprintf('\nobj at iiteration %d\n', iter);    

    
    if (lastF - over) < threshold
        return;
    end
    iter = iter + 1;
    lastF = over;
end
return;
end

