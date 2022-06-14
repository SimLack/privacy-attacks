function [U S V M A] = embed_static(A, K, beta)
% Input: 
% A: N*N adjacency matrix (sparse)
% K: dimensionality of embedding space
% beta: decaying constant, default is 0.8 / spectral radius
% Output:
% U, S, V: the GSVD result of the high-order proximity (katz) matrix
% The high-order proximity (katz) matrix is approximated by U * S * V'
myString = A;
stringA = strcat(A,"_np_adj_");
A = csvread(stringA,1,1);
A = sparse(A);

[N, ~] = size(A);
% Katz: S = sum_{l=1}^{+inf}{beta*A}^l
%if nargin < 3
beta = 0.8 / getRadius(A);
%end
beta = 0.0005;
A = beta .* A; %M_l
M = speye(N)-A; %M_g

[V, S, U] = jdgsvds(A', M', K, 0.0001); % the 0.0001 error tolerance can be modified to speed up while reducing accuracy
U = U(:,1:K); 
S = S(1:K,1:K);
V = V(:,1:K);
%U = U(:,1:K) * sqrt(S(1:K,1:K));
%V = V(:,1:K) * sqrt(S(1:K,1:K));

% test with U=V and V=U
writematrix(U,strcat(myString,"_U_.csv"));
writematrix(S,strcat(myString,"_Sigma_.csv"));
writematrix(V,strcat(myString,"_V_.csv"));
writematrix(M,strcat(myString,"_Ma_.csv"));
writematrix(A,strcat(myString,"_Mb_.csv"));
save(strcat(myString,"_beta_.mat"),'beta');

end

