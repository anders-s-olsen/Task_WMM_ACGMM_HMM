n=5;

A = rand(n,n); % generate a random n x n matrix

% construct a symmetric matrix using either
A = 0.5*(A+A'); 

% since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
%   is symmetric positive definite, which can be ensured by adding nI
A = A + n*eye(n);

L = chol(A,'lower')

inv(eye(n)+A*A')

eye(n)-A*inv(eye(5)+A'*A)*A'


%%

n=100;

A = rand(n,n); % generate a random n x n matrix

% construct a symmetric matrix using either
A = 0.5*(A+A'); 

% since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
%   is symmetric positive definite, which can be ensured by adding nI
A = A + n*eye(n);


X = randn(100,10);
X=normc(X);
A_inv = inv(A);

for i = 1:10
    out(i) = X(:,i)'*A_inv*X(:,i);
end
diag(X'*A_inv*X)
sum(X'*A_inv,2)

sum((X'*A_inv).*(X'*A_inv),2)

%%
k=2;
n = 100;
X = randn(n,10);
X = normc(X);
L = randn(n,k);
A_inv = L*L'+eye(n);

for i = 1:10
    out(i) = X(:,i)'*(eye(n)-L*inv(eye(k)+L'*L)*L')*X(:,i);
    out2(i) = 1-X(:,i)'*L*inv(eye(k)+L'*L)*L'*X(:,i);
end

1-diag(X'*L*inv(eye(k)+L'*L)*L'*X);

1-sum(inv(eye(k)+L'*L)*(X'*L)'.*(X'*L)',1)
1-sum(inv(eye(k)+L'*L)*(X'*L)'.*(X'*L)',1)

B = L'*X;
1-sum(inv(eye(k)+L'*L)*B.*B)

X2 = X';
B = X2*L;
1-sum(inv(eye(k)+L'*L)*B'.*B',1)
1-sum(B*inv(eye(k)+L'*L).*B,2)

1-sum(((X'*inv(eye(k)+L'*L)*L).^2)')

sum((X'*L).^2,2)+1


sig2 = eye(3)+0.99*(ones(3)-eye(3)); %noise is one minus the off diagonal element, log space
sig3 = diag([1e-02,1,1])+0.9*[0,0,0;0,0,1;0,1,0]; %noise is the first diagonal element, log space






















