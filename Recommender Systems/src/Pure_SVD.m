% Pure SVD Implementation
clear; clc;
%% 1. 读取数据
base_file = '../Dataset/ml-100k/u1.base';
test_file = '../Dataset/ml-100k/u1.test';

train = readmatrix(base_file,'FileType','text');
test  = readmatrix(test_file,'FileType','text');

user_train = train(:,1);
user_test = test(:,1);

item_train = train(:,2);
item_test = test(:,2);

rating_train = train(:,3);
rating_test = test(:,3);

fprintf('train=%d 条, test=%d 条\n', size(train,1), size(test,1));

%% 2. 构建评分矩阵
n_users = max(max(user_test),max(user_train));
n_items = max([item_train;item_test]);
R = nan(n_users,n_items);

for k = 1:length(user_train)
    R(user_train(k),item_train(k)) = rating_train(k);
end

% fprintf("over");

% lin = sub2ind([n_users,n_items],user_train,item_train);
% R(lin) = rating_train;

%% 3. Pure SVD

ks = [10,20,50,100,200];

pred = nan(length(user_test), 1);      

for k = ks
    Rhat = pure_svd(R,k);

    % Prediction
    for j = 1:length(user_test)
        pred(j) = Rhat(user_test(j), item_test(j));
    end

    % test_lin = sub2ind([n_users, n_items], user_test, item_test);
    % pred = Rhat(test_lin);

    % Estimation
    rmse = sqrt(mean((pred - rating_test).^2));
    mae = mean(abs(pred - rating_test));
    fprintf('Pure SVD  k=%-4d  RMSE = %.4f\n', k, rmse);
    fprintf('Pure SVD  k=%-4d  MAE = %.4f\n',k,mae);

end


%% 4. Pure SVD Function Implementation

function Rhat = pure_svd(R,k)
    mu = mean(R,2,'omitnan');
    Rc = R - mu*ones(1,size(R,2));
    Rc(isnan(Rc)) = 0;

    % [U,S,V] = svd(Rc,'econ');
    [U,S,V] = svds(Rc,k);

    % Rhat = mu + U(:,1:k) * S(1:k,1:k) * V(:,1:k)';
    Rhat = mu + U * S * V';
end