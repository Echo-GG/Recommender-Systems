%% RSVD (Regularized SVD)

clear; clc; rng(0);

%% 1. 读取数据
base_file = '../Dataset/ml-100k/ua.base';
test_file = '../Dataset/ml-100k/ua.test';

train = readmatrix(base_file,'FileType','text');  
test  = readmatrix(test_file,'FileType','text');

user_train = train(:,1); item_train = train(:,2); rate_train = train(:,3);
user_test = test(:,1);  item_test = test(:,2);  rate_test = test(:,3);

n_users = max([user_train; user_test]);
n_items = max([item_train; item_test]);
fprintf('train = %d 条, test = %d 条\n', size(train,1), size(test,1));

%% 2. 超参数 
d       = 20;      % 潜因子维度
T       = 100;     % SGD 轮数 (epoch)
gama    = 0.01;    % 学习率
alpha_u = 0.01;    % 用户因子 L2 正则系数
alpha_v = 0.01;    % 物品因子 L2 正则系数
beta_u  = 0.01;    % 用户偏置 b_u 的正则系数
beta_v  = 0.01;    % 物品偏置 b_i 的正则系数

%% 3. 训练
[U, V, bu, bi, mu] = SGD_RSVD(user_train, item_train, rate_train, n_users, n_items, d, T, gama, alpha_u, alpha_v, beta_u, beta_v);

%% 4. 预测与评估 (r_hat = mu + b_u + b_i + U_u·V_i^T)
pred_test = mu + bu(user_test) + bi(item_test) + sum(U(user_test,:) .* V(item_test,:), 2);
pred_test = min(max(pred_test, 1), 5);     
rmse = sqrt(mean((pred_test - rate_test).^2));
mae  = mean(abs(pred_test - rate_test));
fprintf('RSVD(SGD)  d=%-3d  T=%-3d  RMSE = %.4f   MAE = %.4f\n', d, T, rmse, mae);

%% ==================== RSVD-SGD ====================
function [U, V, bu, bi, mu] = SGD_RSVD(user, item, rate, n_users, n_items, d, T, gama, alpha_u, alpha_v, beta_u, beta_v)

% RSVD 随机梯度下降训练
%   输入: user/item/rate —— 训练三元组 (1 起始下标), 共 n 条
%         n_users, n_items, d, T, 学习率 gama, 各正则系数
%   输出: U(n×d), V(m×d), bu(n×1), bi(m×1), mu(标量全局均值)
%
% 单条评分 (u,i,r) 的残差:  e = r - (mu + b_u + b_i + U_u·V_i^T)
% 沿负梯度更新 (γ 已吸收梯度中的因子 2):
%   b_u ← b_u + γ (e - β_u b_u)
%   b_i ← b_i + γ (e - β_v b_i)
%   U_u ← U_u + γ (e·V_i - α_u U_u)
%   V_i ← V_i + γ (e·U_u - α_v V_i)
%   μ   ← μ + γ e

    n = numel(rate); % number of elements

    % ---- 初始化 ----
    mu = mean(rate);                 % 全局均值
    bu = zeros(n_users, 1);          % 用户偏置
    bi = zeros(n_items, 1);          % 物品偏置

    U = 0.01 * (rand(n_users,d) - 0.5); % 用户因子
    V = 0.01 * (rand(n_items,d) - 0.5); % 物品因子
    
    for t = 1:T
        order = randperm(n);        
        for k = 1:n
            idx = order(k);
            u = user(idx);  i = item(idx);  r = rate(idx);

            % 残差
            e = r - (mu + bu(u) + bi(i) + U(u,:) * V(i,:)');

            % SGD 更新
            bu(u)  = bu(u)  + gama * (e - beta_u  * bu(u));
            bi(i)  = bi(i)  + gama * (e - beta_v  * bi(i));
            U(u,:) = U(u,:) + gama * (e * V(i,:) - alpha_u * U(u,:));
            V(i,:) = V(i,:) + gama * (e * U(u,:) - alpha_v * V(i,:));
            mu = mu + gama * e;
        end
        gama = gama * 0.9;

    end
end