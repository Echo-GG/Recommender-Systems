%% SVD++

clear; clc; rng(0);

%% 1. 读取数据

base_file = '../Dataset/ml-100k/ub.base';
test_file = '../Dataset/ml-100k/ub.test';

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
alpha_y = 0.01;    % 隐式反馈因子 y_j 的 L2 正则系数
beta_u  = 0.01;    % 用户偏置 b_u 的正则系数
beta_v  = 0.01;    % 物品偏置 b_i 的正则系数

%% 3. 训练

[U, V, Y, bu, bi, mu, rated, inv_sqrt_nu] = SGD_SVDpp( ...
    user_train, item_train, rate_train, n_users, n_items, d, T, ...
    gama, alpha_u, alpha_v, alpha_y, beta_u, beta_v);

%% 4. 预测与评估 (r_hat = mu + b_u + b_i + V_i·(U_u + |N(u)|^{-1/2} Σ y_j))

n_test = numel(rate_test);
pred_test = zeros(n_test,1);

for k = 1:n_test
    u = user_test(k);  i = item_test(k);
    nu = rated{u};
    if isempty(nu)
        su = zeros(1,d);                
    else
        su = sum(Y(nu,:),1) * inv_sqrt_nu(u);   % 1×d
    end
    pred_test(k) = mu + bu(u) + bi(i) + (U(u,:) + su) * V(i,:)';
end

pred_test = min(max(pred_test, 1), 5);
rmse = sqrt(mean((pred_test - rate_test).^2));
mae  = mean(abs(pred_test - rate_test));
fprintf('SVD++(SGD)  d=%-3d  T=%-3d  RMSE = %.4f   MAE = %.4f\n', d, T, rmse, mae);

%% ==================== SVD++-SGD ====================
function [U, V, Y, bu, bi, mu, rated, inv_sqrt_nu] = SGD_SVDpp( ...
    user, item, rate, n_users, n_items, d, T, gama, alpha_u, alpha_v, alpha_y, beta_u, beta_v)

% SVD++ 随机梯度下降训练
%   预测式: r_hat = mu + b_u + b_i + V_i·(U_u + |N(u)|^{-1/2} Σ_{j∈N(u)} y_j)

%   单条评分 (u,i,r) 的残差:  e = r - r_hat
%   沿负梯度更新 (γ 已吸收梯度中的因子 2):
%     b_u ← b_u + γ (e - β_u b_u)
%     b_i ← b_i + γ (e - β_v b_i)
%     U_u ← U_u + γ (e·V_i - α_u U_u)
%     V_i ← V_i + γ (e·p̃_u - α_v V_i)          p̃_u = U_u + |N(u)|^{-1/2} Σ y_j
%     y_j ← y_j + γ (e·|N(u)|^{-1/2}·V_i - α_y y_j)   ∀ j ∈ N(u)
%     μ   ← μ + γ e

    n = numel(rate);

    % ---- 初始化 ----
    mu = mean(rate);                 % 全局均值
    bu = zeros(n_users, 1);          % 用户偏置
    bi = zeros(n_items, 1);          % 物品偏置

    U = 0.01 * (rand(n_users,d) - 0.5);  % 用户因子 p_u
    V = 0.01 * (rand(n_items,d) - 0.5);  % 物品因子 q_i
    Y = 0.01 * (rand(n_items,d) - 0.5);  % 隐式反馈因子 y_j

    % 每个用户评过分的物品集合 N(u) 及其归一化因子 |N(u)|^{-1/2}
    rated = cell(n_users, 1);
    for k = 1:n
        rated{user(k)}(end+1) = item(k);
        % rated{user(k)} = [rated{user(k)}, item(k)];   % 拼接：原列表再接上 item(k)
        % 把当前这条评分的物品 ID，追加到对应用户物品列表的末尾
    end

    inv_sqrt_nu = zeros(n_users,1);
    for u = 1:n_users
        if ~isempty(rated{u})
            inv_sqrt_nu(u) = 1 / sqrt(numel(rated{u})); % |N(u)|^{-1/2}
        end
    end

    for t = 1:T
        order = randperm(n);
        for k = 1:n
            idx = order(k);
            u = user(idx);  i = item(idx);  r = rate(idx);

            % 用户 u 的增强向量 p̃_u = U_u + |N(u)|^{-1/2} * Σ_{j∈N(u)} y_j
            nu = rated{u};
            su = sum(Y(nu,:), 1) * inv_sqrt_nu(u);   % 1×d
            pu_ = U(u,:) + su;                  % 1×d

            % 残差
            e = r - (mu + bu(u) + bi(i) + pu_ * V(i,:)');

            % SGD 更新
            bu(u)  = bu(u)  + gama * (e - beta_u  * bu(u));
            bi(i)  = bi(i)  + gama * (e - beta_v  * bi(i));
            U(u,:) = U(u,:) + gama * (e * V(i,:) - alpha_u * U(u,:));
            V(i,:) = V(i,:) + gama * (e * pu_ - alpha_v * V(i,:));
            Y(nu,:) = Y(nu,:) + gama * (e * inv_sqrt_nu(u) * V(i,:) - alpha_y * Y(nu,:)); 
            mu = mu + gama * e;
        end
        gama = gama * 0.9;
    end
end