%% MF-MPC
%  基于SVD++的多类偏好上下文模型，带leave-one-out
%  修正：预测部分正确获取偏置和隐向量

clear; clc; rng(0);

%% 1. 读取数据
base_file = '../Dataset/ml-100k/ub.base';
test_file = '../Dataset/ml-100k/ub.test';

train = readmatrix(base_file,'FileType','text');
test  = readmatrix(test_file,'FileType','text');

user_train = train(:,1); item_train = train(:,2); rate_train = train(:,3);
user_test  = test(:,1);  item_test  = test(:,2);  rate_test  = test(:,3);

n_users = max([user_train; user_test]);
n_items = max([item_train; item_test]);
fprintf('train = %d 条, test = %d 条\n', size(train,1), size(test,1));

%% 2. 超参数
d       = 20;      % 潜因子维度
T       = 50;      % SGD轮数
gama    = 0.01;    % 初始学习率
lambda  = 0.001;   % 正则化系数
ratings_set = 1:5; % 评分等级

%% 3. 训练
[U, V, Mcell, bu, bi, mu, rated_r] = SGD_MFMPC( ...
    user_train, item_train, rate_train, n_users, n_items, d, T, ...
    gama, lambda, ratings_set);

%% 4. 预测与评估
n_test = numel(rate_test);
pred_test = zeros(n_test,1);

% 冷启动处理：若用户/物品未在训练集出现，使用零偏置和零隐向量
trained_users = unique(user_train);
trained_items = unique(item_train);

for k = 1:n_test
    u = user_test(k);
    i = item_test(k);
    
    % 获取用户偏置和隐向量（冷启动处理）
    if ismember(u, trained_users)
        bu_u = bu(u);
        U_u = U(u,:);
    else
        bu_u = 0;
        U_u = zeros(1, d);
    end
    % 获取物品偏置和隐向量
    if ismember(i, trained_items)
        bi_i = bi(i);
        V_i = V(i,:);
    else
        bi_i = 0;
        V_i = zeros(1, d);
    end
    
    % 计算多类上下文（需剔除目标物品i）
    ubar = compute_ubar(u, i, Mcell, rated_r, ratings_set, d);
    
    % 预测值
    pred_test(k) = mu + bu_u + bi_i + (U_u + ubar) * V_i';
end

% 截断并评估
pred_test_clipped = min(max(pred_test, 1), 5);
rmse = sqrt(mean((pred_test_clipped - rate_test).^2));
mae  = mean(abs(pred_test_clipped - rate_test));
fprintf('MF-MPC(SGD)  d=%-3d  T=%-3d  RMSE = %.4f   MAE = %.4f\n', d, T, rmse, mae);

%% ==================== MF-MPC-SGD 训练函数 ====================
function [U, V, Mcell, bu, bi, mu, rated_r] = SGD_MFMPC( ...
    user, item, rate, n_users, n_items, d, T, gama, lambda, ratings_set)

    n = numel(rate);
    num_r = numel(ratings_set);

    % ---- 初始化 ----
    mu = mean(rate);
    bu = zeros(n_users, 1);
    bi = zeros(n_items, 1);

    U = 0.01 * (rand(n_users,d) - 0.5);
    V = 0.01 * (rand(n_items,d) - 0.5);
    Mcell = cell(num_r, 1);
    for r_idx = 1:num_r
        Mcell{r_idx} = 0.01 * (rand(n_items,d) - 0.5);
    end

    % 构建每个用户各评分等级的物品集合（去重）
    rated_r = cell(n_users, num_r);
    for k = 1:n
        u = user(k);
        r = rate(k);
        rated_r{u, r}(end+1) = item(k);
    end
    % 对每个用户的每个rate去重
    for u = 1:n_users
        for r = 1:num_r
            if ~isempty(rated_r{u, r})
                rated_r{u, r} = unique(rated_r{u, r});
            end
        end
    end

    % ---- SGD训练 ----
    for epoch = 1:T
        order = randperm(n);
        for k = 1:n
            idx = order(k);
            u = user(idx);
            i = item(idx);
            rate_val = rate(idx);

            % 计算用户上下文（含leave-one-out）
            ubar = compute_ubar(u, i, Mcell, rated_r, ratings_set, d);
            pu_ = U(u,:) + ubar;

            % 预测误差
            e = rate_val - (mu + bu(u) + bi(i) + pu_ * V(i,:)');

            % 保存旧物品隐向量（用于更新Mcell）
            V_old_i = V(i,:);

            % ---- 更新参数 ----
            % 更新bu, bi, U（使用旧V）
            bu(u)  = bu(u)  + gama * (e - lambda * bu(u));
            bi(i)  = bi(i)  + gama * (e - lambda * bi(i));
            U(u,:) = U(u,:) + gama * (e * V_old_i - lambda * U(u,:));

            % 更新V（使用当前pu_，即包含旧Mcell）
            V(i,:) = V(i,:) + gama * (e * pu_ - lambda * V(i,:));

            % 更新Mcell（使用V_old_i和e）
            for r_idx = 1:num_r
                rating_level = ratings_set(r_idx);
                items = rated_r{u, rating_level};
                items = items(items ~= i);   % leave-one-out
                if ~isempty(items)
                    Mcell{r_idx}(items,:) = Mcell{r_idx}(items,:) ...
                        + gama * (e * V_old_i / sqrt(numel(items)) - lambda * Mcell{r_idx}(items,:));
                end
            end

            % 更新全局均值
            mu = mu + gama * e;
        end
        % 学习率衰减
        gama = gama * 0.9;
        if mod(epoch, 10) == 0
            fprintf('Epoch %d done, gama=%.6f\n', epoch, gama);
        end
    end
end

%% ==================== 计算多类偏好上下文 (leave-one-out) ====================
function ubar = compute_ubar(u, i, Mcell, rated_r, ratings_set, d)
    % 若u未在训练集出现（或超出范围），则返回零向量
    if u > size(rated_r,1) || all(cellfun(@isempty, rated_r(u,:)))
        ubar = zeros(1, d);
        return;
    end
    
    ubar = zeros(1, d);
    num_r = numel(ratings_set);
    for r_idx = 1:num_r
        rating_level = ratings_set(r_idx);
        items = rated_r{u, rating_level};
        items = items(items ~= i);   % leave-one-out
        if ~isempty(items)
            ubar = ubar + sum(Mcell{r_idx}(items,:), 1) / sqrt(numel(items));
        end
    end
end