%% ALS (alternating least squares) 
clear; clc;

%% 1. 数据读取
base_file = '../Dataset/ml-100k/u1.base';
test_file = '../Dataset/ml-100k/u1.test';

train_raw = readmatrix(base_file,'FileType','text');
train_raw = train_raw(:,1:3);
test_vec  = readmatrix(test_file,'FileType','text');
test_vec  = test_vec(:,1:3);

TrainRatingNumber = size(train_raw,1);
f = randperm(TrainRatingNumber);
train_vec = train_raw( f(1 : floor(TrainRatingNumber*0.8)), : );
valid_vec = train_raw( f(floor(TrainRatingNumber*0.8)+1 : end), : );

n_users = max([train_vec(:,1); valid_vec(:,1); test_vec(:,1)]);
n_items = max([train_vec(:,2); valid_vec(:,2); test_vec(:,2)]);

fprintf("train = %d 条, valid = %d 条, test = %d 条\n", ...
        size(train_vec,1), size(valid_vec,1), size(test_vec,1));

%% 2. 构建训练评分矩阵 R
R = nan(n_users, n_items);
R(sub2ind([n_users, n_items], train_vec(:,1), train_vec(:,2))) = train_vec(:,3);

% 测试集线性索引
te_lin = sub2ind([n_users, n_items], test_vec(:,1), test_vec(:,2));
% 验证集线性索引
valid_lin = sub2ind([n_users, n_items], valid_vec(:,1), valid_vec(:,2));

%% 3. 超参数
d = 20;
alpha = 0.01;
MAX_EPOCH = 100;

%% 4. ALS 训练
[U, V, avg_rating] = ALS_PMF(R, d, alpha, MAX_EPOCH, train_vec(:,3), valid_vec, valid_lin);

%% 5. 预测与评估
pred = U * V' + avg_rating;
pred_test = pred(te_lin);
pred_test = min(max(pred_test, 1), 5);  
rmse = sqrt(mean((pred_test - test_vec(:,3)).^2));
mae  = mean(abs(pred_test - test_vec(:,3)));
fprintf('Final Test: RMSE = %.4f, MAE = %.4f\n', rmse, mae);

%% 6. ALS 函数
function [U, V, avg_rating] = ALS_PMF(R, d, alpha, MAX_EPOCH, rate_tr, valid_vec, valid_lin)
    [n_users, n_items] = size(R);
    valid = ~isnan(R);

    % 预计算索引
    user_items = cell(n_users, 1);
    item_users = cell(n_items, 1);
    for u = 1:n_users
        user_items{u} = find(valid(u, :));
    end
    for i = 1:n_items
        item_users{i} = find(valid(:, i));
    end

    % ---- 初始化 ----
    avg_rating = mean(rate_tr);
    rand('state',0);
    U = avg_rating / d + 0.01 * (rand(n_users, d) - 0.5);
    rand('state',1);
    V = avg_rating / d + 0.01 * (rand(n_items, d) - 0.5);

    % ---- 中心化训练评分 ----
    R(valid) = R(valid) - avg_rating;

    % ---- 早停变量 ----
    best_val_rmse = inf;
    best_U = U;
    best_V = V;
    prev_val_rmse = inf;

    for iter = 1:MAX_EPOCH
        % 更新 V（先物品）
        for i = 1:n_items
            Ii = item_users{i};
            if isempty(Ii), continue; end
            Ui = U(Ii, :);
            r  = R(Ii, i);
            n_i = length(Ii);
            V(i,:) = (Ui' * r)' / (Ui' * Ui + alpha * n_i * eye(d));
        end

        % 更新 U（后用户）
        for u = 1:n_users
            Iu = user_items{u};
            if isempty(Iu), continue; end
            Vi = V(Iu, :);
            r  = R(u, Iu)';
            n_u = length(Iu);
            U(u,:) = (Vi' * r)' / (Vi' * Vi + alpha * n_u * eye(d));
        end

        % ---- 计算验证 RMSE ----
        pred_val = U * V' + avg_rating;
        pred_val = pred_val(valid_lin);
        pred_val = min(max(pred_val, 1), 5);
        val_rmse = sqrt(mean((pred_val - valid_vec(:,3)).^2));

        % % 保存验证集上最优模型（若验证 RMSE 下降则更新）
        if val_rmse < best_val_rmse
            best_val_rmse = val_rmse;
            best_U = U;
            best_V = V;
        end

        % ---- 早停条件 ----
        if iter > 1
            if (prev_val_rmse < val_rmse) || (abs(prev_val_rmse - val_rmse) < 1e-4)
                fprintf('Early stopping at epoch %d (val RMSE = %.4f)\n', iter, val_rmse);
                break;
            end
        end
        prev_val_rmse = val_rmse;

        if mod(iter, 10) == 0
            fprintf('Epoch %d, val RMSE = %.4f\n', iter, val_rmse);
        end
    end


    U = best_U;
    V = best_V;
end