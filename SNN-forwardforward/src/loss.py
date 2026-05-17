import torch


def ff_positive_goodness_loss(goodness, threshold):
    return torch.log(1 + torch.exp(threshold - goodness)).mean()


def ff_negative_goodness_loss(goodness, threshold):
    return torch.log(1 + torch.exp(goodness - threshold)).mean()


def ff_pairwise_goodness_loss(pos_goodness, neg_goodness, threshold):
    return torch.log(
        1 + torch.exp(torch.cat([-pos_goodness + threshold, neg_goodness - threshold]))
    ).mean()


def ff_supervised_delta_loss(pos_goodness, neg_goodness, alpha):
    delta = pos_goodness - neg_goodness
    return torch.log(1 + torch.exp(-alpha * delta)).mean()


def ff_scaled_supervised_delta_loss(pos_goodness, neg_goodness, alpha):
    scaled_delta = alpha * (pos_goodness - neg_goodness)
    return torch.log(1 + torch.exp(-alpha * scaled_delta)).mean()


def pos_derivative(x, theta):
    """
    计算 log(1 + exp(-x + theta)) 关于 x 的导数。

    参数:
        x (np.ndarray): 输入值。
        theta (float): 参数 theta。

    返回:
        np.ndarray: 导数值。
    """
    # 计算 Sigmoid 函数
    sigmoid = -1 / (1 + torch.exp(x - theta))
    
    # 返回导数
    return sigmoid
def neg_derivative(y, theta):
    """
    计算 log(1 + exp(y - theta)) 关于 y 的导数。

    参数:
        y (np.ndarray): 输入值。
        theta (float): 参数 theta。

    返回:
        np.ndarray: 导数值。
    """
    # 计算 Sigmoid 函数
    sigmoid = 1 / (1 + torch.exp(theta - y))
    
    # 返回导数
    return sigmoid

def gradient_calculation_mlp(input_spike_sum, out_freq, goodness, ln_var, ln_mean,
                            loss_threshold, v_threshold, N, is_pos):
    if is_pos:
        derivative = pos_derivative(goodness, loss_threshold)
        loss = ff_positive_goodness_loss(goodness, loss_threshold)
    else:
        derivative = neg_derivative(goodness, loss_threshold)
        loss = ff_negative_goodness_loss(goodness, loss_threshold)
    L_to_s_grad = 2 * out_freq* derivative
    L_to_s_grad = L_to_s_grad.transpose(0,1)
    weight_grad = -1 * L_to_s_grad @ input_spike_sum / N
    
    return weight_grad, loss
def delta_loss_gradient_calculation_mlp(pos_input_spike_sum, pos_out_freq, pos_goodness, pos_ln_var, pos_ln_mean,
                         neg_input_spike_sum, neg_out_freq, neg_goodness, neg_ln_var, neg_ln_mean,
                         alpha, v_threshold, N):
        delta = alpha * (pos_goodness - neg_goodness)
        pos_L_to_s_grad = alpha * pos_derivative(delta,0) * 2 * pos_out_freq
        pos_L_to_s_grad = pos_L_to_s_grad.transpose(0,1)
        pos_weight_grad = -1 * pos_L_to_s_grad @ pos_input_spike_sum / N
       
        neg_L_to_s_grad = -alpha * pos_derivative(delta,0) * 2 * neg_out_freq
        neg_L_to_s_grad = neg_L_to_s_grad.transpose(0,1)
        neg_weight_grad = -1 * neg_L_to_s_grad @ neg_input_spike_sum / N

        delta_loss = ff_supervised_delta_loss(
            pos_goodness,
            neg_goodness,
            alpha,
        )
        weight_grad = pos_weight_grad + neg_weight_grad

        return weight_grad, delta_loss
def gradient_calculation_cnn(input_spike_sum_unfold, out_freq, goodness, ln_var, ln_mean,
                         loss_threshold, v_threshold, B, Cout, is_pos):
    if is_pos:
        derivative = pos_derivative(goodness, loss_threshold)
        loss = ff_positive_goodness_loss(goodness, loss_threshold)
    else:
        derivative = neg_derivative(goodness, loss_threshold)
        loss = ff_negative_goodness_loss(goodness, loss_threshold)
    L_to_s_grad = 2 * out_freq * derivative * (v_threshold / torch.sqrt(ln_var.view(B,1,1,1) + 1e-5))
    L_to_s_grad = L_to_s_grad.view(B, Cout, -1)  # [B,Cout,Hout*Wout]
    # L_to_s_grad [B, Cout, Hout*Wout] → [Cout, B*Hout*Wout]
    L_to_s_grad = L_to_s_grad.permute(1, 0, 2).reshape(Cout, -1)
    # weight_grad [C_out, B*Hout*Wout] @ [B*Hout*Wout, Cin*Kh*Kw] → [C_out, Cin*Kh*Kw]
    weight_grad = -1 * (L_to_s_grad @ input_spike_sum_unfold.T) / B
    return weight_grad, loss
def delta_loss_gradient_calculation_cnn(pos_input_spike_sum_unfold, pos_out_freq, pos_goodness, pos_ln_var, pos_ln_mean,
                         neg_input_spike_sum_unfold, neg_out_freq, neg_goodness, neg_ln_var, neg_ln_mean,
                         alpha, v_threshold, B, Cout):
    delta = alpha * (pos_goodness - neg_goodness)
    pos_L_to_s_grad = alpha * pos_derivative(delta,0) * 2 * pos_out_freq  * (v_threshold / torch.sqrt(pos_ln_var.view(B,1,1,1) + 1e-5))
    pos_L_to_s_grad = pos_L_to_s_grad.view(B, Cout, -1)  # [B,Cout,Hout*Wout]
    # pos_L_to_s_grad [B, Cout, Hout*Wout] → [Cout, B*Hout*Wout]
    pos_L_to_s_grad = pos_L_to_s_grad.permute(1, 0, 2).reshape(Cout, -1)
    # weight_grad [C_out, B*Hout*Wout] @ [B*Hout*Wout, Cin*Kh*Kw] → [C_out, Cin*Kh*Kw]
    pos_weight_grad = -1 * (pos_L_to_s_grad @ pos_input_spike_sum_unfold.T) / B

    neg_L_to_s_grad = -alpha * pos_derivative(delta,0) * 2 * neg_out_freq * (v_threshold / torch.sqrt(neg_ln_var.view(B,1,1,1) + 1e-5))
    neg_L_to_s_grad = neg_L_to_s_grad.view(B, Cout, -1)  # [B,Cout,Hout*Wout]
    # neg_L_to_s_grad [B, Cout, Hout*Wout] → [Cout, B*Hout*Wout]
    neg_L_to_s_grad = neg_L_to_s_grad.permute(1, 0, 2).reshape(Cout, -1)
    # weight_grad [C_out, B*Hout*Wout] @ [B*Hout*Wout, Cin*Kh*Kw] → [C_out, Cin*Kh*Kw]
    neg_weight_grad = -1 * (neg_L_to_s_grad @ neg_input_spike_sum_unfold.T) / B

    delta_loss = ff_supervised_delta_loss(
        pos_goodness,
        neg_goodness,
        alpha,
    )
    weight_grad = pos_weight_grad + neg_weight_grad
    
    return weight_grad, delta_loss
