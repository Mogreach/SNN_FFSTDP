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


def ff_goodness_branch_loss(goodness, threshold, *, is_pos):
    if is_pos:
        return ff_positive_goodness_loss(goodness, threshold)
    return ff_negative_goodness_loss(goodness, threshold)


def pos_derivative(x, theta):
    # d/dx log(1 + exp(-x + theta))
    return -1 / (1 + torch.exp(x - theta))


def neg_derivative(y, theta):
    # d/dy log(1 + exp(y - theta))
    return 1 / (1 + torch.exp(theta - y))


def _expand_derivative_to_match_activity(derivative, activity):
    if derivative.ndim >= activity.ndim:
        return derivative
    expand_dims = (1,) * (activity.ndim - derivative.ndim)
    return derivative.reshape(*derivative.shape, *expand_dims)


def gradient_calculation_mlp(
    input_spike_sum,
    goodness_input_gradient,
    goodness,
    ln_var,
    ln_mean,
    loss_threshold,
    v_threshold,
    N,
    is_pos,
):
    del ln_var, ln_mean, v_threshold
    derivative = (
        pos_derivative(goodness, loss_threshold)
        if is_pos
        else neg_derivative(goodness, loss_threshold)
    )
    loss = ff_goodness_branch_loss(goodness, loss_threshold, is_pos=is_pos)
    L_to_s_grad = goodness_input_gradient * derivative
    L_to_s_grad = L_to_s_grad.transpose(0, 1)
    weight_grad = -1 * L_to_s_grad @ input_spike_sum / N
    return weight_grad, loss


def pairwise_loss_gradient_calculation_mlp(
    pos_input_spike_sum,
    pos_goodness_input_gradient,
    pos_goodness,
    pos_ln_var,
    pos_ln_mean,
    neg_input_spike_sum,
    neg_goodness_input_gradient,
    neg_goodness,
    neg_ln_var,
    neg_ln_mean,
    threshold,
    v_threshold,
    N,
):
    del pos_ln_var, pos_ln_mean, neg_ln_var, neg_ln_mean, v_threshold
    pos_L_to_s_grad = (
        pos_goodness_input_gradient * pos_derivative(pos_goodness, threshold)
    )
    pos_L_to_s_grad = pos_L_to_s_grad.transpose(0, 1)
    pos_weight_grad = -1 * pos_L_to_s_grad @ pos_input_spike_sum / N

    neg_L_to_s_grad = (
        neg_goodness_input_gradient * neg_derivative(neg_goodness, threshold)
    )
    neg_L_to_s_grad = neg_L_to_s_grad.transpose(0, 1)
    neg_weight_grad = -1 * neg_L_to_s_grad @ neg_input_spike_sum / N

    pairwise_loss = ff_pairwise_goodness_loss(
        pos_goodness,
        neg_goodness,
        threshold,
    )
    return pos_weight_grad + neg_weight_grad, pairwise_loss


def delta_loss_gradient_calculation_mlp(
    pos_input_spike_sum,
    pos_goodness_input_gradient,
    pos_goodness,
    pos_ln_var,
    pos_ln_mean,
    neg_input_spike_sum,
    neg_goodness_input_gradient,
    neg_goodness,
    neg_ln_var,
    neg_ln_mean,
    alpha,
    v_threshold,
    N,
):
    del pos_ln_var, pos_ln_mean, neg_ln_var, neg_ln_mean, v_threshold
    delta = alpha * (pos_goodness - neg_goodness)
    pos_L_to_s_grad = alpha * pos_derivative(delta, 0) * pos_goodness_input_gradient
    pos_L_to_s_grad = pos_L_to_s_grad.transpose(0, 1)
    pos_weight_grad = -1 * pos_L_to_s_grad @ pos_input_spike_sum / N

    neg_L_to_s_grad = -alpha * pos_derivative(delta, 0) * neg_goodness_input_gradient
    neg_L_to_s_grad = neg_L_to_s_grad.transpose(0, 1)
    neg_weight_grad = -1 * neg_L_to_s_grad @ neg_input_spike_sum / N

    delta_loss = ff_supervised_delta_loss(
        pos_goodness,
        neg_goodness,
        alpha,
    )
    return pos_weight_grad + neg_weight_grad, delta_loss


def gradient_calculation_cnn(
    input_spike_sum_unfold,
    goodness_input_gradient,
    goodness,
    ln_var,
    ln_mean,
    loss_threshold,
    v_threshold,
    B,
    Cout,
    is_pos,
):
    del ln_mean
    derivative = (
        pos_derivative(goodness, loss_threshold)
        if is_pos
        else neg_derivative(goodness, loss_threshold)
    )
    derivative = _expand_derivative_to_match_activity(derivative, goodness_input_gradient)
    loss = ff_goodness_branch_loss(goodness, loss_threshold, is_pos=is_pos)
    L_to_s_grad = goodness_input_gradient * derivative * (
        v_threshold / torch.sqrt(ln_var.view(B, 1, 1, 1) + 1e-5)
    )
    L_to_s_grad = L_to_s_grad.view(B, Cout, -1)
    L_to_s_grad = L_to_s_grad.permute(1, 0, 2).reshape(Cout, -1)
    weight_grad = -1 * (L_to_s_grad @ input_spike_sum_unfold.T) / B
    return weight_grad, loss


def pairwise_loss_gradient_calculation_cnn(
    pos_input_spike_sum_unfold,
    pos_goodness_input_gradient,
    pos_goodness,
    pos_ln_var,
    pos_ln_mean,
    neg_input_spike_sum_unfold,
    neg_goodness_input_gradient,
    neg_goodness,
    neg_ln_var,
    neg_ln_mean,
    threshold,
    v_threshold,
    B,
    Cout,
):
    del pos_ln_mean, neg_ln_mean
    pos_derivative_value = _expand_derivative_to_match_activity(
        pos_derivative(pos_goodness, threshold),
        pos_goodness_input_gradient,
    )
    pos_L_to_s_grad = pos_goodness_input_gradient * pos_derivative_value * (
        v_threshold / torch.sqrt(pos_ln_var.view(B, 1, 1, 1) + 1e-5)
    )
    pos_L_to_s_grad = pos_L_to_s_grad.view(B, Cout, -1)
    pos_L_to_s_grad = pos_L_to_s_grad.permute(1, 0, 2).reshape(Cout, -1)
    pos_weight_grad = -1 * (pos_L_to_s_grad @ pos_input_spike_sum_unfold.T) / B

    neg_derivative_value = _expand_derivative_to_match_activity(
        neg_derivative(neg_goodness, threshold),
        neg_goodness_input_gradient,
    )
    neg_L_to_s_grad = neg_goodness_input_gradient * neg_derivative_value * (
        v_threshold / torch.sqrt(neg_ln_var.view(B, 1, 1, 1) + 1e-5)
    )
    neg_L_to_s_grad = neg_L_to_s_grad.view(B, Cout, -1)
    neg_L_to_s_grad = neg_L_to_s_grad.permute(1, 0, 2).reshape(Cout, -1)
    neg_weight_grad = -1 * (neg_L_to_s_grad @ neg_input_spike_sum_unfold.T) / B

    pairwise_loss = ff_pairwise_goodness_loss(
        pos_goodness,
        neg_goodness,
        threshold,
    )
    return pos_weight_grad + neg_weight_grad, pairwise_loss


def delta_loss_gradient_calculation_cnn(
    pos_input_spike_sum_unfold,
    pos_goodness_input_gradient,
    pos_goodness,
    pos_ln_var,
    pos_ln_mean,
    neg_input_spike_sum_unfold,
    neg_goodness_input_gradient,
    neg_goodness,
    neg_ln_var,
    neg_ln_mean,
    alpha,
    v_threshold,
    B,
    Cout,
):
    del pos_ln_mean, neg_ln_mean
    delta = alpha * (pos_goodness - neg_goodness)
    delta_derivative = _expand_derivative_to_match_activity(
        pos_derivative(delta, 0),
        pos_goodness_input_gradient,
    )
    pos_L_to_s_grad = alpha * delta_derivative * pos_goodness_input_gradient * (
        v_threshold / torch.sqrt(pos_ln_var.view(B, 1, 1, 1) + 1e-5)
    )
    pos_L_to_s_grad = pos_L_to_s_grad.view(B, Cout, -1)
    pos_L_to_s_grad = pos_L_to_s_grad.permute(1, 0, 2).reshape(Cout, -1)
    pos_weight_grad = -1 * (pos_L_to_s_grad @ pos_input_spike_sum_unfold.T) / B

    neg_L_to_s_grad = -alpha * delta_derivative * neg_goodness_input_gradient * (
        v_threshold / torch.sqrt(neg_ln_var.view(B, 1, 1, 1) + 1e-5)
    )
    neg_L_to_s_grad = neg_L_to_s_grad.view(B, Cout, -1)
    neg_L_to_s_grad = neg_L_to_s_grad.permute(1, 0, 2).reshape(Cout, -1)
    neg_weight_grad = -1 * (neg_L_to_s_grad @ neg_input_spike_sum_unfold.T) / B

    delta_loss = ff_supervised_delta_loss(
        pos_goodness,
        neg_goodness,
        alpha,
    )
    return pos_weight_grad + neg_weight_grad, delta_loss
