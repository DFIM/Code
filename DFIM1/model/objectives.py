import torch
import torch.nn as nn
import torch.nn.functional as F
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_sscau_loss(image_features,augmented_image_features, alpha=0.01):

    f_a_norm = (image_features - image_features.mean(0)) / (image_features.std(0) + 1e-6)
    f_b_norm = (augmented_image_features - augmented_image_features.mean(0)) / (augmented_image_features.std(0) + 1e-6)
    c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
    off_diag = off_diagonal(c).pow_(2).mean()
    factorization_loss = on_diag + alpha * off_diag

    return factorization_loss

def compute_ssimg_loss(aug_i_feats, i_feats, temperature):

    bs = aug_i_feats.size(0)

    labels = torch.arange(bs, device=aug_i_feats.device)

    aug_i_feats = F.normalize(aug_i_feats, dim=-1)
    i_feats = F.normalize(i_feats, dim=-1)

    sim_aa = aug_i_feats @ aug_i_feats.t()
    sim_aa = temperature * sim_aa

    sim_ab = aug_i_feats @ i_feats.t()
    sim_ab = temperature * sim_ab

    sim_ba = i_feats @ aug_i_feats.t()
    sim_ba = temperature * sim_ba

    sim_bb = i_feats @ i_feats.t()
    sim_bb = temperature * sim_bb

    masks = torch.where(F.one_hot(labels, aug_i_feats.size(0)) == 0, 0, float('-inf'))
    sim_aa += masks
    sim_bb += masks
    sim_a = torch.cat([sim_ab, sim_aa], 1)
    sim_b = torch.cat([sim_ba, sim_bb], 1)
    loss_a = F.cross_entropy(sim_a, labels)
    loss_b = F.cross_entropy(sim_b, labels)

    return (loss_a + loss_b) * 0.5

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

def KL(alpha, c):
    beta = torch.ones((1, c)).to(alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def mse_loss(label, alpha, c, lambda2):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    C = lambda2 * KL(alp, c)
    return (A + B) + C

def mse_loss_tanh(label, alpha, c, lambda2):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    u_1 = alpha.size(0) / S
    min_u = torch.min(u_1)
    max_u = torch.max(u_1)
    u_norm = (u_1 - min_u) / (max_u - min_u)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    C = lambda2 * KL(alp, c)
    return (A + B) + C

def get_alpha_t(sims):
    evidences = torch.exp(torch.tanh(sims) / 0.1)
    sum_e = evidences + evidences.t()
    norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
    alpha_i2t = evidences + 1
    alpha_t2i = evidences.t() + 1
    return alpha_i2t, alpha_t2i, norm_e

def get_alpha(sims):
    evidences = F.relu(sims)
    sum_e = evidences + evidences.t()
    norm_e = sum_e / torch.sum(sum_e, dim=1, keepdim=True)
    alpha_i2t = evidences + 1
    alpha_t2i = evidences.t() + 1
    return alpha_i2t, alpha_t2i, norm_e

def course_function(epoch, total_epoch, total_snippet_num, amplitude):
    idx = torch.arange(total_snippet_num)
    theta = 2 * (idx + 0.5) / total_snippet_num - 1
    delta = - 2 * epoch / total_epoch + 1
    curve = amplitude * torch.tanh(theta * delta) + 1

    return curve

def get_unc_loss(image_fetures, text_fetures, epoch, total_epoch, amplititude=0.7):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = torch.arange(batch_size, device=image_fetures.device)
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()

    text_proj_image = t2i_cosine_theta

    alpha_t2i, alpha_i2t, _ = get_alpha(text_proj_image)
    alpha_t2i_t, alpha_i2t_t, _ = get_alpha_t(text_proj_image)

    loss = (mse_loss_tanh(labels, alpha_t2i_t, batch_size, 0.1)) + 1.0 * (
        mse_loss_tanh(labels, alpha_i2t_t, batch_size, 0.1))
    loss1 = (mse_loss(labels, alpha_t2i, batch_size, 0.1)) + 1.0 * (
        mse_loss(labels, alpha_i2t, batch_size, 0.1))
    total_loss_guide = 0.5 * loss + 0.5 * loss1

    return total_loss_guide.mean()


def get_cyclic_loss_one(image_fetures, text_fetures, logit_scale,citc_lambda1,citc_lambda2):

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    logits_image_per_image = logit_scale * image_norm @ image_norm.t()
    logits_text_per_text = logit_scale * text_norm @ text_norm.t()
    inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / ( logit_scale * logit_scale )

    logits_text_per_image = logit_scale * image_norm @ text_norm.t()
    logits_image_per_text = logit_scale * text_norm @ image_norm.t()
    crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / ( logit_scale * logit_scale)

    citc_loss = citc_lambda1 * inmodal_cyclic_loss + citc_lambda2 * crossmodal_cyclic_loss

    return citc_loss

def info_nce_loss(similarity_matrix, temp=None):
    if temp is None:
        temp = 0.07
    labels = torch.eye(len(similarity_matrix)).float().cuda()

    # select and combine multiple positives
    pos = similarity_matrix[labels.bool()].view(similarity_matrix.shape[0], -1)
    # preds = pos
    pos = torch.exp(pos / temp).squeeze(1)

    # select only the negatives the negatives
    neg = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    neg = torch.exp(neg / temp)

    neg_sum = neg.sum(1)

    loss = -(torch.log(pos / (pos + neg_sum)))
    preds = pos / (pos + neg_sum)
    return loss, preds

def get_cyclic_loss_two(image_fetures, text_fetures, logit_scale,gamma):

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    logits_text_per_image = logit_scale * image_norm @ text_norm.t()
    logits_image_per_text = logit_scale * text_norm @ image_norm.t()

    loss_cl, _ = info_nce_loss(logits_text_per_image)
    loss_cl_t, _ = info_nce_loss(logits_image_per_text)
    loss_cl = (loss_cl + loss_cl_t) * 0.5

    logits_image_per_image = logit_scale * image_norm @ image_norm.t()
    logits_text_per_text = logit_scale * text_norm @ text_norm.t()

    sims_img = torch.sigmoid(logits_image_per_image)
    sims_cap = torch.sigmoid(logits_text_per_text)

    sims_topo = sims_img @ sims_cap.t()

    loss_topo, _ = info_nce_loss(sims_topo, temp=1.0)

    loss = loss_cl.mean() + loss_topo.mean() * gamma

    return loss

def compute_simclr_loss(logits_a, logits_b, logits_a_gathered, logits_b_gathered, labels, temperature):

    sim_aa = logits_a @ logits_a_gathered.t() / temperature
    sim_ab = logits_a @ logits_b_gathered.t() / temperature
    sim_ba = logits_b @ logits_a_gathered.t() / temperature
    sim_bb = logits_b @ logits_b_gathered.t() / temperature
    masks = torch.where(F.one_hot(labels, logits_a_gathered.size(0)) == 0, 0, float('-inf'))
    sim_aa += masks
    sim_bb += masks
    sim_a = torch.cat([sim_ab, sim_aa], 1)
    sim_b = torch.cat([sim_ba, sim_bb], 1)
    loss_a = F.cross_entropy(sim_a, labels)
    loss_b = F.cross_entropy(sim_b, labels)

    return (loss_a + loss_b) * 0.5
