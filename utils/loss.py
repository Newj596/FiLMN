# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
import torch.nn.functional as F


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


# class FocalLoss(nn.Module):
#     """Implements truncated focal loss with confidence clipping to mitigate noise learning."""
#
#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25, trunc_thresh=0.2):
#         super().__init__()
#         self.loss_fcn = loss_fcn  # Must be BCEWithLogitsLoss
#         self.gamma = gamma
#         self.alpha = alpha
#         self.trunc_thresh = trunc_thresh
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = "none"
#
#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         pred_prob = torch.sigmoid(pred)
#
#         # Confidence truncation mechanism
#         p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
#         truncated_p = torch.clamp(p_t, min=self.trunc_thresh)  # Confidence floor
#
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = (1.0 - truncated_p) ** self.gamma  # Use truncated confidence
#
#         loss *= alpha_factor * modulating_factor
#
#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         return loss


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                # lbox += (1.0 - iou).mean()  # iou loss
                focal_iou_loss = ((1.0 - iou) ** 1.2).mean()  # gamma例如取1.5
                lbox += focal_iou_loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
                torch.tensor(
                    [
                        [0, 0],
                        [1, 0],
                        [0, 1],
                        [-1, 0],
                        [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                    ],
                    device=self.device,
                ).float()
                * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            targets = targets[:, 0:-1]
            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeLossLD:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                weight = targets[:, -1].mean()
                lbox += (weight * (1.0 - iou)).mean()  # iou loss
                # pbox = self.generalized_softmax(pbox, temperature=10)
                # tbox[i] = self.generalized_softmax(tbox[i], temperature=10)
                # lbox += self.kl_divergence_loss(pbox, tbox[i])

                # prob_boxes = self.local2dis_joint(pbox, N=10, temperature=10.0)
                # prob_teacher_boxes = self.local2dis_joint(tbox[i], N=10, temperature=10.0)
                # for prob_box, prob_techer_box in zip(prob_boxes, prob_teacher_boxes):
                #     lbox += self.kl_divergence_loss(prob_box, prob_techer_box)
                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def kl_divergence_loss(self, p, q):
        return F.kl_div(F.log_softmax(q, dim=-1), p, reduction='batchmean')
        # return F.cross_entropy(q, p.argmax(dim=-1))

    def get_bin_indices(self, coords, bins, N):
        indices = torch.bucketize(coords, bins) - 1
        indices = torch.clamp(indices, 0, N - 1)
        return indices

    def softmax_with_temperature(self, logits, temperature):
        scaled_logits = logits / temperature
        return torch.nn.functional.softmax(scaled_logits, dim=-1)

    # def local2dis(self, boxes, N, temperature):
    #     bins = torch.linspace(0, 1, N + 1).to(boxes.device)
    #     x1_bins = self.get_bin_indices(boxes[:, 0], bins, N)
    #     y1_bins = self.get_bin_indices(boxes[:, 1], bins, N)
    #     x2_bins = self.get_bin_indices(boxes[:, 2], bins, N)
    #     y2_bins = self.get_bin_indices(boxes[:, 3], bins, N)
    #
    #     hist_x1 = torch.zeros(N, dtype=torch.float32)
    #     hist_y1 = torch.zeros(N, dtype=torch.float32)
    #     hist_x2 = torch.zeros(N, dtype=torch.float32)
    #     hist_y2 = torch.zeros(N, dtype=torch.float32)
    #
    #     for i in range(boxes.shape[0]):
    #         hist_x1[x1_bins[i]] += 1
    #         hist_y1[y1_bins[i]] += 1
    #         hist_x2[x2_bins[i]] += 1
    #         hist_y2[y2_bins[i]] += 1
    #
    #     prob_x1 = hist_x1
    #     prob_y1 = hist_y1
    #     prob_x2 = hist_x2
    #     prob_y2 = hist_y2
    #
    #     smoothed_prob_x1 = self.softmax_with_temperature(prob_x1, temperature)
    #     smoothed_prob_y1 = self.softmax_with_temperature(prob_y1, temperature)
    #     smoothed_prob_x2 = self.softmax_with_temperature(prob_x2, temperature)
    #     smoothed_prob_y2 = self.softmax_with_temperature(prob_y2, temperature)
    #
    #     probs = torch.stack([smoothed_prob_x1, smoothed_prob_y1, smoothed_prob_x2, smoothed_prob_y2], dim=0)
    #     # return smoothed_prob_x1, smoothed_prob_y1, smoothed_prob_x2, smoothed_prob_y2
    #     return probs

    def local2dis_joint(self, boxes, N, temperature):
        bins = torch.linspace(0, 1, N + 1).to(boxes.device)
        x1_bins = self.get_bin_indices(boxes[:, 0], bins, N)
        y1_bins = self.get_bin_indices(boxes[:, 1], bins, N)
        x2_bins = self.get_bin_indices(boxes[:, 2], bins, N)
        y2_bins = self.get_bin_indices(boxes[:, 3], bins, N)

        indices_x = x1_bins * N + x2_bins
        hist_x_joint = torch.bincount(indices_x, minlength=N * N).view(N, N).float().to(boxes.device)
        prob_x_joint = hist_x_joint / hist_x_joint.sum() if hist_x_joint.sum() != 0 else hist_x_joint

        indices_y = y1_bins * N + y2_bins
        hist_y_joint = torch.bincount(indices_y, minlength=N * N).view(N, N).float().to(boxes.device)
        prob_y_joint = hist_y_joint / hist_y_joint.sum() if hist_y_joint.sum() != 0 else hist_y_joint

        smoothed_joint_x = self.softmax_with_temperature(prob_x_joint.view(-1), temperature)
        smoothed_joint_y = self.softmax_with_temperature(prob_y_joint.view(-1), temperature)

        probs = torch.stack([smoothed_joint_x, smoothed_joint_y], dim=0)

        # return smoothed_joint_x, smoothed_joint_y
        return probs
    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
                torch.tensor(
                    [
                        [0, 0],
                        [1, 0],
                        [0, 1],
                        [-1, 0],
                        [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                    ],
                    device=self.device,
                ).float()
                * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            targets = targets[:, 0:-1]
            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

