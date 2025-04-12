'''Implementation of single shot multibox detection (SSD) object detection from scratch.'''

import torch
import toolbox as ricky
import matplotlib.pyplot as plt
import os
import torchvision
from torch import nn
import torch.nn.functional as F



def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx,cy,w,h),axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """Convert from (center, weidth, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1,y1,x2,y2), axis=-1)
    return boxes


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(
        xy = (bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
              fill=False, edgecolor=color, linewidth=2)


def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    """
    Input:
        data (tensor): images of shape (batch_size, c, h, w), 
                       top-left-bottom-right representation
        sizes (list): list of float numbers denoting scales
        ratios (list): list of float numbers denoting aspect ratios
    Output:
        output (tensor): anchor boxes of shape (1, num of anchor boxes, 4)
                         top-left-bottom-right representation
    """
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # Scaled steps in y axis
    steps_w = 1.0 / in_width # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate 'boxes_per_pixel' number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor*torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.vstack((-w, -h, w, h)).T.repeat(
                                            in_height*in_width,1)/2
    
    # Each center point will have 'boxes_per_pixel' number of anchor boxes, so
    # generate a grid of all anchor boxes centers with 'boxes_per_pixel' repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""
    """
    Input:
        axes: fig.axes
        bboxes (tensor): (num of anchor boxes, 4)
                          top-left-bottom-right representation
        labels (list): list of labels strings
        colors (list): list of colors strings
    Output:
        None
    """

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
        
    labels = make_list(labels)
    colors = make_list(colors, ['b','g','r','m','c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color=='w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                        va='center', ha='center', fontsize=9, color=text_color,
                        bbox=dict(facecolor=color, lw=0))


def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    """
    Input:
        boxes1 (tensor): (num of anchor boxes1 (N), 4)
                         top-left-bottom-right representation
        boxes2 (tensor): (num of anchor boxes2 (M), 4)
                         top-left-bottom-right representation
    Output:
        inter_areas/union_areas (tensor): (N, M)
    """
    box_area = lambda boxes: ((boxes[:,2]-boxes[:,0])*
                              (boxes[:,3]-boxes[:,1]))
    # Shape of 'boxes1', 'boxes2', 'areas1', 'areas2': (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_achor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the ith row and jth column is the IoU of the anchor
    # box i and the ground-truth bounding j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                 device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,),-1)
    row_discard = torch.full((num_gt_boxes,),-1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard) # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offset."""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:,:2] - c_anc[:,:2]) / c_anc[:,2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    """
    Input:
        anchors (tensor): (batch_size, num_anchors, 4)
        labels: (tensor): (batch_size, num_boxes, 5)
    Output:
        class_labels (tensor): (batch_size, num_anchors)
        bbox_mask (tensor): (batch_size, 4 * num_anchors)
        bbox_offset (tensor): (batch_size, 4 * num_anchors)
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i,:,:]
        anchors_bbox_map = assign_achor_to_bbox(label[:,1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1,4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors,4), dtype=torch.float32, device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map>=0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx,0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """Predicting bounding boxes based on anchor boxes with predicted offsets."""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:,:2]*anc[:,2:]/10) + anc[:,:2]
    pred_bbox_wh = torch.exp(offset_preds[:,2:]/5) * anc[:,2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    """
    Input:
        boxes (tensor): (num_anchors, 4)
        scores (tensor): (num_anchors,)
    Output:
        torch.tensor(keep, device=boxes.device): (num of keep boxes,)
    """
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = [] # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i,:].reshape(-1,4),
                      boxes[B[1:], :].reshape(-1,4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds+1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999, nms_flag=True):
    """Predict bounding boxes using non-maximal suppression."""
    """
    Input:
        cls_probs (tensor): (batch_size, num_classes, num_anchors)
        offset_preds (tensor): (batch_size, num_anchors * 4)
        anchors (tensor): (batch_size, num_anchors, 4)
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1,4)
        conf, class_id = torch.max(cls_prob[1:],0)
        prediected_bb = offset_inverse(anchors, offset_pred)
        if nms_flag:
            keep = nms(prediected_bb, conf, nms_threshold)
        else:
            keep = nms(prediected_bb, conf, iou_threshold=1)
        # Find all non-'keep' indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts==1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, prediected_bb = conf[all_id_sorted], prediected_bb[all_id_sorted]
        # Here 'pos_threshold' is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1), prediected_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

def test_multibox_detection():
    anchors = torch.tensor(
        [[0.1,0.08,0.52,0.92],[0.08,0.2,0.56,0.95],
         [0.15,0.3,0.62,0.91], [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor([[0] * 4,  # Predicted background likelihood
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
    fig = ricky.plt.imshow(img)
    h, w = img.shape[:2]
    bbox_scale = torch.tensor((w,h,w,h))
    #show_bboxes(fig.axes, anchors * bbox_scale,
    #        ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
    output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
    for i in output[0].detach().numpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
        

def display_anchors(fmap_w, fmap_h, s):
    # Values on the first two dimensions do not affect the output
    fmap = torch.zeros((1,10,fmap_h,fmap_w))
    anchors = multibox_prior(fmap, sizes=s, ratios=[1,2,0.5])
    bbox_scale = torch.tensor((w,h,w,h))
    show_bboxes(ricky.plt.imshow(img).axes, anchors[0]*bbox_scale)



"""Some test functions."""

def test_anchor():
    h, w = img.shape[:2]
    print(h,w)
    X = torch.rand(size=(1,3,h,w)) # Construct input data
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1,2,0.5])
    boxes = Y.reshape(h,w,5,4)
    print(boxes[250,250,0,:])

def example():
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                [1, 0.55, 0.2, 0.9, 0.88]])
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.92, 0.9]])
    h, w = img.shape[:2]
    bbox_scale = torch.tensor((w,h,w,h))
    fig = ricky.plt.imshow(img)
    show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
    show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

    labels = multibox_target(anchors.unsqueeze(dim=0),
                            ground_truth.unsqueeze(dim=0))
    print(labels[2])
    print(labels[1])
    print(labels[0])


def test_draw_anchor():
    h, w = img.shape[:2]
    bbox_scale = torch.tensor((w,h,w,h))
    fig = ricky.plt.imshow(img)
    X = torch.rand(size=(1,3,h,w)) # Construct input data
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1,2,0.5])
    boxes = Y.reshape(h,w,5,4)
    show_bboxes(fig.axes, boxes[250,250,:,:]*bbox_scale,
                ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75 r=2',
                's=0.75, r=0.5'])

    ricky.plt.show()


def test_bbox():
    dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
    boxes = torch.tensor((dog_bbox,cat_bbox))
    print(box_center_to_corner(box_corner_to_center(boxes)) == boxes)
    fig = ricky.plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
    fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
    ricky.plt.show()

torch.set_printoptions(2)




if __name__ == '__main__':
    folder = ricky.os.path.dirname(__file__)
    img = ricky.plt.imread(ricky.os.path.join(folder,'img/catdog.jpg'))
    h, w = img.shape[:2]
    #display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
    #ricky.plt.show()


'''
# Robot plot generation
def robot1():
    folder = os.path.dirname(os.path.dirname(__file__))
    img = ricky.plt.imread(ricky.os.path.join(folder,'robot_1.png'))
    h, w = img.shape[:2]
    display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
    ricky.plt.show()

folder = os.path.dirname(os.path.dirname(__file__))
img = ricky.plt.imread(ricky.os.path.join(folder,'robot_1.png'))
h, w = img.shape[:2]

bbox_scale = torch.tensor((w, h, w, h))
fig = ricky.plt.imshow(img)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.5, 0.33, 0.2], ratios=[1, 2, 0.5])
boxes = Y.reshape(h, w, 5, 4)
show_bboxes(fig.axes, boxes[435, 570, :, :] * bbox_scale,
            ['s=0.5, r=1', 's=0.33, r=1', 's=0.2, r=1', 's=0.5, r=2',
             's=0.5, r=0.5'])
ricky.plt.show()
'''




def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors*(num_classes+1), kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors*4, kernel_size=3, padding=1)

def forward(x, block):
    return block(x)

#Y1 = forward(torch.zeros((2,8,20,20)), cls_predictor(8,5,10))
#Y2 = forward(torch.zeros((2,16,10,10)), cls_predictor(16,3,10))
#print(Y1.shape, Y2.shape)

def flatten_pred(pred):
    return torch.flatten(pred.permute(0,2,3,1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


#print(concat_preds([Y1,Y2]).shape)

def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

#print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)

def base_net():
    blk = []
    num_filters = [3,16,32,64]
    for i in range(len(num_filters)-1):
        blk.append(down_sample_blk(num_filters[i],num_filters[i+1]))
    return nn.Sequential(*blk)

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64,128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128,128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = multibox_prior(Y, sizes=size, ratios=ratio)
    cls_pred = cls_predictor(Y)
    bbox_pred = bbox_predictor(Y)
    return (Y, anchors, cls_pred, bbox_pred)

#sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
#         [0.88, 0.961]]
#ratios = [[1, 2, 0.5]] * 5
#num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        idx_to_in_channels = [64,128,128,128,128]
        for i in range(5):
            # Equivalent to the assignment statement 'self.blk_i = get_blk(i)'
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],num_anchors))
        
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None]*5, [None]*5, [None]*5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

net = TinySSD(num_classes=1)
#X = torch.zeros((32,3,256,256))
#anchors, cls_preds, bbox_preds = net(X)
#print('output anchors:', anchors.shape)
#print('output class preds:', cls_preds.shape)
#print('output bbox preds:', bbox_preds.shape)

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size,-1).mean(dim=1)
    bbox = bbox_loss(bbox_preds*bbox_masks, bbox_labels*bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels-bbox_preds)*bbox_masks)).sum())



def main_train():
    batch_size = 32
    train_iter, _ = ricky.load_data_bananas(batch_size)
    device, net = ricky.try_gpu(), TinySSD(num_classes=1)

    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
    num_epochs, timer = 20, ricky.Timer()
    animator = ricky.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])
    net = net.to(device)

    for epoch in range(num_epochs):
        # Sum of training accuracy, no. of examples in sum of training accuracy,
        # Sum of absolute error, no. of examples in sum of absolute error
        metric = ricky.Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, Y)
            # Calculate the loss function using the predicted and labeled values
            # of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                    bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                    bbox_labels.numel())
            
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))
    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')



    def predict(X):
        net.eval()
        anchors, cls_preds, bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds, dim=2).permute(0,2,1)
        output = multibox_detection(cls_probs, bbox_preds, anchors)
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        return output[0, idx]

    def predict_non(X):
        net.eval()
        anchors, cls_preds, bbox_preds = net(X.to(device))
        cls_probs = F.softmax(cls_preds, dim=2).permute(0,2,1)
        output = multibox_detection(cls_probs, bbox_preds, anchors, nms_flag=False)
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        return output[0, idx]


    folder = ricky.os.path.dirname(__file__)
    X = torchvision.io.read_image(ricky.os.path.join(folder, 'img/77.png')).unsqueeze(0).float()
    img = X.squeeze(0).permute(1,2,0).long()
    output = predict(X)
    output_non = predict_non(X)

    def display(img, output, threshold):
        fig = ricky.plt.imshow(img)
        for row in output:
            score = float(row[1])
            if score < threshold:
                continue
            h, w = img.shape[:2]
            bbox = [row[2:6] * torch.tensor((w,h,w,h), device=row.device)]
            show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
    ricky.plt.close()
    display(img, output.cpu(), threshold=0.9)
    ricky.plt.show()
    display(img, output_non.cpu(), threshold=0.9)
    ricky.plt.show()

main_train()