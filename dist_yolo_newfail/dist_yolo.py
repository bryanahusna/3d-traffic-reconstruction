import copy
from PIL import Image
import tensorflow as tf
import keras
from scipy.special import expit, softmax
import numpy as np

path_prefix = './dist_yolo'

class DistYOLO():
    def __init__(self) -> None:
        self.custom_object_dict = get_custom_objects()
        self.model = keras.models.load_model(f'{path_prefix}/trained_final.h5', compile=False, custom_objects=self.custom_object_dict)
        self.anchors = get_anchors(f'{path_prefix}/configs/yolo3_anchors.txt')
        self.conf_threshold = 0.001
        self.elim_grid_sense = False
        self.class_names = get_classes(f'{path_prefix}/configs/kitty_all_except_nodata.txt')
        self.num_classes = len(self.class_names)
        self.model_image_size = (int(608), int(608))

    
    def detect(self, image):
        pred_boxes, pred_classes, pred_scores, pred_distances = self.yolo_predict_keras(image)
        return pred_boxes, pred_classes, pred_scores, pred_distances

    
    def yolo_predict_keras(self, image):
        image_data = self.preprocess_image(image)
        # origin image shape, in (height, width) format
        image_shape = tuple(reversed(image.size))

        predictions = self.model.predict([image_data])
        pred_boxes, pred_classes, pred_scores, pred_distances = self.yolo3_postprocess_np(predictions, image_shape, max_boxes=100)
        return pred_boxes, pred_classes, pred_scores, pred_distances

    def preprocess_image(self, image):
        """
        Prepare model input image data with letterbox
        resize, normalize and dim expansion

        # Arguments
            image: origin input image
                PIL Image object containing image data
            model_image_size: model input image size
                tuple of format (height, width).

        # Returns
            image_data: numpy array of image data for model input.
        """
        #resized_image = cv2.resize(image, tuple(reversed(model_image_size)), cv2.INTER_AREA)
        resized_image = letterbox_resize(image, tuple(reversed(self.model_image_size)))
        image_data = np.asarray(resized_image).astype('float32')
        image_data = normalize_image(image_data)
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        return image_data
    
    def yolo3_postprocess_np(self, yolo_outputs, image_shape, max_boxes=100, iou_threshold=0.4):
        predictions = self.yolo3_decode(yolo_outputs)
        predictions = self.yolo_correct_boxes(predictions, image_shape)

        boxes, classes, scores, distances = self.yolo_handle_predictions(
                                                        predictions,
                                                        image_shape,
                                                        max_boxes=max_boxes,
                                                        confidence=self.conf_threshold,
                                                        iou_threshold=iou_threshold)

        boxes = self.yolo_adjust_boxes(boxes, image_shape)

        return boxes, classes, scores, distances
    
    def yolo_correct_boxes(self, predictions, img_shape):
        '''rescale predicition boxes back to original image shape'''
        box_xy = predictions[..., :2]
        box_wh = predictions[..., 2:4]
        objectness = np.expand_dims(predictions[..., 4], -1)
        class_scores = predictions[..., 5:-1]
        dist = np.expand_dims(predictions[..., -1], -1)

        # model_image_size & image_shape should be (height, width) format
        model_image_size = np.array(self.model_image_size, dtype='float32')
        image_shape = np.array(img_shape, dtype='float32')
        height, width = image_shape

        new_shape = np.round(image_shape * np.min(model_image_size/image_shape))
        offset = (model_image_size-new_shape)/2./model_image_size
        scale = model_image_size/new_shape
        # reverse offset/scale to match (w,h) order
        offset = offset[..., ::-1]
        scale = scale[..., ::-1]

        box_xy = (box_xy - offset) * scale
        box_wh *= scale

        # Convert centoids to top left coordinates
        box_xy -= box_wh / 2

        # Scale boxes back to original image shape.
        image_wh = image_shape[..., ::-1]
        box_xy *= image_wh
        box_wh *= image_wh

        return np.concatenate([box_xy, box_wh, objectness, class_scores, dist], axis=2)
    
    def yolo_handle_predictions(self, predictions, image_shape, max_boxes=100, confidence=0.1, iou_threshold=0.4, use_cluster_nms=False, use_wbf=False):
        boxes = predictions[:, :, :4]
        box_confidences = np.expand_dims(predictions[:, :, 4], -1)
        box_class_probs = predictions[:, :, 5:-1]
        box_distances = np.expand_dims(predictions[:, :, -1], -1)

        # filter boxes with confidence threshold
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= confidence)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]
        distances = box_distances[pos]

        if use_cluster_nms:
            # use Fast/Cluster NMS for boxes postprocess
            n_boxes, n_classes, n_scores = fast_cluster_nms_boxes(boxes, classes, scores, iou_threshold, confidence=confidence)
        elif use_wbf:
            # use Weighted-Boxes-Fusion for boxes postprocess
            n_boxes, n_classes, n_scores = weighted_boxes_fusion([boxes], [classes], [scores], image_shape, weights=None, iou_thr=iou_threshold)
        else:
            # Boxes, Classes and Scores returned from NMS
            n_boxes, n_classes, n_scores, n_distances = nms_boxes(boxes, classes, scores, distances, iou_threshold, confidence=confidence)

        if n_boxes:
            boxes = np.concatenate(n_boxes)
            classes = np.concatenate(n_classes).astype('int32')
            scores = np.concatenate(n_scores)
            distances = np.concatenate(n_distances)
            boxes, classes, scores, distances = filter_boxes(boxes, classes, scores, distances, max_boxes)

            return boxes, classes, scores, distances

        else:
            return [], [], []
    
    def yolo_adjust_boxes(self, boxes, img_shape):
        '''
        change box format from (x,y,w,h) top left coordinate to
        (xmin,ymin,xmax,ymax) format
        '''
        if boxes is None or len(boxes) == 0:
            return []

        image_shape = np.array(img_shape, dtype='float32')
        height, width = image_shape

        adjusted_boxes = []
        for box in boxes:
            x, y, w, h = box

            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h

            ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
            xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
            ymax = min(height, np.floor(ymax + 0.5).astype('int32'))
            xmax = min(width, np.floor(xmax + 0.5).astype('int32'))
            adjusted_boxes.append([xmin,ymin,xmax,ymax])

        return np.array(adjusted_boxes,dtype=np.int32)
    
    def yolo3_decode(self, predictions):
        """
        YOLOv3 Head to process predictions from YOLOv3 models

        :param num_classes: Total number of classes
        :param anchors: YOLO style anchor list for bounding box assignment
        :param input_dims: Input dimensions of the image
        :param predictions: A list of three tensors with shape (N, 19, 19, 255), (N, 38, 38, 255) and (N, 76, 76, 255)
        :return: A tensor with the shape (N, num_boxes, 85)
        """
        assert len(predictions) == len(self.anchors)//3, 'anchor numbers does not match prediction.'

        if len(predictions) == 3: # assume 3 set of predictions is YOLOv3
            anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
            scale_x_y = [1.05, 1.1, 1.2] if self.elim_grid_sense else [None, None, None]
        elif len(predictions) == 2: # 2 set of predictions is YOLOv3-tiny
            anchor_mask = [[3,4,5], [0,1,2]]
            scale_x_y = [1.05, 1.05] if self.elim_grid_sense else [None, None]
        else:
            raise ValueError('Unsupported prediction length: {}'.format(len(predictions)))

        results = []
        for i, prediction in enumerate(predictions):
            results.append(self.yolo_decode(prediction, self.anchors[anchor_mask[i]], scale_x_y=scale_x_y[i], use_softmax=False))

        return np.concatenate(results, axis=1)
    
    def yolo_decode(self, prediction, anchors, scale_x_y=None, use_softmax=False):
        '''Decode final layer features to bounding box parameters.'''
        batch_size = np.shape(prediction)[0]
        num_anchors = len(anchors)

        grid_size = np.shape(prediction)[1:3]
        #check if stride on height & width are same
        assert self.model_image_size[0]//grid_size[0] == self.model_image_size[1]//grid_size[1], 'model stride mismatch.'
        stride = self.model_image_size[0] // grid_size[0]

        prediction = np.reshape(prediction,
                                (batch_size, grid_size[0] * grid_size[1] * num_anchors, self.num_classes + 5 + 1))

        ################################
        # generate x_y_offset grid map
        grid_y = np.arange(grid_size[0])
        grid_x = np.arange(grid_size[1])
        x_offset, y_offset = np.meshgrid(grid_x, grid_y)

        x_offset = np.reshape(x_offset, (-1, 1))
        y_offset = np.reshape(y_offset, (-1, 1))

        x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
        x_y_offset = np.tile(x_y_offset, (1, num_anchors))
        x_y_offset = np.reshape(x_y_offset, (-1, 2))
        x_y_offset = np.expand_dims(x_y_offset, 0)

        ################################

        # Log space transform of the height and width
        anchors = np.tile(anchors, (grid_size[0] * grid_size[1], 1))
        anchors = np.expand_dims(anchors, 0)

        if scale_x_y:
            # Eliminate grid sensitivity trick involved in YOLOv4
            #
            # Reference Paper & code:
            #     "YOLOv4: Optimal Speed and Accuracy of Object Detection"
            #     https://arxiv.org/abs/2004.10934
            #     https://github.com/opencv/opencv/issues/17148
            #
            box_xy_tmp = expit(prediction[..., :2]) * scale_x_y - (scale_x_y - 1) / 2
            box_xy = (box_xy_tmp + x_y_offset) / np.array(grid_size)[::-1]
        else:
            box_xy = (expit(prediction[..., :2]) + x_y_offset) / np.array(grid_size)[::-1]
        box_wh = (np.exp(prediction[..., 2:4]) * anchors) / np.array(self.model_image_size)[::-1]

        # Sigmoid objectness scores
        objectness = expit(prediction[..., 4])  # p_o (objectness score)
        objectness = np.expand_dims(objectness, -1)  # To make the same number of values for axis 0 and 1

        if use_softmax:
            # Softmax class scores
            class_scores = softmax(prediction[..., 5:5+self.num_classes], axis=-1)
        else:
            # Sigmoid class scores
            class_scores = expit(prediction[..., 5:5+self.num_classes])

        dist = prediction[..., -1]
        dist = np.expand_dims(dist, -1)

        return np.concatenate([box_xy, box_wh, objectness, class_scores, dist], axis=2)

    
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def letterbox_resize(image, target_size, return_padding_info=False):
    """
    Resize image with unchanged aspect ratio using padding

    # Arguments
        image: origin image to be resize
            PIL Image object containing image data
        target_size: target image size,
            tuple of format (width, height).
        return_padding_info: whether to return padding size & offset info
            Boolean flag to control return value

    # Returns
        new_image: resized PIL Image object.

        padding_size: padding image size (keep aspect ratio).
            will be used to reshape the ground truth bounding box
        offset: top-left offset in target image padding.
            will be used to reshape the ground truth bounding box
    """
    src_w, src_h = image.size
    target_w, target_h = target_size

    # calculate padding scale and padding offset
    scale = min(target_w/src_w, target_h/src_h)
    padding_w = int(src_w * scale)
    padding_h = int(src_h * scale)
    padding_size = (padding_w, padding_h)

    dx = (target_w - padding_w)//2
    dy = (target_h - padding_h)//2
    offset = (dx, dy)

    # create letterbox resized image
    image = image.resize(padding_size, Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128,128,128))
    new_image.paste(image, offset)

    if return_padding_info:
        return new_image, padding_size, offset
    else:
        return new_image

def normalize_image(image):
    """
    normalize image array from 0 ~ 255
    to 0.0 ~ 1.0

    # Arguments
        image: origin input image
            numpy image array with dtype=float, 0.0 ~ 255.0

    # Returns
        image: numpy image array with dtype=float, 0.0 ~ 1.0
    """
    image = image / 255.0

    return image

def nms_boxes(boxes, classes, scores, distances, iou_threshold, confidence=0.1, use_diou=True, is_soft=False, use_exp=False, sigma=0.5):
    nboxes, nclasses, nscores, ndistances = [], [], [], []
    for c in set(classes):
        # handle data for one class
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        d = distances[inds]

        # make a data copy to avoid breaking
        # during nms operation
        b_nms = copy.deepcopy(b)
        c_nms = copy.deepcopy(c)
        s_nms = copy.deepcopy(s)
        d_nms = copy.deepcopy(d)

        while len(s_nms) > 0:
            # pick the max box and store, here
            # we also use copy to persist result
            i = np.argmax(s_nms, axis=-1)
            nboxes.append(copy.deepcopy(b_nms[i]))
            nclasses.append(copy.deepcopy(c_nms[i]))
            nscores.append(copy.deepcopy(s_nms[i]))
            ndistances.append(copy.deepcopy(d_nms[i]))

            # swap the max line and first line
            b_nms[[i,0],:] = b_nms[[0,i],:]
            c_nms[[i,0]] = c_nms[[0,i]]
            s_nms[[i,0]] = s_nms[[0,i]]
            d_nms[[i, 0]] = d_nms[[0, i]]

            if use_diou:
                iou = box_diou(b_nms)
                #iou = box_diou_matrix(b_nms, b_nms)[0][1:]
            else:
                iou = box_iou(b_nms)
                #iou = box_iou_matrix(b_nms, b_nms)[0][1:]

            # drop the last line since it has been record
            b_nms = b_nms[1:]
            c_nms = c_nms[1:]
            s_nms = s_nms[1:]
            d_nms = d_nms[1:]

            if is_soft:
                # Soft-NMS
                if use_exp:
                    # score refresh formula:
                    # score = score * exp(-(iou^2)/sigma)
                    s_nms = s_nms * np.exp(-(iou * iou) / sigma)
                else:
                    # score refresh formula:
                    # score = score * (1 - iou) if iou > threshold
                    depress_mask = np.where(iou > iou_threshold)[0]
                    s_nms[depress_mask] = s_nms[depress_mask]*(1-iou[depress_mask])
                keep_mask = np.where(s_nms >= confidence)[0]
            else:
                # normal Hard-NMS
                keep_mask = np.where(iou <= iou_threshold)[0]

            # keep needed box for next loop
            b_nms = b_nms[keep_mask]
            c_nms = c_nms[keep_mask]
            s_nms = s_nms[keep_mask]
            d_nms = d_nms[keep_mask]

    # reformat result for output
    nboxes = [np.array(nboxes)]
    nclasses = [np.array(nclasses)]
    nscores = [np.array(nscores)]
    ndistances = [np.array(ndistances)]
    return nboxes, nclasses, nscores, ndistances

def filter_boxes(boxes, classes, scores, distances, max_boxes):
    '''
    Sort the prediction boxes according to score
    and only pick top "max_boxes" ones
    '''
    # sort result according to scores
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    nboxes = boxes[sorted_indices]
    nclasses = classes[sorted_indices]
    nscores = scores[sorted_indices]
    ndistances = distances[sorted_indices]

    # only pick max_boxes
    nboxes = nboxes[:max_boxes]
    nclasses = nclasses[:max_boxes]
    nscores = nscores[:max_boxes]
    ndistances = ndistances[:max_boxes]

    return nboxes, nclasses, nscores, ndistances

def box_iou(boxes):
    """
    Calculate IoU value of 1st box with other boxes of a box array

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates

    Returns
    -------
    iou: numpy array, shape=(N-1,)
         IoU value of boxes[1:] with boxes[0]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[1:], x[0])
    inter_ymin = np.maximum(y[1:], y[0])
    inter_xmax = np.minimum(x[1:] + w[1:], x[0] + w[0])
    inter_ymax = np.minimum(y[1:] + h[1:], y[0] + h[0])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[1:] + areas[0] - inter)
    return iou


def box_diou(boxes):
    """
    Calculate DIoU value of 1st box with other boxes of a box array
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates

    Returns
    -------
    diou: numpy array, shape=(N-1,)
         IoU value of boxes[1:] with boxes[0]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[1:], x[0])
    inter_ymin = np.maximum(y[1:], y[0])
    inter_xmax = np.minimum(x[1:] + w[1:], x[0] + w[0])
    inter_ymax = np.minimum(y[1:] + h[1:], y[0] + h[0])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[1:] + areas[0] - inter)

    # box center distance
    x_center = x + w/2
    y_center = y + h/2
    center_distance = np.power(x_center[1:] - x_center[0], 2) + np.power(y_center[1:] - y_center[0], 2)

    # get enclosed area
    enclose_xmin = np.minimum(x[1:], x[0])
    enclose_ymin = np.minimum(y[1:], y[0])
    enclose_xmax = np.maximum(x[1:] + w[1:], x[0] + w[0])
    enclose_ymax = np.maximum(x[1:] + w[1:], x[0] + w[0])
    enclose_w = np.maximum(0.0, enclose_xmax - enclose_xmin + 1)
    enclose_h = np.maximum(0.0, enclose_ymax - enclose_ymin + 1)
    # get enclosed diagonal distance
    enclose_diagonal = np.power(enclose_w, 2) + np.power(enclose_h, 2)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + np.finfo(float).eps)

    return diou

def get_custom_objects():
    '''
    form up a custom_objects dict so that the customized
    layer/function call could be correctly parsed when keras
    .h5 model is loading or converting
    '''
    custom_objects_dict = {
        'tf': tf,
        'swish': swish,
        'hard_sigmoid': hard_sigmoid,
        'hard_swish': hard_swish,
        'mish': mish
    }

    return custom_objects_dict

def swish(x):
    """Swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The Swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if keras.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return tf.nn.swish(x)
        except AttributeError:
            pass

    return x * keras.backend.sigmoid(x)

def hard_sigmoid(x):
    return keras.layers.ReLU(6.)(x + 3.) * (1. / 6.)

def hard_swish(x):
    return keras.layers.Multiply()([keras.layers.Activation(hard_sigmoid)(x), x])

def mish(x):
    return x * keras.backend.tanh(keras.backend.softplus(x))
