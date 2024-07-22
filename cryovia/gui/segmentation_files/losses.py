


def seg_loss(relative_weights=[1.0, 1.0, 5.0], classes=3,use_distance_weights=False):
    import tensorflow as tf
    from tensorflow.nn import softmax_cross_entropy_with_logits as cross_entropy
    import tensorflow.keras.backend as K
    class_weights = tf.constant([relative_weights])

    def seg_loss(y_true, y_pred):
        
        extra = 0
        if use_distance_weights:
            extra += 1
        # print("**************************************************************************")
        # print(classes, relative_weights, y_true.shape, )
        splitted_true = tf.split(y_true, classes + extra, axis=-1 )
        if use_distance_weights:
            true_classes = splitted_true[:-1]
            distance_weights = splitted_true[-1]
            
        else:
            
            true_classes = splitted_true
            distance_weights = tf.ones_like(true_classes[0], dtype=tf.float32)
        
        pred_classes = tf.split(y_pred, classes, axis=-1)
        
        # true_classes = tf.cast(true_classes,tf.float32)
        # distance_weights = tf.cast(distance_weights, tf.float32)
        

        onehot_gt = tf.reshape(tf.stack(true_classes, axis=3), [-1, classes])
        weighted_gt = tf.reduce_sum(class_weights * onehot_gt, axis=1)
        
        distance_weights = tf.reshape(distance_weights,[-1])

        weighted_gt *= distance_weights
               

        onehot_pred = tf.reshape(tf.stack(pred_classes, axis=-1), [-1, classes])

        segmentation_loss = K.mean(
            tf.reduce_sum(onehot_gt, axis=-1) * (cross_entropy(logits=onehot_pred, labels=onehot_gt) * weighted_gt)
        )

        return segmentation_loss

    return seg_loss




def segmentationLoss(relative_weights=[1.0, 1.0, 5.0], classes=3):
    import tensorflow as tf
    from tensorflow.nn import softmax_cross_entropy_with_logits as cross_entropy
    import tensorflow.keras.backend as K
    class_weights = tf.constant([relative_weights])

    def seg_loss(y_true, y_pred):


        # print("**************************************************************************")
        # print(classes, relative_weights, y_true.shape, )
        splitted_true = tf.split(y_true, classes, axis=-1 )

        
        true_classes = splitted_true
        
        pred_classes = tf.split(y_pred, classes, axis=-1)
        
        # true_classes = tf.cast(true_classes,tf.float32)
        # distance_weights = tf.cast(distance_weights, tf.float32)
        

        onehot_gt = tf.reshape(tf.stack(true_classes, axis=3), [-1, classes])
        weighted_gt = tf.reduce_sum(class_weights * onehot_gt, axis=1)
        
        # distance_weights = tf.reshape(distance_weights,[-1])

        # weighted_gt *= distance_weights
               

        onehot_pred = tf.reshape(tf.stack(pred_classes, axis=-1), [-1, classes])

        segmentation_loss = K.mean(
            tf.reduce_sum(onehot_gt, axis=-1) * (cross_entropy(logits=onehot_pred, labels=onehot_gt) * weighted_gt)
        )

        return segmentation_loss

    return seg_loss