# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    # Basic configurations of the executor
    name='Hand Pose Estimation',
    camera_id=0,
    camera_max_fps=30,
    # Define nodes.
    # The configuration of a node usually includes:
    #   1. 'type': Node class name
    #   2. 'name': Node name
    #   3. I/O buffers (e.g. 'input_buffer', 'output_buffer'): specify the
    #       input and output buffer names. This may depend on the node class.
    #   4. 'enable_key': assign a hot-key to toggle enable/disable this node.
    #       This may depend on the node class.
    #   5. Other class-specific arguments
    nodes=[
        # 'DetectorNode':
        # This node performs object detection from the frame image using an
        # MMDetection model.
        dict(
            type='DetectorNode',
            name='detector',
            model_config='demo/mmdetection_cfg/'
            'ssdlite_mobilenetv2_scratch_600e_onehand.py',
            model_checkpoint='https://download.openmmlab.com'
            '/mmpose/mmdet_pretrained/'
            'ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth',
            input_buffer='_input_',  # `_input_` is an executor-reserved buffer
            output_buffer='det_result'),
        # 'TopDownPoseEstimatorNode':
        # This node performs keypoint detection from the frame image using an
        # MMPose top-down model. Detection results is needed.
        dict(
            type='TopDownPoseEstimatorNode',
            name='hand pose estimator',
            model_config='configs/hand/2d_kpt_sview_rgb_img/'
            'topdown_heatmap/onehand10k/'
            'res50_onehand10k_256x256.py',
            model_checkpoint='https://download.openmmlab.com/mmpose/top_down/'
            'resnet/res50_onehand10k_256x256-e67998f6_20200813.pth',
            labels=['hand'],
            smooth=True,
            input_buffer='det_result',
            output_buffer='hand_pose'),
        # 'ObjectAssignerNode':
        # This node binds the latest model inference result with the current
        # frame. (This means the frame image and inference result may be
        # asynchronous).
        dict(
            type='ObjectAssignerNode',
            name='object assigner',
            frame_buffer='_frame_',  # `_frame_` is an executor-reserved buffer
            object_buffer='hand_pose',
            output_buffer='frame'),
        # 'ObjectVisualizerNode':
        # This node draw the pose visualization result in the frame image.
        # Pose results is needed.
        dict(
            type='ObjectVisualizerNode',
            name='object visualizer',
            enable_key='v',
            enable=True,
            show_bbox=True,
            must_have_keypoint=False,
            show_keypoint=True,
            input_buffer='frame',
            output_buffer='vis'),
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='instruction',
            enable_key='h',
            enable=False,
            input_buffer='vis',
            output_buffer='vis_notice',
            content_lines=[
                'This is a demo for hand pose visualization',
                '', 'Hot-keys:',
                '"v": Pose estimation result visualization',
                '"h": Show help information',
                '"m": Show diagnostic information', '"q": Exit'
            ],
        ),
        # 'MonitorNode':
        # This node show diagnostic information in the frame image. It can
        # be used for debugging or monitoring system resource status.
        dict(
            type='MonitorNode',
            name='monitor',
            enable_key='m',
            enable=False,
            input_buffer='vis_notice',
            output_buffer='_display_'),
    ])
