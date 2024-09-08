# --------------------------------------------------------
# Visual Instruction Tuning
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# LISA Questions and GSVA questions

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

SHORT_QUESTION_LIST_MODE4 = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What are {class_name} in this image? Please respond with segmentation masks.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What are {class_name} in this image? Please output segmentation masks."
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explaination.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

ANSWER_LIST_MODE1 = [
    "Here it is.",
    "Sure.",
    "Sure, this is the target.",
    "Sure, here is the segmentation result.",
    "Here you are."
]

ANSWER_LIST_MODE4_START = [
    "The segmentation results are",
    "Sure, they are",
    "Sure,",
    "Sure,",
    "Sure,"
]

ANSWER_LIST_MODE4_TEMPLATE = [
    "{class_name} [SEG]",
    "{class_name}:[SEG]",
    "the mask of {class_name} is [SEG]",
    "the segmentation of {class_name} is [SEG]",
    "the referred {class_name} is [SEG]"
]

ANSWER_LIST_MODE4_END = [
    ".", ".", ".", ".", "."
]