import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torchvision
from torchvision.transforms import v2
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import gc

def plot_vectors(vectors, figure, title):
    for i in range(len(vectors)):
        if(type(vectors)==torch.Tensor):
            kp = vectors[i].detach().cpu().numpy().copy()
        else:
            kp = vectors[i].copy()
        xs = [0,kp[0]]
        ys = [0,kp[1]]
        figure.plot(xs,ys)
    figure.set_title(title)
    
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def cosine_similarity(pose1, pose2):
    if(type(pose1) == torch.Tensor):
        pose1 = pose1.detach().cpu().numpy()
    if(type(pose2) == torch.Tensor):
        pose2 = pose2.detach().cpu().numpy()
    return np.diagonal(cos_sim(pose1, pose2))


def weight_distance(pose1, pose2, conf1):
    # D(U,V) = (1 / sum(conf1)) * sum(conf1 * ||pose1 - pose2||) = sum1 * sum2
    if type(conf1)==torch.Tensor:
        conf1 = conf1.detach().cpu().numpy()[0].copy()
    else:
        conf1 = conf1[0].copy()
    sum1 = 1 / np.sum(conf1)
    sum2 = 0

    for i in range(len(pose1)):
        # каждый индекс i имеет x и y, у которых одинаковая оценка достоверности
        conf_ind = math.floor(i / 2)
        sum2 += conf1[conf_ind] * abs(pose1[i] - pose2[i])

    weighted_dist = sum1 * sum2
    del conf1
    del conf2
    return weighted_dist

class pose_model:
    keypoints_name = ['nose','left_eye','right_eye',\
        'left_ear','right_ear','left_shoulder',\
        'right_shoulder','left_elbow','right_elbow',\
        'left_wrist','right_wrist','left_hip',\
        'right_hip','left_knee', 'right_knee', \
        'left_ankle','right_ankle']

    skeleton = [
        [keypoints_name.index("right_eye"), keypoints_name.index("nose")],
        [keypoints_name.index("right_eye"), keypoints_name.index("right_ear")],
        [keypoints_name.index("left_eye"), keypoints_name.index("nose")],
        [keypoints_name.index("left_eye"), keypoints_name.index("left_ear")],
        [keypoints_name.index("right_shoulder"), keypoints_name.index("right_elbow")],
        [keypoints_name.index("right_elbow"), keypoints_name.index("right_wrist")],
        [keypoints_name.index("left_shoulder"), keypoints_name.index("left_elbow")],
        [keypoints_name.index("left_elbow"), keypoints_name.index("left_wrist")],
        [keypoints_name.index("right_hip"), keypoints_name.index("right_knee")],
        [keypoints_name.index("right_knee"), keypoints_name.index("right_ankle")],
        [keypoints_name.index("left_hip"), keypoints_name.index("left_knee")],
        [keypoints_name.index("left_knee"), keypoints_name.index("left_ankle")],
        [keypoints_name.index("right_shoulder"), keypoints_name.index("left_shoulder")],
        [keypoints_name.index("right_hip"), keypoints_name.index("left_hip")],
        [keypoints_name.index("right_shoulder"), keypoints_name.index("right_hip")],
        [keypoints_name.index("left_shoulder"), keypoints_name.index("left_hip")],
    ]

    body_points = [
        keypoints_name.index("right_shoulder"),
        keypoints_name.index("left_shoulder"),
        keypoints_name.index("right_hip"),
        keypoints_name.index("left_hip")
    ]
    head_points = [
        keypoints_name.index("right_eye"),
        keypoints_name.index("left_eye"),
        keypoints_name.index("nose"),
        keypoints_name.index("right_ear"),
        keypoints_name.index("left_ear"),
    ]
    limbs_vectors = [
        [keypoints_name.index("right_shoulder"), keypoints_name.index("right_elbow")],
        [keypoints_name.index("right_elbow"), keypoints_name.index("right_wrist")],
        [keypoints_name.index("left_shoulder"), keypoints_name.index("left_elbow")],
        [keypoints_name.index("left_elbow"), keypoints_name.index("left_wrist")],
        [keypoints_name.index("right_hip"), keypoints_name.index("right_knee")],
        [keypoints_name.index("right_knee"), keypoints_name.index("right_ankle")],
        [keypoints_name.index("left_hip"), keypoints_name.index("left_knee")],
        [keypoints_name.index("left_knee"), keypoints_name.index("left_ankle")],
    ]

    def __init__(self, pose_image, body_coef = 10, head_coef = 5, body_height = 500):
        self.head_coef = head_coef
        self.body_coef = body_coef
        self.body_height = body_height
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.transform = v2.Compose([torchvision.transforms.ToTensor()])
        self.image = self.transform(pose_image)
        self.model.eval()
        self.model_output = self.model(self.image.unsqueeze(0))[0]
        self.cmap = plt.get_cmap("rainbow")
        self.model_keypoints, self.model_scores = self.get_output_for_person(self.model_output)
        self.add_colors_to_skeleton()
        self.transformed_keypoints = self.transorm_pose()
        self.limbs_vectors_by_start_points = self.create_limbs_vectors_by_start_points()


    def get_output_for_person(self, input):
        person_index = torch.argmax(input['scores'])
        return input['keypoints'][person_index, ...][...,:-1], input['keypoints_scores'][person_index, ...]

    def add_colors_to_skeleton(self):
        num_keypoints = len(self.skeleton)
        colour_step=255 // num_keypoints # Зададим шаг цвета
        for i in range(num_keypoints):
            colour = self.cmap(i * colour_step / 255)
            self.skeleton[i].append(colour)

    def draw_coloured_skeleton(self, keypoint_threshold=2, line_thickness=5):
        if(type(self.image) == torch.Tensor):
          img = self.image.permute(1, 2, 0).numpy()
        # создаём копию изображений
        img_copy = img.copy()
        color = (0, 255, 0)
        keypoints_restored = []
        for kp in range(len(self.model_scores)):
            # проверяем степень уверенности детектора опорной точки
            if self.model_scores[kp] > keypoint_threshold:
                # конвертируем массив ключевых точек в список целых чисел
                keypoint = tuple(
                    map(int, self.model_keypoints[kp].detach().numpy().tolist())
                )
                keypoints_restored.append(keypoint)
                # рисуем кружок радиуса 5 вокруг точки
                cv2.circle(img_copy, keypoint, 5, color, -1)
        for limb in self.skeleton:
          first = limb[0]
          second = limb[1]
          if (self.model_scores[first] and self.model_scores[second])>keypoint_threshold:
            cv2.line(img_copy, keypoints_restored[first], keypoints_restored[second], limb[2], line_thickness)
        return img_copy

    def transpose_keypoints(self, keypoints):
        if type(keypoints)==torch.Tensor:
            kps = keypoints.detach().numpy().copy()
        else:
            kps = keypoints.copy()

        # Базовые точки туловища
        left_hip = kps[self.keypoints_name.index('left_hip')]
        right_hip = kps[self.keypoints_name.index('right_hip')]
        left_shoulder = kps[self.keypoints_name.index('left_shoulder')]
        right_shoulder = kps[self.keypoints_name.index('right_shoulder')]
        body_center = np.array(line_intersection([left_hip, right_shoulder], [right_hip, left_shoulder]))
        result = []
        for kp in kps:
            result.append(kp - body_center)
        return result

    def rotate_keypoints(self, keypoints):
        if type(keypoints)==torch.Tensor:
            kps = keypoints.detach().numpy()
        else:
            kps = keypoints

        # Базовые точки туловища
        left_hip = kps[self.keypoints_name.index('left_hip')]
        right_hip = kps[self.keypoints_name.index('right_hip')]
        left_shoulder = kps[self.keypoints_name.index('left_shoulder')]
        right_shoulder = kps[self.keypoints_name.index('right_shoulder')]
        inter_hips_dot = right_hip + (left_hip - right_hip)/2
        inter_shoulders_dot = right_shoulder + (left_shoulder - right_shoulder)/2
        self.body_vector = inter_hips_dot - inter_shoulders_dot
        magnitude = np.linalg.norm(self.body_vector)
        sin = self.body_vector[0] / magnitude
        cos = self.body_vector[1] / magnitude
        M = np.array(((cos, -sin), (sin, cos)))
        return np.dot(kps, M.T)

    def scale_keypoints(self, keypoints):
        if type(keypoints)==torch.Tensor:
            kps = keypoints.detach().numpy()
        else:
            kps = keypoints
        scale = np.linalg.norm(self.body_vector / self.body_height)
        M = np.array(((1 / scale, 0), (0, 1 / scale)))
        return np.dot(kps, M.T)

    def transorm_pose(self):
        transformed = self.transpose_keypoints(self.model_keypoints)
        transformed = self.rotate_keypoints(transformed)
        return self.scale_keypoints(transformed)

    def create_limbs_vectors_by_start_points(self):
        result = np.empty((len(self.limbs_vectors), 2))
        for i, limb in enumerate(self.limbs_vectors):
            result[i] = self.transformed_keypoints[limb[1]] - self.transformed_keypoints[limb[0]]
        return result

    def body_parts_cosin_score(self, pose_to_compare, body_coef=10, head_coef=5):
        pose1 = self.transformed_keypoints
        pose2 = pose_to_compare.transformed_keypoints
        if type(pose1)==torch.Tensor:
            pose1 = pose1.detach().numpy()
        if type(pose2)==torch.Tensor:
            pose2 = pose2.detach().numpy()
        # Поскольку в результате преобразования позиция туловища оказывается фиксированной,
        # то его вектора будут менятся крайне незначительно.
        # И чтобы придать этим изменениям больший вес возведем значения косинусной близости этих точек в степень
        body_score = cosine_similarity(pose1[self.body_points], pose2[self.body_points])**body_coef
        # Вектора головы также будут менятся незначительно тк размах ее движений не очень вилик
        # Поэтому тоже возведем их косинусную близость в степень
        head_score = cosine_similarity(pose1[self.head_points], pose2[self.head_points])**head_coef
        # для финального подсчета голову и туловище будем учитывать как отдельные точки целеком
        limbs_score = cosine_similarity(self.limbs_vectors_by_start_points, pose_to_compare.limbs_vectors_by_start_points)
        scores = [np.mean(body_score), np.mean(head_score)]
        scores.extend(limbs_score)
        return np.mean(scores), limbs_score, body_score, head_score

    def plot_skeleton(self,  figure, title):
        for limb in self.skeleton:
            first = limb[0]
            second = limb[1]
            x = [self.transformed_keypoints[first][0], self.transformed_keypoints[second][0]]
            y = [self.transformed_keypoints[first][1], self.transformed_keypoints[second][1]]
            figure.set_title(title)
            plot = figure.plot([x[0], x[1]], [y[0], y[1]], linewidth = 3)
        figure.axes.invert_yaxis()
