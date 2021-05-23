#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <time.h>

#define Print(str) cout << #str << "=" << str << endl
using namespace std;

torch::Tensor iou(const torch::Tensor box, const torch::Tensor boxes, const bool DIoU=false);

torch::Tensor nms(const torch::Tensor boxes, torch::Tensor scores, const float thresh = 0.3, const bool DIoU=false);

torch::Tensor xywh2xyxy(torch::Tensor boxes);

torch::Tensor xyxy2xywh(torch::Tensor boxes);

int non_max_suppression(std::vector<torch::Tensor> &output, torch::Tensor &xs, float conf_thresh = 0.1, float iou_thresh = 0.45, const bool DIoU=false);

#endif // __UTILS_H__