#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#define UTILS_EXPORTS
#include "utils.h"
// #include <opencv2/opencv.hpp>
// #include <torch/script.h>
// #include <torch/torch.h>
// #include <algorithm>
// #include <iostream>
// #include <time.h>
using namespace std;
/*
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
*/
torch::Tensor iou(const torch::Tensor box, const torch::Tensor boxes, const bool DIoU)
{
    // Print(DIoU);
    float eps = 1e-12;
    auto box_x1 = box.select(0, 0);
    auto box_y1 = box.select(0, 1);
    auto box_x2 = box.select(0, 2);
    auto box_y2 = box.select(0, 3);

    auto boxes_x1 = boxes.select(1, 0);
    auto boxes_y1 = boxes.select(1, 1);
    auto boxes_x2 = boxes.select(1, 2);
    auto boxes_y2 = boxes.select(1, 3);

    auto box_area = (box_x2 - box_x1) * (box_y2 - box_y1);
    ;
    auto boxes_area = (boxes_x2 - boxes_x1) * (boxes_y2 - boxes_y1);

    auto x1 = torch::max(box_x1, boxes_x1);
    auto y1 = torch::max(box_y1, boxes_y1);
    auto x2 = torch::min(box_x2, boxes_x2);
    auto y2 = torch::min(box_y2, boxes_y2);

    auto w = torch::clamp(x2 - x1, 0);
    auto h = torch::clamp(y2 - y1, 0);

    auto inter = w * h;

    auto iou_ = inter / (box_area + boxes_area - inter);
    if (DIoU)
    {
        auto cx = torch::max(box_x2, boxes_x2) - torch::min(box_x1, boxes_x1);
        auto cy = torch::max(box_y2, boxes_y2) - torch::min(box_y1, boxes_y1);
        auto c2 = torch::pow(cx, 2) + torch::pow(cy, 2) + eps;
        auto rho2 = (torch::pow((boxes_x1+boxes_x2-box_x1-box_x2),2) + torch::pow((boxes_y1+boxes_y2-box_y1-box_y2),2))/4;
        iou_ -= rho2/c2;
    }

    return iou_;
}


torch::Tensor nms(const torch::Tensor boxes, torch::Tensor scores, const float thresh, const bool DIoU)
{
    // Print(DIoU);
    vector<torch::Tensor> keep_boxes;
    vector<torch::Tensor> keep_indices;
    if (boxes.size(0) == 0)
        return torch::empty({0, 0});

    auto args = scores.argsort(-1, true);
    auto sort_boxes = boxes.index_select(0, args);

    torch::Tensor box, arg, idx;
    while (sort_boxes.size(0) > 0)
    {

        box = sort_boxes[0];
        arg = args[0];
        keep_boxes.push_back(box);
        keep_indices.push_back(arg);
        if (sort_boxes.size(0) > 1)
        {
            sort_boxes = sort_boxes.slice(0, 1);
            args = args.slice(0, 1);
            idx = (iou(box, sort_boxes, DIoU).le(thresh)).nonzero().select(1, 0);
            sort_boxes = sort_boxes.index_select(0, idx);
            args = args.index_select(0, idx);
        }
        else
        {
            break;
        }
    }
    auto rst_id = torch::stack(keep_indices);
    return rst_id;
}

torch::Tensor xywh2xyxy(torch::Tensor boxes)
{
    auto cx = boxes.select(1, 0);
    auto cy = boxes.select(1, 1);
    auto w = boxes.select(1, 2);
    auto h = boxes.select(1, 3);
    // auto cf = boxes.select(1, 0);

    auto half_w = w / 2;
    auto half_h = h / 2;

    auto x1 = cx - half_w;
    auto y1 = cy - half_h;
    auto x2 = cx + half_w;
    auto y2 = cy + half_h;

    auto rst = torch::stack({x1, y1, x2, y2}, 1);

    return rst;
}

torch::Tensor xyxy2xywh(torch::Tensor boxes)
{
    auto x1 = boxes.select(1, 0);
    auto y1 = boxes.select(1, 1);
    auto x2 = boxes.select(1, 2);
    auto y2 = boxes.select(1, 3);

    auto cx = (x1 + x2) / 2;
    auto cy = (y1 + y2) / 2;
    auto w = x2 - x1;
    auto h = y2 - y1;

    auto rst = torch::stack({cx, cy, w, h}, 1);

    return rst;
}

int non_max_suppression(std::vector<torch::Tensor> &output, torch::Tensor &xs, float conf_thresh, float iou_thresh, const bool DIoU)
{

    // Print(DIoU);
    // std::vector<torch::Tensor> output;
    // xs shape [1, 25200, 85]
    int max_nms = 30000;
    int min_wh = 2;
    int max_wh = 4096;
    int max_det = 300;
    int nc = xs.size(2) - 5; // 80 cls
    auto xc = xs.select(2, 4) > conf_thresh;
    auto xs_ = xs[0].slice(0, 0, 10).slice(1, 0, 4);

    for (size_t xi = 0; xi < xs.size(0); xi++)
    {

        torch::Tensor x = xs[xi].index_select(0, xc[xi].nonzero().select(1, 0));
        auto x_ = x.slice(1, 0, 4);

        // # Compute conf
        // x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        x.slice(1, 5, x.size(1)) *= x.select(1, 4).reshape({-1, 1});
        // # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        auto box = xywh2xyxy(x.slice(1, 0, 4));
        // cout << "box:" <<box.sizes()<<endl;
        std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(x.slice(1, 5, x.size(1)), 1, true); //keep_dim
        auto conf = std::get<0>(max_tuple);
        auto cls = std::get<1>(max_tuple);

        /*
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        */
        x = torch::cat({box, conf, cls}, 1).index_select(0, (conf.view(-1) > conf_thresh).nonzero().select(1, 0));
        int n = x.size(0);

        if (n == 0)
        {
            continue;
        }
        else if (n > max_nms)
        {
            x = x.index_select(0, x.select(1, 4).argsort(-1, true)).slice(0, 0, max_nms);
        }

        /*
        # Batched NMS
                c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
                boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
                i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        */
        auto c = x.slice(1, 5, 6) * max_wh;
        auto boxes = x.slice(1, 0, 4) + c;
        auto scores = x.select(1, 4);

        auto i = nms(boxes, scores, iou_thresh, DIoU);

        if (i.size(0) > max_det)
        {
            i = i.slice(0, 0, max_det);
        }
        //  output[xi] = x[i]
        output.push_back(x.index_select(0, i).clamp(0));
    }
    return 0;
}

#endif // __UTILS_HPP__