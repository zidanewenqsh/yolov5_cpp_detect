//
// Created by wen on 2020/6/24.
//
#define DETECTOR_EXPORTS
#include "detector.h"
#include "utils.h"
// 加载网络

void Detector::set_batchsize(int batchsize)
{
    batch_size = batchsize;
} //: batch_size(batchsize){}
void Detector::Detector::set_channelsize(int channelsize)
{
    channel_size = channelsize;
}
void Detector::set_size(int w_, int h_)
{
    w = w_;
    h = h_;
}
void Detector::set_cfdthresh(float cfdthreshold)
{
    cfd_threshold = cfdthreshold;
}
void Detector::set_nmsthresh(float nmsthreshold)
{
    nms_threshold = nmsthreshold;
}
void Detector::set_diou(bool diou)
{
    DIoU = diou;
}

int Detector::get_batchsize()
{
    return batch_size;
} //: batch_size(batchsize){}
int Detector::Detector::get_channelsize()
{
    return channel_size;
}
std::tuple<int, int> Detector::get_size()
{
    std::tuple<int, int> size = std::make_tuple(w, h);
    return size;
}
float Detector::get_cfdthresh()
{
    return cfd_threshold;
}
float Detector::get_nmsthresh()
{
    return nms_threshold;
}
bool Detector::get_diou()
{
    return DIoU;
}

// 加载图片
int Detector::load_image(const string &path, cv::Mat &img, cv::Mat &img0)
{
    img = cv::imread(path, channel_size > 1 ? 1 : 0);
    img0 = img.clone();
    int width, height;
    width = img.cols;
    height = img.rows;
    if (img.empty())
    {
        return -1;
    }
    cv::resize(img, img, cv::Size(w, h));
    return 0;
}

// 前向计算（单输出）
int Detector::module_forward(torch::Tensor &y, const torch::Tensor &x)
{
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(x);
    try
    {
        y = module.forward(inputs).toTensor();
    }
    catch (c10::Error &e)
    {
        // std::cerr << "error forwarding the model\n";
        return -1;
    }
    return 0;
}

// Mat数组转Tensor
int Detector::mat2tensor(const cv::Mat &img)
{
    if ((img.rows % batch_size) != 0)
        return -1;
    cv::Mat imgc = img.clone();
    cv::cvtColor(imgc, imgc, cv::COLOR_BGR2RGB); // BGR -> RGB
    // imgc.convertTo(imgc, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
    img_tensor = torch::from_blob(imgc.data, {batch_size, (int)(h / batch_size), w, channel_size}, torch::kByte);
    // img_tensor = torch::from_blob(imgc.data, {1, imgc.rows, imgc.cols, imgc.channels()});
    img_tensor = img_tensor.toType(torch::kFloat);
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor = img_tensor.div_(255.0);

    img_tensor = img_tensor.to(device);

    return 0;
}
// TensorMat转数组
// Detect前向
int Detector::forward(cv::Mat &img)
{
    output.clear();

    if (mat2tensor(img) < 0)
    {
        std::cerr << "mat to tensor failed";
        return -1;
    }

    if (module_forward(preds, img_tensor) < 0)
    {
        std::cerr << "error forwarding the model";
        return -1;
    }

    if ((::non_max_suppression(output, preds, cfd_threshold, nms_threshold, DIoU)) < 0)
    {
        std::cerr << "non_max_suppression failed";
        return -1;
    }

    return 0;
}

// int draw(const cv::Mat& img, const string &savepath);
// int draw(const cv::Mat& img, const int width, const int height, const string &savepath);

int Detector::draw(const cv::Mat &img, const string &savepath)
{
    cv::Mat img0 = img.clone();
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int thickness = 1;
    cv::Point origin;
    cv::Scalar color(0, 255, 0);
    cv::Point pt1, pt2;

    int i;
    int x1, y1, x2, y2;
    float conf;
    int cls;
    std::vector<torch::Tensor>::iterator output_it;
    int width = img0.size().width;
    int height = img0.size().height;
    // int w = img.size().width;
    // int h = img.size().height;

    for (output_it = output.begin(); output_it != output.end(); ++output_it)
    {
        auto det = *output_it;

        for (i = 0; i < det.size(0); i++)
        {
            // x1 = det[i][0].item().toFloat();
            // y1 = det[i][1].item().toFloat();
            // x2 = det[i][2].item().toFloat();
            // y2 = det[i][3].item().toFloat();
            x1 = (int)(det[i][0].item().toFloat() * width / w);
            y1 = (int)(det[i][1].item().toFloat() * height / h);
            x2 = (int)(det[i][2].item().toFloat() * width / w);
            y2 = (int)(det[i][3].item().toFloat() * height / h);

            conf = det[i][4].item().toFloat();
            cls = det[i][5].item().toInt();
            pt1 = cv::Point(x1, y1);
            pt2 = cv::Point(x2, y2);

            cv::rectangle(img0, pt1, pt2, color, thickness);
            origin.x = x1;
            origin.y = y1 - 10;
            cv::putText(img0, std::to_string(cls), origin, font_face, font_scale, color, thickness, 8, 0);
            // cv::rectangle(frame, cv::Rect(x1, y1, (x2 - x1), (y2 - y1)), cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imwrite(savepath, img0);
    return 0;
}

// int Detector::draw(const cv::Mat& img, const int width, const int height, const string &savepath)
// {
//     cv::Mat img0 = img.clone();
//     int font_face = cv::FONT_HERSHEY_SIMPLEX;
//     double font_scale = 0.5;
//     int thickness = 1;
//     cv::Point origin;
//     cv::Scalar color(0, 255, 0);
//     cv::Point pt1, pt2;

//     int i;
//     int x1, y1, x2, y2;
//     float conf;
//     int cls;
//     std::vector<torch::Tensor>::iterator output_it;
//     // int width = img0.size().width;
//     // int height = img0.size().height;
//     // int w = img.size().width;
//     // int h = img.size().height;

//     for (output_it = output.begin(); output_it != output.end(); ++output_it)
//     {
//         auto det = *output_it;

//         for (i = 0; i < det.size(0); i++)
//         {
//             x1 = (int)(det[i][0].item().toFloat() * width / w);
//             y1 = (int)(det[i][1].item().toFloat() * height / h);
//             x2 = (int)(det[i][2].item().toFloat() * width / w);
//             y2 = (int)(det[i][3].item().toFloat() * height / h);

//             conf = det[i][4].item().toFloat();
//             cls = det[i][5].item().toInt();
//             pt1 = cv::Point(x1, y1);
//             pt2 = cv::Point(x2, y2);

//             cv::rectangle(img0, pt1, pt2, color, thickness);
//             origin.x = x1;
//             origin.y = y1 - 10;
//             cv::putText(img0, std::to_string(cls), origin, font_face, font_scale, color, thickness, 8, 0);
//             // cv::rectangle(frame, cv::Rect(x1, y1, (x2 - x1), (y2 - y1)), cv::Scalar(0, 255, 0), 2);
//         }
//     }
//     Print(savepath);
//     cv::imwrite(savepath, img0);
//     return 0;
// }