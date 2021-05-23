#include "main.h"
// #include "utils.hpp"
// #include <string>
// #include <iostream>
#include "detector.h"
using namespace std;
// #define Print(str) cout<<#str<<"="<<str<<endl
int main(int argc, char *argv[])
{
#if 1
       if (argc < 6) {
           printf("Usage:argv[0] netpath imgpath savepath w h\n");
           exit(-1);
       }

    string netpath = argv[1];
    string imgdir = argv[2];
    string savedir = argv[3];

    int w = stoi(argv[4]);
    int h = stoi(argv[5]);
    int i;

    time_t start, end;
    clock_t start2, end2;
    double t;
    // Detector detector(netpath, 1, 3, w, h, 0.25, 0.45, true);
    Detector detector(netpath, 1, 3, w, h, true);
    // detector.set_diou(true);
    std::vector<cv::String> imgpaths;
    cv::glob(imgdir, imgpaths, false);
    cv::Mat img, img0;
    std::vector<std::string>::iterator imgpaths_it;
    start = time(NULL);
    start2 = clock();
    string imgname, savepath;
    for (imgpaths_it = imgpaths.begin(); imgpaths_it != imgpaths.end(); ++imgpaths_it)
    {
        std::vector<torch::Tensor> output;

        auto imgpath = *imgpaths_it;
        Print(imgpath);

        detector.load_image(imgpath, img, img0);
        detector.forward(img);
        output = detector.output;
            
        int sepid;
        if ((sepid=imgpath.rfind("\\"))!=string::npos)
        {
            imgname = imgpath.substr(sepid + 1, imgpath.rfind("."));
        } else
        {
            imgname = imgpath.substr(imgpath.rfind("/") + 1, imgpath.rfind("."));
        }

        savepath = savedir + "/" + imgname;
        detector.draw(img0, savepath);
    }
    end = time(NULL);
    end2 = clock();
    t = difftime(end, start);
    printf("time:%lf\n", t);
    cout << "time2:" << (end2 - start2) / CLOCKS_PER_SEC << endl;

#endif

    return 0;
}
