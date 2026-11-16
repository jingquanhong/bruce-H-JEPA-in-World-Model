#include "libobsensor/ObSensor.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace std;
void intrinsic(ostream &o, OBCameraIntrinsic k) {
    o << "{\"fx\":" << k.fx << ",\"fy\":" << k.fy << ",\"cx\":" << k.cx
      << ",\"cy\":" << k.cy << ",\"width\":" << k.width << ",\"height\":" << k.height << "}";
}
void distortion(ostream &o, OBCameraDistortion d) {
    o << "[" << d.k1 << "," << d.k2 << "," << d.p1 << "," << d.p2 << "," << d.k3
      << "," << d.k4 << "," << d.k5 << "," << d.k6 << "]";
}
void calibration(ostream &o, OBCameraParam p) {
    o << "{\"depth\":"; intrinsic(o,p.depthIntrinsic);
    o << ",\"rgb\":"; intrinsic(o,p.rgbIntrinsic);
    o << ",\"depth_distortion\":"; distortion(o,p.depthDistortion);
    o << ",\"rgb_distortion\":"; distortion(o,p.rgbDistortion);
    o << ",\"depth_to_rgb_rotation\":[";
    for(int i=0;i<9;i++)o << (i?",":"") << p.transform.rot[i];
    o << "],\"depth_to_rgb_translation_mm\":[";
    for(int i=0;i<3;i++)o << (i?",":"") << p.transform.trans[i];
    o << "],\"is_mirrored\":" << (p.isMirrored?"true":"false") << "}";
}
void save_raw(const string &name, shared_ptr<ob::Frame> f) {
    ofstream o(name,ios::binary); o.write((const char*)f->data(),f->dataSize());
    if(!o)throw runtime_error("Cannot save "+name);
}

int main(int argc,char **argv) try {
    if(argc!=3)throw runtime_error("Usage: target_capture EXISTING_NEW_DIRECTORY raw|sw|filter");
    string out=argv[1], mode=argv[2];
    bool use_y12=mode=="sw12";
    if(use_y12)mode="sw";
    if(mode!="raw" && mode!="sw" && mode!="filter")throw runtime_error("Unknown alignment mode");
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
    ob::Pipeline pipe;
    auto dp=pipe.getStreamProfileList(OB_SENSOR_DEPTH)->getVideoStreamProfile(640,400,use_y12?OB_FORMAT_Y12:OB_FORMAT_Y11,30);
    auto cp=pipe.getStreamProfileList(OB_SENSOR_COLOR)->getVideoStreamProfile(640,480,OB_FORMAT_RGB,30);
    auto cfg=make_shared<ob::Config>();cfg->enableStream(dp);cfg->enableStream(cp);
    cfg->setAlignMode(mode=="sw"?ALIGN_D2C_SW_MODE:ALIGN_DISABLE);
    pipe.start(cfg);
    OBCameraParam param=pipe.getCameraParam();
    shared_ptr<ob::Align> align;
    if(mode=="filter")align=make_shared<ob::Align>(OB_STREAM_COLOR);
    vector<shared_ptr<ob::FrameSet>> samples;
    vector<long long> received;
    auto start=chrono::steady_clock::now();int frames=0;
    while(samples.size()<16 && chrono::duration<double>(chrono::steady_clock::now()-start).count()<8.) {
        auto fs=pipe.waitForFrames(200);
        if(!fs || !fs->depthFrame() || !fs->colorFrame())continue;
        frames++;
        if(frames<30 || frames%5!=0)continue;
        if(align)fs=align->process(fs)->as<ob::FrameSet>();
        if(!fs->depthFrame() || !fs->colorFrame())continue;
        samples.push_back(fs);
        received.push_back(chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now().time_since_epoch()).count());
    }
    pipe.stop();
    if(samples.size()<8)throw runtime_error("Insufficient RGB-D frames");
    ofstream meta(out+"/capture.json");meta << setprecision(10);
    meta << "{\"mode\":\"" << mode << "\",\"input_depth_format\":" << (use_y12?12:11)
         << ",\"alignment_requested\":" << (mode=="raw"?"false":"true")
         << ",\"hardware_frame_sync\":false,\"camera_param\":";
    calibration(meta,param);meta << ",\"samples\":[";
    for(size_t i=0;i<samples.size();i++) {
        auto d=samples[i]->depthFrame();auto c=samples[i]->colorFrame();
        if(d->format()!=OB_FORMAT_Y16 || c->format()!=OB_FORMAT_RGB)
            throw runtime_error("Unexpected output pixel format");
        string prefix=out+"/frame-"+to_string(i);
        save_raw(prefix+"-depth.raw",d);save_raw(prefix+"-rgb.raw",c);
        if(i)meta << ",";
        meta << "{\"index\":" << i << ",\"depth_width\":" << d->width() << ",\"depth_height\":" << d->height()
             << ",\"rgb_width\":" << c->width() << ",\"rgb_height\":" << c->height()
             << ",\"depth_scale_mm\":" << d->getValueScale()
             << ",\"depth_timestamp_us\":" << d->timeStampUs() << ",\"rgb_timestamp_us\":" << c->timeStampUs()
             << ",\"host_monotonic_ns\":" << received[i] << "}";
    }
    meta << "]}\n";meta.close();
    cout << "SAVED " << samples.size() << " RGB-D samples mode=" << mode << endl;
    return 0;
} catch(ob::Error &e) {
    cerr << "SDK_ERROR " << e.getName() << ": " << e.getMessage() << endl;return 1;
} catch(exception &e) {cerr << e.what() << endl;return 1;}
