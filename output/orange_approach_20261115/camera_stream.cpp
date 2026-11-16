#include "libobsensor/ObSensor.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

volatile std::sig_atomic_t running=1;
void stop_stream(int) { running=0; }
double monotonic_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
template<class T> void field(std::ofstream &out,const T &value) {out.write((const char*)&value,sizeof(value));}
void intrinsics(std::ostream &o,OBCameraIntrinsic k) {
    o << "{\"fx\":" << k.fx << ",\"fy\":" << k.fy << ",\"cx\":" << k.cx
      << ",\"cy\":" << k.cy << ",\"width\":" << k.width << ",\"height\":" << k.height << "}";
}
int main(int argc,char **argv) try {
    if(argc!=3)throw std::runtime_error("Usage: camera_stream EXISTING_RAM_DIRECTORY SECONDS");
    std::string root=argv[1];double duration=std::stod(argv[2]);
    if(duration<1 || duration>300)throw std::runtime_error("Duration must be 1..300 seconds");
    std::signal(SIGINT,stop_stream);std::signal(SIGTERM,stop_stream);
    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
    ob::Pipeline pipe;
    auto config=std::make_shared<ob::Config>();
    config->enableStream(pipe.getStreamProfileList(OB_SENSOR_DEPTH)->getVideoStreamProfile(640,400,OB_FORMAT_Y12,30));
    config->enableStream(pipe.getStreamProfileList(OB_SENSOR_COLOR)->getVideoStreamProfile(640,480,OB_FORMAT_RGB,30));
    config->setAlignMode(ALIGN_D2C_SW_MODE);pipe.start(config);
    auto param=pipe.getCameraParam();
    std::ofstream calibration(root+"/calibration.json");calibration << std::setprecision(10);
    calibration << "{\"mode\":\"SDK software D2C\",\"input_depth_format\":12,\"hardware_frame_sync\":false,\"frame_timestamp_source\":\"oldest_host_receive\",\"frame_id_source\":\"depth_host_receive_us\",\"host_pair_skew_limit_s\":0.05,\"rgb\":";
    intrinsics(calibration,param.rgbIntrinsic);calibration << ",\"depth\":";intrinsics(calibration,param.depthIntrinsic);calibration << "}\n";calibration.close();
    uint64_t sequence=0;double start=monotonic_seconds(),published=0;
    while(running && monotonic_seconds()-start<duration) {
        auto frames=pipe.waitForFrames(200);
        if(!frames || !frames->depthFrame() || !frames->colorFrame())continue;
        sequence++;
        if(sequence==20) {
            auto device=pipe.getDevice();
            for(auto id:{OB_PROP_IR_EXPOSURE_INT,OB_PROP_IR_GAIN_INT,OB_PROP_MIN_DEPTH_INT,OB_PROP_MAX_DEPTH_INT})try {
                if(device->isPropertySupported(id,OB_PERMISSION_READ))std::cout<<"STREAM_INT_PROPERTY "<<id<<" "<<device->getIntProperty(id)<<std::endl;
            }catch(ob::Error&e){std::cout<<"STREAM_PROPERTY_ERROR "<<id<<" "<<e.getMessage()<<std::endl;}
            try{if(device->isPropertySupported(OB_PROP_LASER_BOOL,OB_PERMISSION_READ))std::cout<<"STREAM_LASER "<<device->getBoolProperty(OB_PROP_LASER_BOOL)<<std::endl;}catch(ob::Error&e){std::cout<<"LASER_PROPERTY_ERROR "<<e.getMessage()<<std::endl;}
            std::cout<<"FRAME_HOST_TIMESTAMPS_US depth="<<frames->depthFrame()->systemTimeStampUs()<<" rgb="<<frames->colorFrame()->systemTimeStampUs()<<std::endl;
        }
        if(sequence<20)continue;
        double now=monotonic_seconds();if(now-published<.09)continue;
        auto depth=frames->depthFrame();auto rgb=frames->colorFrame();
        uint64_t depth_received=depth->systemTimeStampUs(),rgb_received=rgb->systemTimeStampUs();
        double wall_now=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        double reception_age=wall_now-std::min(depth_received,rgb_received)/1e6;
        double pair_skew=std::abs(double(depth_received)-double(rgb_received))/1e6;
        // These are host reception times, not synchronized exposure times.
        // Include SDK queue/processing age and reject badly separated pairs.
        if(!depth_received || !rgb_received || reception_age<-.01 || reception_age>.2 || pair_skew>.05)continue;
        double stamp=monotonic_seconds()-std::max(0.,reception_age);
        uint32_t width=rgb->width(),height=rgb->height();float scale=depth->getValueScale();
        if(depth->width()!=width || depth->height()!=height || width!=640 || height!=480
            || depth->format()!=OB_FORMAT_Y16 || rgb->format()!=OB_FORMAT_RGB
            || depth->dataSize()!=width*height*2 || rgb->dataSize()!=width*height*3)
            throw std::runtime_error("Unexpected aligned frame dimensions or formats");
        std::string temporary=root+"/live-frame.new",destination=root+"/live-frame.bin";
        std::ofstream out(temporary,std::ios::binary);
        // Header is little-endian <QdIIf on this ARM64 device, exactly 28 bytes.
        field(out,depth_received);field(out,stamp);field(out,width);field(out,height);field(out,scale);
        out.write((const char*)rgb->data(),rgb->dataSize());out.write((const char*)depth->data(),depth->dataSize());
        out.close();if(!out || std::rename(temporary.c_str(),destination.c_str()))throw std::runtime_error("Frame publication failed");
        published=now;
    }
    pipe.stop();std::cout << "STREAM_FINISHED frames=" << sequence << std::endl;return 0;
} catch(ob::Error &e) {std::cerr << "SDK_ERROR " << e.getMessage() << std::endl;return 1;}
catch(std::exception &e) {std::cerr << e.what() << std::endl;return 1;}
