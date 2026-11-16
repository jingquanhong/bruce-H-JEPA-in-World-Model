#include "libobsensor/ObSensor.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;
void raw(const string &name,shared_ptr<ob::Frame> f) { ofstream o(name,ios::binary);o.write((char*)f->data(),f->dataSize());if(!o)throw runtime_error("save failed"); }
int main(int argc,char **argv) try {
 ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_WARN);
 ob::Context ctx;auto list=ctx.queryDeviceList();cout<<"DEVICE_COUNT "<<list->deviceCount()<<endl;
 if(!list->deviceCount())return 2;
 auto dev=list->getDevice(0);auto info=dev->getDeviceInfo();cout<<"DEVICE "<<info->name()<<" PID "<<info->pid()<<" FIRMWARE "<<info->firmwareVersion()<<endl;
 ob::Pipeline pipe(dev);shared_ptr<ob::VideoStreamProfile> dp,cp;
 for(auto type:{OB_SENSOR_DEPTH,OB_SENSOR_COLOR})try{
  auto profiles=pipe.getStreamProfileList(type);cout<<"SENSOR "<<type<<" PROFILES "<<profiles->count()<<endl;
  for(uint32_t i=0;i<profiles->count();i++){
   auto v=profiles->getProfile(i)->as<ob::VideoStreamProfile>();cout<<"PROFILE "<<type<<" "<<v->width()<<"x"<<v->height()<<" FPS "<<v->fps()<<" FORMAT "<<v->format()<<endl;
  }
  if(type==OB_SENSOR_DEPTH){
   dp=profiles->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
   try{dp=profiles->getVideoStreamProfile(640,400,OB_FORMAT_Y16,30);}catch(ob::Error&){}
  }else{
   cp=profiles->getProfile(OB_PROFILE_DEFAULT)->as<ob::VideoStreamProfile>();
   try{cp=profiles->getVideoStreamProfile(640,480,OB_FORMAT_RGB,30);}catch(ob::Error&){}
  }
 }catch(ob::Error&e){cerr<<"SENSOR_ERROR "<<e.getMessage()<<endl;}
 if(argc<2)return 0;
 if(!dp)return 3;
 auto conf=make_shared<ob::Config>();conf->enableStream(dp);if(cp)conf->enableStream(cp);conf->setAlignMode(ALIGN_DISABLE);
 pipe.start(conf);auto start=chrono::steady_clock::now();shared_ptr<ob::FrameSet> best;size_t bestValid=0;int depthCount=0,colorCount=0;
 while(chrono::duration<double>(chrono::steady_clock::now()-start).count()<8.){
  auto fs=pipe.waitForFrames(300);if(!fs)continue;
  auto d=fs->depthFrame();if(fs->colorFrame())colorCount++;
  if(!d)continue;depthCount++;if(depthCount<15)continue;
  auto values=(uint16_t*)d->data();size_t count=0;
  for(size_t i=0;i<d->dataSize()/2;i++)if(values[i]>0)count++;
  if(!best||count>bestValid){best=fs;bestValid=count;}
 }
 if(!best)throw runtime_error("No depth frames");
 auto depth=best->depthFrame();auto color=best->colorFrame();string out=argv[1];raw(out+"/depth.raw",depth);if(color)raw(out+"/color.raw",color);
 auto param=pipe.getCameraParam();pipe.stop();
 ofstream meta(out+"/capture.json");
 meta<<"{\n\"device\":\""<<info->name()<<"\",\"pid\":"<<info->pid()<<",\"firmware\":\""<<info->firmwareVersion()<<"\",\n";
 meta<<"\"depth_width\":"<<depth->width()<<",\"depth_height\":"<<depth->height()<<",\"depth_format\":"<<depth->format()<<",\"depth_scale_mm\":"<<depth->getValueScale()<<",\"depth_frames\":"<<depthCount<<",\"color_frames\":"<<colorCount<<",\"duration_seconds\":"<<chrono::duration<double>(chrono::steady_clock::now()-start).count()<<",\"valid_depth_pixels\":"<<bestValid<<",\n";
 if(color)meta<<"\"color_width\":"<<color->width()<<",\"color_height\":"<<color->height()<<",\"color_format\":"<<color->format()<<",\n";
 auto k=param.depthIntrinsic;meta<<"\"depth_intrinsics\":{\"fx\":"<<k.fx<<",\"fy\":"<<k.fy<<",\"cx\":"<<k.cx<<",\"cy\":"<<k.cy<<",\"width\":"<<k.width<<",\"height\":"<<k.height<<"}\n}\n";meta.close();
 try{
  ob::PointCloudFilter pc;pc.setCameraParam(param);pc.setPositionDataScaled(depth->getValueScale());pc.setCreatePointFormat(OB_FORMAT_POINT);auto points=pc.process(best);auto data=(OBPoint*)points->data();size_t n=points->dataSize()/sizeof(OBPoint),valid=0;
  for(size_t i=0;i<n;i++)if(isfinite(data[i].z)&&data[i].z>0)valid++;
  ofstream ply(out+"/cloud_m.ply");ply<<"ply\nformat ascii 1.0\ncomment coordinates in meters, SDK camera frame\nelement vertex "<<valid<<"\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
  for(size_t i=0;i<n;i++)if(isfinite(data[i].z)&&data[i].z>0)ply<<data[i].x/1000.<<" "<<data[i].y/1000.<<" "<<data[i].z/1000.<<"\n";
  cout<<"POINTS_SAVED "<<valid<<endl;
 }catch(ob::Error&e){cerr<<"POINTCLOUD_ERROR "<<e.getMessage()<<endl;}
 cout<<"CAPTURE_SAVED "<<out<<" DEPTH_FRAMES "<<depthCount<<" COLOR_FRAMES "<<colorCount<<" VALID "<<bestValid<<endl;
 return 0;
}catch(ob::Error&e){cerr<<"SDK_ERROR "<<e.getMessage()<<endl;return 1;}catch(exception&e){cerr<<e.what()<<endl;return 1;}
