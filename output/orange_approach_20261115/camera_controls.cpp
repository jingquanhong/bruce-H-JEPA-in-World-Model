#include "libobsensor/ObSensor.hpp"
#include <iostream>
int main() try {
 ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_ERROR);
 ob::Context ctx;auto list=ctx.queryDeviceList();if(!list->deviceCount())return 2;auto dev=list->getDevice(0);
 for(auto id:{OB_PROP_COLOR_EXPOSURE_INT,OB_PROP_COLOR_GAIN_INT,OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT})try{
  bool readable=dev->isPropertySupported(id,OB_PERMISSION_READ);std::cout<<"PROP "<<id<<" readable "<<readable;
  if(readable){auto r=dev->getIntPropertyRange(id);std::cout<<" current "<<r.cur<<" min "<<r.min<<" max "<<r.max<<" step "<<r.step<<" default "<<r.def;}
  std::cout<<std::endl;
 }catch(ob::Error&e){std::cout<<"PROPERTY_ERROR "<<id<<" "<<e.getMessage()<<std::endl;}
 try{std::cout<<"AUTO_EXPOSURE "<<dev->getBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL)<<std::endl;}catch(ob::Error&e){std::cout<<e.getMessage()<<std::endl;}
 return 0;
}catch(ob::Error&e){std::cerr<<e.getMessage()<<std::endl;return 1;}
